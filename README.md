# TegraTrace

**Vulkan Graphics Validation and Diagnostics Platform**

A low-level graphics tooling platform built on Vulkan 1.2. TegraTrace renders scenes through a
configurable GPU pipeline and wraps them in integrated subsystems: frame capture, replay,
GPU profiling, regression validation, synchronization diagnostics, IBL rendering, GPU-driven
indirect rendering, and a live debug overlay.

The renderer is the subject under test — not the project itself. The same pattern used internally
in driver and graphics tools teams: a controlled rendering harness that can inspect, reproduce,
and validate its own output.

---

## Measured Performance (RTX 3060 Laptop GPU)

### PBR model scene — Leon S. Kennedy (RE4R XPS, 49 submeshes, 194,242 verts, 4K textures)

| Metric | Value |
|---|---|
| Headless FPS (300 frames) | **~979 avg** |
| CPU frame time p50 | **0.27 ms** |
| GPU frame time p50 | **0.22 ms** |
| GPU frame time p95 | **6.48 ms** |
| GPU frame time p99 | **12.24 ms** (pipeline warmup tail; eliminated by pipeline cache on warm runs) |
| Hottest submesh (pants) | **~0.21 ms avg** |
| Hottest submesh (jacket) | **~0.19 ms avg** |
| Barrier probe overhead | **< 0.01 ms** |
| Unique texture uploads | **58** (cache dedup from ~160 references) |
| Spike frames | **0** |
| GPU timestamp resolution | **1 ns/tick** |

### Simple scenes

| Metric | Value |
|---|---|
| Headless FPS (scene 0 — single cube) | **~18,724 avg** |
| Headless FPS (scene 1 — 25-cube grid) | **~17,627 avg** |
| Regression PSNR (deterministic repro) | **100.0 dB** |

---

## Subsystems

### 1. Rendering Core

Full Vulkan 1.2 render loop: instance, device, swapchain, depth-buffered render pass, graphics
pipeline, descriptor sets, UBOs, push constants (per-object model matrix), vertex and index
buffers, GPU texture upload with layout transitions, dynamic viewport/scissor, and
synchronization via semaphores and fences.

**Nsight / RenderDoc debug markers** — `VK_EXT_debug_utils` is always requested at instance
creation. Per-frame label regions are opened/closed via `vkCmdBeginDebugUtilsLabelEXT` /
`vkCmdEndDebugUtilsLabelEXT` (function pointers loaded at device creation, null-safe on hardware
that does not expose the extension). In scene 3 each submesh draw is wrapped in its own
`PBR_Model / sub:<name>` label hierarchy, making pass boundaries immediately visible in Nsight
Frame Debugger and RenderDoc's event tree.

Four built-in scenes:

- **Scene 0** — single rotating cube, 1 draw call/frame, 36 indices
- **Scene 1** — 5×5 grid of 25 independently rotating cubes, 25 draw calls/frame
- **Scene 2** — procedural UV sphere (28,800 triangles) or OBJ mesh via `--mesh`. Lambertian diffuse.
- **Scene 3** — multi-submesh PBR FBX model via `--fbx`. Cook-Torrance BRDF with proper split-sum
  cubemap IBL (8 texture bindings: albedo, normal, roughness, metallic, AO, GGX prefiltered cube,
  BRDF LUT, diffuse irradiance cube). GPU frustum culling compute pass. Orbit camera with mouse
  drag and scroll zoom.

### 2. PBR Model Loader

Loads FBX files via Assimp 5.4.3. Per-material texture resolution handles XPS export conventions
where the FBX may reference only a subset of available maps:

- **Suffix-based slot detection** — `_D`/`_ATOC` → albedo (sRGB), `_N` → normal (linear),
  `_R`/`_S` → roughness, `_M` → metallic, `_AO` → ambient occlusion
- **Base-name inference** — after the Assimp reference pass, any empty slot triggers a directory
  scan: strips the known suffix from loaded texture filenames (e.g. `Leon_Hair_N.png` → base
  `Leon_Hair`), then probes for the missing file (`Leon_Hair_D.png`, `Leon_Hair_R.png`, etc.)
- **Texture cache** — resolves duplicate references across materials to a single `VkImage`,
  eliminating redundant GPU uploads (58 unique vs ~160 total references for Leon)
- **Full mip chain generation** — every texture upload blit-generates all mip levels via
  `vkCmdBlitImage` (`VK_FILTER_LINEAR`), `mipLevels = floor(log2(max(w,h)))+1`. Sampler uses
  `VK_LOD_CLAMP_NONE` to let the hardware select optimal mip at all distances.
- **Cook-Torrance BRDF** — GGX NDF, Smith-Schlick geometry, Fresnel-Schlick; Reinhard
  tonemapping + gamma correction. Alpha discard at 0.1 for hair/transparency submeshes.
- **Tangent-space normal mapping** — TBN matrix with Gram-Schmidt re-orthogonalization in
  the vertex shader; Assimp `CalcTangentSpace` provides per-vertex tangent/bitangent.
- **IBL (split-sum, GPU compute)** — at load time generates a 256×256 GGX BRDF integration
  LUT on the CPU (512 Hammersley samples per texel), then runs three compute shaders to build
  proper cubemap resources: `env_to_cube.comp` converts the equirectangular HDR to a 512×512
  cubemap; `ibl_prefilter.comp` convolves it with GGX at 8 roughness mip levels (256×256,
  Hammersley 1024-sample importance sampling, split-sum V=N approximation); `ibl_irradiance.comp`
  integrates the hemisphere to a 32×32 diffuse irradiance cubemap (~3906 samples/texel). The
  fragment shader binds all three as `samplerCube`: `texEnvPrefiltered` (binding 6, roughness LOD),
  `texBrdfLut` (binding 7), `texIrradiance` (binding 8).
- **GPU-driven indirect rendering** — all submesh geometry is consolidated into a single VBO
  and IBO at load time. A `VkBuffer` of `VkDrawIndexedIndirectCommand` entries (one per submesh)
  is uploaded with `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`. Scene 3 records geometry buffers once
  and then issues one `vkCmdDrawIndexedIndirect` call per submesh, with draw parameters
  (index count, first index, vertex offset) residing entirely in GPU memory.
- **GPU frustum culling** — a compute shader (`cull.comp`) runs before the draw pass each frame.
  It extracts 6 frustum planes from the view-projection matrix (Gribb-Hartmann), tests per-submesh
  world-space bounding spheres (precomputed at load time with the fixed model transform applied),
  and writes `instanceCount = 0` for culled draws into a per-frame culled indirect buffer.
  A `VkMemoryBarrier` synchronizes compute writes to indirect draw reads. No CPU readback.

### 3. Frame Capture and Replay

`--capture` writes one JSON file per captured frame to `captures/`:

```json
{
  "frame": 200,
  "wall_timestamp_ms": 1234.5,
  "draw_calls": [{
    "pipeline": "pbr.vert",
    "vertex_count": 31057,
    "index_count": 93171,
    "viewport_w": 1280.0,
    "viewport_h": 720.0,
    "ubo": { "view": [...], "proj": [...], "model": [...] }
  }]
}
```

`--replay captures/frame_0200.json` restores view/proj/model matrices, re-renders
deterministically, reads back the framebuffer, and compares to a saved reference via PSNR.

### 4. GPU Profiling Layer

Two Vulkan query pools per frame (up to 64 passes tracked per frame):

- **`VK_QUERY_TYPE_TIMESTAMP`** — GPU time for each render pass. Scene 3 records a separate
  timestamp pair per submesh, giving per-draw-call GPU cost breakdown across all 49 submeshes.
  Results read back after fence wait, no pipeline stall.
- **`VK_QUERY_TYPE_PIPELINE_STATISTICS`** — VS invocations, FS invocations, IA primitives,
  clipping invocations, clipping primitives. Overdraw ratio derived as FS invocations ÷ viewport pixels.

Frame-to-frame jitter tracking flags synchronization stalls when deviation exceeds 50% of the
rolling 30-frame mean. Spike detection filters passes below 0.05 ms absolute threshold to avoid
false positives on barrier probes.

Exports to `reports/gpu_profiling.json` with per-pass averages, p50/p95/p99, spike frame indices.

### 5. Regression Validation System

`--save-ref` reads back the resolved swapchain image at frame 100 and frame 400 via a
host-visible staging buffer (`vkCmdCopyImageToBuffer` + layout transitions), saves PNG.

`--regression` captures the same frames and computes per-pixel RMSE and PSNR. Deterministic
headless rendering produces **100.0 dB PSNR** (pixel-perfect). Threshold: ≥ 40 dB = PASS.

`--multi-res` additionally tests at 50% and 25% scale. Per-pixel diff heatmaps are written
alongside each result.

### 6. VRAM Budget Monitoring

`VK_EXT_memory_budget` is optionally enabled at device creation (gracefully absent on older
drivers). `vkGetPhysicalDeviceMemoryProperties2` is called each frame with a chained
`VkPhysicalDeviceMemoryBudgetPropertiesEXT` to read per-heap budget and usage in bytes. The
debug overlay's VRAM Budget panel renders a colour-coded progress bar per heap (green below 85%,
red at or above 85%).

### 7. Pipeline Cache and Shader Hot-Reload

**`VkPipelineCache` serialization** — on startup, `loadPipelineCache("pipeline.cache")` populates
the pipeline cache from a binary file written by the previous run. All pipeline creation calls
(`vkCreateGraphicsPipelines`, `vkCreateComputePipelines`) pass this cache object; the driver
reuses compiled binaries instead of recompiling from SPIR-V. On exit, `savePipelineCache` writes
the updated binary back. This directly eliminates the p99=12.24 ms warmup spike seen on cold runs.

**Shader hot-reload** — `ShaderWatcher` polls `std::filesystem::last_write_time` on .spv files
every 60 frames. When a timestamp changes, `Pipeline::tryReload()` calls `vkDeviceWaitIdle`,
recompiles the pipeline from the new SPIR-V, swaps the `VkPipeline` handle, and destroys the old
one. Descriptor sets and `VkPipelineLayout` are unchanged. Failures leave the old pipeline intact.

### 8. GPU Frustum Culling

A pre-draw compute pass dispatched each frame:

1. Extracts 6 view-frustum planes from the combined VP matrix (Gribb-Hartmann row method).
2. Tests each submesh's world-space bounding sphere (precomputed at load time with the fixed model
   transform applied — no per-frame matrix upload needed).
3. Writes `instanceCount = 0` for culled submeshes into a per-frame `VkBuffer` that is then used
   as the indirect draw source. A `VkMemoryBarrier` (`SHADER_WRITE → INDIRECT_COMMAND_READ`)
   ensures the GPU sees the culled counts before the draw pass reads them.

### 9. VK_KHR_performance_query

`VkPhysicalDevicePerformanceQueryFeaturesKHR` is chained into device creation when the extension
is available. On initialization, all vendor-specific hardware performance counters are enumerated
via `vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR` and stored with their
category and description strings. The debug overlay's GPU Performance Counters panel lists the
available counter names by category, giving immediate visibility into what hardware-level metrics
the platform can collect (e.g., SM utilization, L2 hit rate, instruction throughput).

### 10. Debug UI (Dear ImGui)

Live overlay rendered inside the Vulkan render pass via `imgui_impl_vulkan`:

| Panel | Contents |
|---|---|
| Frame Timing | FPS, measured CPU frame time, GPU frame time, rolling graphs, barrier probe, jitter, sync stall flag, spike count |
| Pipeline Statistics | Draw calls, VS/FS invocations, IA/clip primitives, overdraw ratio |
| Pipeline Inspector | Active pipeline name |
| Validation Log | Color-coded messages (error/warning/info) with fix suggestions |
| Command Buffer Inspector | Draw calls in GPU submission order with vtx/idx counts |
| Submesh GPU Timings | Per-submesh GPU time as scaled progress bars (scene 3 only) |
| Scene Control | Switch scenes at runtime |
| Replay Controls | Scan captures dir, select frame, trigger replay, display PSNR result |
| VRAM Budget | Per-heap budget/used (MiB) as colour-coded progress bars |
| GPU Performance Counters | Available `VK_KHR_performance_query` counter names by category (shown when extension supported) |

---

## Build

**Prerequisites:** Vulkan SDK, CMake 3.20+, Visual Studio 2022 (Desktop C++ workload).

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Debug build enables validation layers:

```bat
cmake -S . -B build-debug -G "Visual Studio 17 2022" -A x64
cmake --build build-debug --config Debug
```

Dependencies fetched automatically via CMake FetchContent: GLFW 3.4, GLM 1.0.1,
nlohmann_json v3.11.3, stb (master), Dear ImGui v1.91.5, Assimp 5.4.3 (FBX + OBJ importers).

---

## Usage

```bat
build\Release\tegratrace.exe [options]
```

| Flag | Description |
|---|---|
| `--frames N` | Render N frames then exit (default: unlimited) |
| `--headless` | Offscreen render, no window |
| `--scene N` | Select scene: 0=cube, 1=25-cube grid, 2=sphere/OBJ, 3=PBR FBX |
| `--mesh path.obj` | Load OBJ for scene 2 (implies `--scene 2`) |
| `--fbx path.fbx` | Load FBX for PBR scene 3 (implies `--scene 3`) |
| `--capture` | Write per-frame JSON to `captures/` |
| `--capture-every N` | Capture every Nth frame (default: 50) |
| `--regression` | Compare frames 100 and 400 against references |
| `--save-ref` | Save frames 100 and 400 as regression references |
| `--multi-res` | Also test at 50% and 25% scale |
| `--replay path` | Replay a captured frame JSON |

---

## Output Files

```
captures/frame_NNNN.json      Per-frame draw call log, matrices, shader metadata
references/frame_NNN.png      Regression reference images (--save-ref)
reports/regression.json       PSNR, RMSE, pass/fail per test, diff heatmap paths
reports/gpu_profiling.json    Per-pass GPU timing, p50/p95/p99, pipeline statistics, spike list
metrics.json                  FPS, CPU latency percentiles, frame count, draw call totals
```

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) builds on `windows-latest` with Vulkan SDK 1.3.275.0
and uploads `tegratrace.exe` as an artifact on every push to `main`.

---

## Architecture

```
src/
  core/        VulkanContext (debug labels, VRAM budget, pipeline cache, VK_KHR_performance_query),
               Window, Swapchain
  renderer/    Buffer, RenderPass, Pipeline (tryReload), ShaderWatcher, Renderer,
               PBRModel (IBL compute, mipmaps, GPU frustum culling, GPU-driven indirect, texture cache)
  profiling/   GPUProfiler  — timestamp + pipeline stats, per-submesh timing, spike/jitter detection
  capture/     FrameCapture, FrameReplayer
  validation/  RegressionTester — readback, PSNR, multi-res, heatmap diff
  metrics/     MetricsCollector — FPS, CPU latency percentiles, JSON export
  ui/          DebugUI — 10-panel ImGui overlay
shaders/       GLSL compiled to SPIR-V by CMake/glslc:
               pbr.vert/frag, cull.comp, env_to_cube.comp, ibl_prefilter.comp, ibl_irradiance.comp,
               triangle.vert/frag, mesh.vert/frag, cube_field.vert
```
