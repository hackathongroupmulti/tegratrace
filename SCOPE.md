# TegraTrace — Scope, Status & Session Context

**Identity:** Vulkan graphics validation and diagnostics platform.
A simplified fusion of RenderDoc + NSight + graphics CI tooling.
The renderer is the subject under test — not the project itself.

---

## Current State

The project builds and runs clean on RTX 3060. Activate the PBR scene with:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\Release\tegratrace.exe --fbx path/to/model.fbx
```

### What is working right now

**Core renderer**
- Full Vulkan renderer: instance, device, swapchain, render pass with depth,
  graphics pipeline, descriptor sets, UBO transforms, command buffers,
  synchronization, resize handling

**GPU profiling infrastructure**
- `VK_QUERY_TYPE_TIMESTAMP` wired into `recordCommandBuffer` — real GPU ms numbers
- `VK_QUERY_TYPE_PIPELINE_STATISTICS` — VS/FS invocations, IA/clip primitives
- Per-submesh GPU timing via nested timestamp pairs around each draw call
- Frame spike detection, jitter tracking, sync-stall heuristic

**ImGui overlay — 10 live panels**
- Frame Timing: FPS, CPU ms, GPU ms, rolling graphs, barrier probe, sync stall indicator
- Pipeline Statistics: VS/FS/IA/clip counts, overdraw ratio
- Pipeline Inspector: active pipeline name + mesh shader active/fallback indicator
- Validation Log: severity-coloured Vulkan debug messages (error/warning/info)
- Scene Control: live scene switching (cube / 5×5 grid)
- VRAM Budget: per-heap usage bar (VK_EXT_memory_budget, red above 85%)
- Command Buffer Inspector: per-draw submission order
- Submesh GPU Timings: per-submesh ms bars
- Replay Controls: scan captures/, scrub and replay, PSNR display
- GPU Performance Counters: VK_KHR_performance_query counter enumeration

**PBR rendering (scene 3)**
- Cook-Torrance BRDF: GGX NDF, Smith masking, Schlick Fresnel
- Full IBL: equirect HDR → cubemap → GGX specular prefilter + diffuse irradiance + BRDF LUT
- Multi-submesh FBX loading via Assimp with per-material texture deduplication
- Full mip chains on all textures (stb_image + Vulkan blit chain)

**Feature 1 — Async compute (VK_KHR_timeline_semaphore)**
- Dedicated compute queue (separate family from graphics on supported hardware)
- GPU frustum culling runs on compute queue with timeline semaphore synchronisation
- Graphics queue waits on compute timeline value before consuming cull output
- `VkDrawIndexedIndirectCommand` buffer produced by compute, consumed by graphics

**Feature 2 — Bindless textures (VK_EXT_descriptor_indexing)**
- `sampler2D tex2D[256]` runtime array in fragment shader
- `PARTIALLY_BOUND` + `UPDATE_AFTER_BIND` flags on descriptor set layout
- All material textures (albedo, normal, roughness, metallic, AO) + BRDF LUT
  packed into one array; per-submesh indices pushed via push constants
- One descriptor set per frame-in-flight (not per submesh) — O(1) bind cost

**Feature 3 — Ray-traced shadows (VK_KHR_ray_query)**
- BLAS built from consolidated geometry (triangle geometry, device-local)
- TLAS with one instance per model (identity transform configurable)
- Inline shadow ray in fragment shader: `rayQueryEXT` + `rayQueryInitializeEXT`
  toward directional light, hard shadow with 8% ambient retention in shadow
- `rtEnabled` push constant gates the ray query — zero cost when TLAS absent

**Feature 4 — Mesh shaders (VK_EXT_mesh_shader)**
- Task shader (32 threads/group): per-meshlet frustum cull via MVP sphere test,
  surviving indices written to `taskPayloadSharedEXT`, `EmitMeshTasksEXT`
- Mesh shader (32 threads/group): unpacks vertices from flat `float[]` SSBO
  (14 floats/vertex: pos+normal+uv+tangent+bitangent), reconstructs TBN,
  emits `gl_PrimitiveTriangleIndicesEXT` — no vertex input state
- Greedy meshlet packer: 64 verts / 124 tris per meshlet, AABB-center + tight
  radius (max actual vertex distance from center, not AABB half-diagonal)
- One `vkCmdDrawMeshTasksEXT` per submesh; push constants carry meshletOffset
  and meshletCount to delimit each submesh's meshlet range in the task shader
- Separate 9-binding descriptor set layout (vs PBR's 5): adds SSBOs for
  meshlet data, meshlet vertex refs, meshlet triangle indices, consolidated VBO
- Falls back silently to vertex pipeline when hardware lacks mesh shader support

**Other infrastructure**
- Pipeline cache (eliminates warmup spikes across runs)
- Shader hot-reload via `ShaderWatcher` — F5 reloads all pipelines including mesh
- VK_EXT_memory_budget: per-heap VRAM budget queried per frame
- VK_KHR_performance_query: hardware counter enumeration displayed in UI
- Nsight/RenderDoc debug labels on all passes (VK_EXT_debug_utils)
- Frame capture: `--capture` writes per-frame JSON to `captures/`
- Regression testing: `--save-ref` + `--regression` computes PSNR vs PNG reference
- MetricsCollector: p50/p95/p99 latency percentiles, JSON export on exit

---

## Architecture

```
src/
  core/        VulkanContext, Window, Swapchain
  renderer/    Shader, Buffer, RenderPass, Pipeline, Renderer, PBRModel
  profiling/   GPUProfiler  — timestamp + pipeline stats, spike detection
  capture/     FrameCapture — per-frame JSON; FrameReplayer
  validation/  RegressionTester — readback, PSNR, report
  metrics/     MetricsCollector — FPS, latency percentiles
  ui/          DebugUI — 10 ImGui panels
shaders/       pbr.vert/frag, pbr.task, pbr.mesh, cull.comp,
               env_to_cube/ibl_prefilter/ibl_irradiance.comp
```

---

## Build Instructions

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\Release\tegratrace.exe --frames 999999
```

Flags: `--capture`, `--regression`, `--save-ref`, `--frames N`, `--capture-every N`,
       `--fbx path.fbx`, `--mesh path.obj`, `--scene N`

---

## Resume Bullets

```latex
\resumeItem{Built a Vulkan renderer and GPU diagnostics platform implementing
VK\_EXT\_mesh\_shader (task/mesh pipeline with greedy meshlet packing and
per-meshlet frustum culling), VK\_KHR\_ray\_query (inline shadow rays via BLAS/TLAS),
VK\_EXT\_descriptor\_indexing (bindless sampler2D[256] heap), and async compute
with VK\_KHR\_timeline\_semaphore — all active simultaneously on an RTX 3060}

\resumeItem{Implemented a real-time GPU diagnostics overlay (Dear ImGui) driven
by VK\_QUERY\_TYPE\_TIMESTAMP and VK\_QUERY\_TYPE\_PIPELINE\_STATISTICS, with per-submesh
GPU timing bars, VRAM budget tracking (VK\_EXT\_memory\_budget), frame spike
detection, and PSNR-based screenshot regression testing across render scenes}
```
