# TegraTrace

**Vulkan Graphics Validation and Diagnostics Platform**

A low-level graphics tooling platform built on Vulkan 1.2. TegraTrace renders scenes through a
configurable GPU pipeline and wraps them in six integrated subsystems: frame capture, replay,
GPU profiling, regression validation, synchronization diagnostics, and a live debug overlay.

The renderer is the subject under test — not the project itself. The same pattern used internally
in driver and graphics tools teams: a controlled rendering harness that can inspect, reproduce,
and validate its own output.

---

## Measured Performance (RTX 3060 Laptop GPU)

| Metric | Value |
|---|---|
| Headless FPS (1-cube scene, 1000 frames) | **~15,800 avg** |
| Headless FPS (25-cube grid, 1000 frames) | **~15,300 avg** |
| GPU frame time avg | **0.01 ms** |
| GPU frame time p99 | **0.02 ms** |
| CPU frame time avg | **0.06 ms** |
| Pipeline barrier probe overhead | **< 0.01 ms** |
| Regression PSNR (deterministic repro) | **100.0 dB** |
| GPU timestamp resolution | **1 ns/tick** |
| Zero-spike frames (1000-frame run) | **1000 / 1000** |

---

## Subsystems

### 1. Rendering Core

Full Vulkan 1.2 render loop: instance, device, swapchain, depth-buffered render pass, graphics
pipeline, descriptor sets, UBOs (view/proj), push constants (per-object model matrix), vertex and
index buffers, GPU texture upload with proper layout transitions, dynamic viewport/scissor, and
synchronization via semaphores and fences.

Two built-in scenes:
- **Scene 0** — single rotating cube, 1 draw call/frame, 36 indices, 8 vertices
- **Scene 1** — 5×5 grid of 25 independently rotating cubes, 25 draw calls/frame with per-object
  push constants

### 2. Frame Capture and Replay

`--capture` writes one JSON file per captured frame to `captures/`:

```json
{
  "frame": 200,
  "wall_timestamp_ms": 1234.5,
  "draw_calls": [{
    "pipeline": "triangle.vert",
    "vert_shader": "shaders/triangle.vert.spv",
    "frag_shader": "shaders/triangle.frag.spv",
    "vertex_count": 8,
    "index_count": 36,
    "viewport_w": 1280.0,
    "viewport_h": 720.0,
    "ubo": { "view": [...], "proj": [...], "model": [...] }
  }]
}
```

`--replay captures/frame_0200.json` restores the captured view/proj/model matrices, re-renders
deterministically, reads back the framebuffer, and compares to a saved reference via PSNR.
Replay runs 4 warm-up frames to fill the in-flight pipeline before the comparison frame.

### 3. GPU Profiling Layer

Two Vulkan query pools per frame:

- **`VK_QUERY_TYPE_TIMESTAMP`** — GPU time for the main render pass and a post-render barrier
  probe. Results read back after the previous frame's fence wait, avoiding pipeline stalls.
- **`VK_QUERY_TYPE_PIPELINE_STATISTICS`** — VS invocations, FS invocations, IA primitives,
  clipping invocations, clipping primitives.

Frame-to-frame jitter tracking flags suspected synchronization stalls when deviation exceeds 50%
of the rolling 30-frame mean. Spike detection flags frames where GPU time exceeds p99 × 1.5.

Exports to `reports/gpu_profiling.json` with per-pass averages, p50/p95/p99, spike frame indices.

### 4. Regression Validation System

`--save-ref` reads back the resolved swapchain image at frame 100 and frame 400 via a
host-visible staging buffer (`vkCmdCopyImageToBuffer` with layout transitions), saves PNG with
stb_image_write.

`--regression` captures the same frames and computes per-pixel RMSE and PSNR against the stored
references. Deterministic headless rendering produces **100.0 dB PSNR** (pixel-perfect) across
runs. Threshold: ≥ 40 dB = PASS.

`--multi-res` additionally tests at 50% and 25% scale using stb_image_resize2, validating
multi-resolution image consistency. Per-pixel diff heatmaps are written alongside each result.

Exports to `reports/regression.json`.

### 5. Validation / Diagnostics Layer

`VkDebugUtilsMessengerEXT` routes all Vulkan validation layer messages into a structured
in-memory log with severity ranking (Error / Warning / Info), frame number association, and
actionable fix suggestions pattern-matched from VUID codes and message text.

Debug builds (`-DCMAKE_BUILD_TYPE=Debug`) enable `VK_LAYER_KHRONOS_validation` automatically
via the `TEGRATRACE_VALIDATION` compile definition.

### 6. Debug UI (Dear ImGui)

Live overlay panels rendered inside the Vulkan render pass via `imgui_impl_vulkan`:

| Panel | Contents |
|---|---|
| Frame Timing | FPS, CPU ms, GPU ms, rolling graphs, barrier probe ms, jitter ms, sync stall flag |
| Pipeline Statistics | VS/FS invocations, IA primitives, clipping primitives, draw call count |
| Pipeline Inspector | Active pipeline name and shader paths |
| Validation Log | Color-coded messages (red=error, yellow=warning) with fix suggestions |
| Command Buffer Inspector | Draw calls in GPU submission order: pipeline, vertex count, index count |
| Scene Control | Switch between scenes at runtime |
| Replay Controls | Scan captures dir, slider to select frame, trigger replay, show PSNR result |

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

Dependencies are fetched automatically via CMake FetchContent: GLFW 3.4, GLM 1.0.1,
nlohmann_json v3.11.3, stb (master), Dear ImGui v1.91.5.

---

## Usage

```bat
build\Release\tegratrace.exe [options]
```

| Flag | Description |
|---|---|
| `--frames N` | Render N frames then exit (default: 500) |
| `--headless` | Offscreen render, no window |
| `--scene N` | Select scene: 0=single cube, 1=5×5 grid |
| `--capture` | Write per-frame JSON to `captures/` |
| `--capture-every N` | Capture every Nth frame (default: 50) |
| `--regression` | Compare frames 100 and 400 against references |
| `--save-ref` | Save frames 100 and 400 as regression references |
| `--multi-res` | Also test at 50% and 25% scale |
| `--replay path` | Replay a captured frame JSON |

---

## Output Files

```
captures/frame_NNNN.json      Per-frame draw call log, matrices, shader metadata, pipeline stats
references/frame_NNN.png      Regression reference images (saved with --save-ref)
reports/regression.json       PSNR, RMSE, pass/fail per test, diff heatmap paths
reports/gpu_profiling.json    Per-pass GPU timing, p50/p95/p99, pipeline statistics, spike list
metrics.json                  FPS, CPU latency percentiles, frame count, draw call totals
```

---

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) builds on `windows-latest` with
Vulkan SDK 1.3.275.0 and uploads `tegratrace.exe` as an artifact on every push to `main`.

---

## Architecture

```
src/
  core/        VulkanContext, Window, Swapchain
  renderer/    Shader, Buffer, RenderPass, Pipeline, Renderer
  profiling/   GPUProfiler  — timestamp + pipeline stats, barrier probe, spike/jitter detection
  capture/     FrameCapture, FrameReplayer
  validation/  RegressionTester — readback, PSNR, multi-res, heatmap diff
  metrics/     MetricsCollector — FPS, CPU latency percentiles, JSON export
  ui/          DebugUI — 7-panel ImGui overlay
shaders/       GLSL source (triangle, mesh, cube_field) compiled to SPIR-V by CMake/glslc
```
