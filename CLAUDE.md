# TegraTrace

Low-level Vulkan graphics renderer + debugging/validation/profiling platform targeting NVIDIA graphics engineering roles.

## Prerequisites

Install in this order:

1. **Vulkan SDK** — https://vulkan.lunarg.com/sdk/home (installs glslc, validation layers, headers)
2. **CMake 3.20+** — https://cmake.org/download/
3. **Visual Studio 2022** — install "Desktop development with C++" workload

Verify: `echo %VULKAN_SDK%` should print your SDK path (e.g. `C:\VulkanSDK\1.3.xxx.0`).

## Build

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Debug build enables Vulkan validation layers:
```bat
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --config Debug
```

## Run

```bat
build\Release\tegratrace.exe
```

Flags:
- `--frames N`      render N frames then exit (default: 500)
- `--capture`       write per-frame JSON to captures/
- `--regression`    run regression tests against references/ and write reports/
- `--headless`      no window, render offscreen (requires VK_EXT_headless_surface or swapchain-less path)

## Output

After running you get:
- `captures/frame_NNNN.json`   — per-frame draw call log + GPU timings + pipeline stats
- `reports/regression.json`    — image diff results per test scene
- `metrics.json`               — aggregate FPS, latency histogram, draw call counts

## Architecture

```
src/
  core/        — VulkanContext, Window, Swapchain
  renderer/    — Shader, Buffer, RenderPass, Pipeline, Renderer
  profiling/   — GPUProfiler (timestamp queries, pipeline statistics)
  capture/     — FrameCapture (draw call logging, state serialization)
  validation/  — RegressionTester (screenshot readback, image diff, PSNR)
  metrics/     — MetricsCollector (FPS, latency, JSON export)
shaders/       — GLSL source; compiled to SPIR-V by CMake via glslc
```

## Subsystem notes

**GPUProfiler** uses `VK_QUERY_TYPE_TIMESTAMP` and `VK_QUERY_TYPE_PIPELINE_STATISTICS`. Results are read back after each frame and accumulated into rolling histograms.

**FrameCapture** hooks into `Renderer::recordFrame()` via a callback and serializes each draw call, bound pipeline, viewport, and uniform data to JSON.

**RegressionTester** reads back the resolved swapchain image via a host-visible staging buffer, saves PNG with stb_image_write, then computes per-pixel RMSE and PSNR against a reference image.

**MetricsCollector** tracks wall-clock FPS, CPU-side frame time, GPU timestamp delta, and draw call count. Writes a final JSON report on exit including p50/p95/p99 latency percentiles.
