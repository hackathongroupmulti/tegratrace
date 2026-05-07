#include "core/VulkanContext.h"
#include "core/Window.h"
#include "core/Swapchain.h"
#include "renderer/RenderPass.h"
#include "renderer/Pipeline.h"
#include "renderer/Renderer.h"
#include "profiling/GPUProfiler.h"
#include "ui/DebugUI.h"
#include "capture/FrameCapture.h"
#include "validation/RegressionTester.h"
#include "metrics/MetricsCollector.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <cstring>

namespace fs = std::filesystem;

struct AppConfig {
    uint32_t    targetFrames   = 500;
    bool        captureEnabled = false;
    bool        regression     = false;
    bool        saveRef        = false;
    uint32_t    captureEvery   = 50;
    std::string exeDir         = ".";
};

AppConfig parseArgs(int argc, char** argv) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--frames")        && i+1 < argc) cfg.targetFrames   = std::stoul(argv[++i]);
        else if (!strcmp(argv[i], "--capture"))                      cfg.captureEnabled = true;
        else if (!strcmp(argv[i], "--regression"))                   cfg.regression     = true;
        else if (!strcmp(argv[i], "--save-ref"))                     cfg.saveRef        = true;
        else if (!strcmp(argv[i], "--capture-every") && i+1 < argc) cfg.captureEvery   = std::stoul(argv[++i]);
    }
    return cfg;
}

int main(int argc, char** argv) {
    auto cfg = parseArgs(argc, argv);
    cfg.exeDir = fs::path(argv[0]).parent_path().string();
    if (cfg.exeDir.empty()) cfg.exeDir = ".";
    auto path = [&](const std::string& rel) { return cfg.exeDir + "/" + rel; };

    fs::create_directories(path("captures"));
    fs::create_directories(path("reports"));
    fs::create_directories(path("references"));

#ifdef TEGRATRACE_VALIDATION
    bool validation = true;
#else
    bool validation = false;
#endif

    try {
        tgt::Window        window(1280, 720, "TegraTrace");
        tgt::VulkanContext ctx(validation);
        auto surface = window.createSurface(ctx.instance());
        ctx.initSurface(surface);

        // Inner scope: all Vulkan objects that depend on the swapchain/surface
        // are destroyed here (in reverse order) BEFORE we destroy the surface.
        // Vulkan spec requires: swapchain destroyed before surface, surface before instance.
        {
            tgt::Swapchain   swapchain(ctx, window, surface);
            tgt::RenderPass  renderPass(ctx, swapchain);

            tgt::PipelineConfig pipelineCfg{};
            pipelineCfg.vertSpvPath = path("shaders/triangle.vert.spv");
            pipelineCfg.fragSpvPath = path("shaders/triangle.frag.spv");

            tgt::Pipeline    pipeline(ctx, renderPass, swapchain.extent(), pipelineCfg);
            tgt::Renderer    renderer(ctx, swapchain, renderPass, pipeline);

            tgt::GPUProfiler      profiler(ctx, tgt::Swapchain::kMaxFramesInFlight);
            renderer.setProfiler(&profiler);
            tgt::DebugUI          ui(ctx, window.handle(), renderPass.handle(),
                                     swapchain.imageCount(), renderer.commandPool());
            tgt::MetricsCollector metrics;
            tgt::FrameCapture     capture(path("captures"));
            tgt::RegressionTester regression(ctx, swapchain,
                                              path("references"), path("reports"));

            uint32_t frameNumber = 0;

            renderer.setFrameCallback(
                [&](uint32_t /*frame*/, VkCommandBuffer cmd, tgt::FrameDrawStats& stats) {
                    tgt::UIFrameData data;
                    data.fps           = static_cast<float>(metrics.currentFPS());
                    data.cpuFrameMs    = data.fps > 0.0f ? 1000.0f / data.fps : 0.0f;
                    auto& rep          = profiler.lastReport();
                    data.gpuFrameMs    = static_cast<float>(rep.totalGpuMs);
                    data.vsInvocations = rep.pipelineStats.vertexShaderInvocations;
                    data.fsInvocations = rep.pipelineStats.fragmentShaderInvocations;
                    data.iaPrimitives  = rep.pipelineStats.inputAssemblyPrimitives;
                    data.clippingPrims = rep.pipelineStats.clippingPrimitives;
                    data.drawCalls     = stats.drawCalls;
                    data.pipelineName  = pipeline.name();
                    ui.render(cmd, data);
                });

            renderer.setCaptureCallback(
                [&](uint32_t frame, const std::vector<tgt::DrawCallRecord>& draws) {
                    if (!cfg.captureEnabled || frame % cfg.captureEvery != 0) return;
                    tgt::CapturedFrame cf{};
                    cf.frameIndex      = frame;
                    cf.wallTimestampMs = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    for (auto& d : draws) {
                        tgt::CapturedDrawCall dc{};
                        dc.vertexCount   = d.vertexCount;
                        dc.instanceCount = d.instanceCount;
                        dc.firstVertex   = d.firstVertex;
                        dc.pipeline      = d.pipeline;
                        dc.viewportW     = d.viewportW;
                        dc.viewportH     = d.viewportH;
                        cf.drawCalls.push_back(dc);
                    }
                    capture.recordFrame(cf);
                });

            std::cout << "[TegraTrace] Rendering " << cfg.targetFrames << " frames...\n";

            while (!window.shouldClose() && frameNumber < cfg.targetFrames) {
                window.pollEvents();

                if (window.wasResized()) {
                    window.clearResized();
                    renderer.handleResize();
                    continue;
                }

                metrics.beginFrame();
                ui.newFrame();

                bool ok = renderer.drawFrame(frameNumber);
                frameNumber++;
                if (!ok) continue;

                metrics.endFrame(frameNumber, 0.0, 1, 36);

                if (cfg.regression && (frameNumber == 100 || frameNumber == 400)) {
                    renderer.waitIdle();
                    std::string testName = "frame_" + std::to_string(frameNumber);
                    if (cfg.saveRef)
                        regression.saveReference(testName, renderer.lastImageIndex(), renderer.commandPool());
                    else
                        regression.captureAndTest(testName, renderer.lastImageIndex(), renderer.commandPool());
                }
            }

            renderer.waitIdle();

            metrics.printSummary();
            profiler.printSummary();
            metrics.exportJSON(path("metrics.json"));
            profiler.exportJSON(path("reports/gpu_profiling.json"));
            if (cfg.captureEnabled) capture.flush();
            if (cfg.regression && !cfg.saveRef)
                regression.exportReport(path("reports/regression.json"));

        } // swapchain, renderPass, pipeline, renderer, profiler, regression all destroyed here

        // Surface destroyed after swapchain, before instance (ctx destructor)
        window.destroySurface(ctx.instance(), surface);

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
