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
#include <fstream>
#include <chrono>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

struct AppConfig {
    uint32_t    targetFrames   = 500;
    bool        captureEnabled = false;
    bool        regression     = false;
    bool        saveRef        = false;
    bool        headless       = false;
    uint32_t    captureEvery   = 50;
    int         scene          = 0;
    std::string replayPath;
    std::string exeDir         = ".";
};

AppConfig parseArgs(int argc, char** argv) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--frames")        && i+1 < argc) cfg.targetFrames   = std::stoul(argv[++i]);
        else if (!strcmp(argv[i], "--capture"))                      cfg.captureEnabled = true;
        else if (!strcmp(argv[i], "--regression"))                   cfg.regression     = true;
        else if (!strcmp(argv[i], "--save-ref"))                     cfg.saveRef        = true;
        else if (!strcmp(argv[i], "--headless"))                     cfg.headless       = true;
        else if (!strcmp(argv[i], "--capture-every") && i+1 < argc) cfg.captureEvery   = std::stoul(argv[++i]);
        else if (!strcmp(argv[i], "--scene")         && i+1 < argc) cfg.scene           = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--replay")        && i+1 < argc) cfg.replayPath       = argv[++i];
    }
    return cfg;
}

int main(int argc, char** argv) {
    auto cfg = parseArgs(argc, argv);
    cfg.exeDir = fs::path(argv[0]).parent_path().string();
    if (cfg.exeDir.empty()) cfg.exeDir = ".";
    if (!cfg.replayPath.empty()) cfg.headless = true;  // replay is always headless
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
        // Windowed: create a GLFW window first (needed before surface creation)
        std::unique_ptr<tgt::Window> window;
        if (!cfg.headless)
            window = std::make_unique<tgt::Window>(1280, 720, "TegraTrace");

        tgt::VulkanContext ctx(validation, cfg.headless);

        // Headless: offscreen path — no surface, no swapchain, graphics queue only
        // Windowed: create surface, then pick device with present support
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (!cfg.headless)
            surface = window->createSurface(ctx.instance());
        ctx.initSurface(surface);  // VK_NULL_HANDLE → headless device setup

        // Inner scope: Vulkan objects destroyed before surface and instance
        {
            tgt::Swapchain   swapchain(ctx, window.get(), surface);
            tgt::RenderPass  renderPass(ctx, swapchain);

            tgt::PipelineConfig pipelineCfg{};
            pipelineCfg.vertSpvPath = path("shaders/triangle.vert.spv");
            pipelineCfg.fragSpvPath = path("shaders/triangle.frag.spv");

            tgt::Pipeline    pipeline(ctx, renderPass, swapchain.extent(), pipelineCfg);
            tgt::Renderer    renderer(ctx, swapchain, renderPass, pipeline);

            tgt::GPUProfiler      profiler(ctx, tgt::Swapchain::kMaxFramesInFlight);
            renderer.setProfiler(&profiler);

            // Scene 1: 8×8 cube field using cube_field.vert (instanced)
            std::unique_ptr<tgt::Pipeline> scenePipeline;
            if (cfg.scene == 1) {
                tgt::PipelineConfig sceneCfg{};
                sceneCfg.vertSpvPath = path("shaders/cube_field.vert.spv");
                sceneCfg.fragSpvPath = path("shaders/triangle.frag.spv");
                scenePipeline = std::make_unique<tgt::Pipeline>(
                    ctx, renderPass, swapchain.extent(), sceneCfg);
                renderer.setScene(1, scenePipeline.get(), 64);
            }

            std::unique_ptr<tgt::DebugUI> ui;
            if (!cfg.headless)
                ui = std::make_unique<tgt::DebugUI>(ctx, window->handle(), renderPass.handle(),
                                                     swapchain.imageCount(), renderer.commandPool());

            tgt::MetricsCollector metrics;
            tgt::FrameCapture     capture(path("captures"));
            tgt::RegressionTester regression(ctx, swapchain,
                                              path("references"), path("reports"));

            uint32_t frameNumber = 0;

            if (!cfg.headless) {
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
                        ui->render(cmd, data);
                    });
            }

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
                        dc.indexCount    = d.indexCount;
                        dc.firstVertex   = d.firstVertex;
                        dc.pipeline      = d.pipeline;
                        dc.vertShader    = d.vertShader;
                        dc.fragShader    = d.fragShader;
                        dc.viewportW     = d.viewportW;
                        dc.viewportH     = d.viewportH;
                        std::memcpy(dc.model, d.model, sizeof(dc.model));
                        std::memcpy(dc.view,  d.view,  sizeof(dc.view));
                        std::memcpy(dc.proj,  d.proj,  sizeof(dc.proj));
                        cf.drawCalls.push_back(dc);
                    }
                    capture.recordFrame(cf);
                });

            if (!cfg.replayPath.empty()) {
                // --- Frame replay path ---
                using json = nlohmann::json;
                std::ifstream jf(cfg.replayPath);
                if (!jf) throw std::runtime_error("Cannot open replay file: " + cfg.replayPath);
                json doc; jf >> doc;

                uint32_t capturedFrameIdx = doc.value("frame", 0u);
                std::string testName = "frame_" + std::to_string(capturedFrameIdx);

                // Load UBO from first draw call
                tgt::UniformBufferObject ubo{};
                if (doc.contains("draw_calls") && !doc["draw_calls"].empty()) {
                    auto& dc = doc["draw_calls"][0];
                    if (dc.contains("ubo")) {
                        auto& u = dc["ubo"];
                        auto loadMat = [&](const char* key, float* dst) {
                            if (u.contains(key)) {
                                auto& arr = u[key];
                                for (int i = 0; i < 16 && i < (int)arr.size(); ++i)
                                    dst[i] = arr[i].get<float>();
                            }
                        };
                        loadMat("model", ubo.model);
                        loadMat("view",  ubo.view);
                        loadMat("proj",  ubo.proj);
                    }
                }
                renderer.setUBOOverride(ubo);

                // Warm up (fill the in-flight pipeline), then capture on the last frame
                for (uint32_t f = 0; f < 4; ++f) {
                    ctx.setCurrentFrame(f);
                    renderer.drawFrame(f);
                }
                renderer.waitIdle();

                auto result = regression.captureAndTest(
                    testName, renderer.lastImageIndex(), renderer.commandPool());
                renderer.clearUBOOverride();

                std::cout << "[Replay] " << cfg.replayPath << "\n"
                          << "  Test  : " << testName << "\n"
                          << "  PSNR  : " << result.psnrDb << " dB\n"
                          << "  Result: " << (result.passed ? "PASS" : "FAIL") << "\n";
            } else {
                // --- Normal render loop ---
                std::cout << "[TegraTrace] Rendering " << cfg.targetFrames << " frames"
                          << (cfg.headless ? " (headless)...\n" : "...\n");

                while (frameNumber < cfg.targetFrames) {
                    if (!cfg.headless) {
                        if (window->shouldClose()) break;
                        window->pollEvents();
                        if (window->wasResized()) {
                            window->clearResized();
                            renderer.handleResize();
                            continue;
                        }
                    }

                    ctx.setCurrentFrame(frameNumber);
                    metrics.beginFrame();
                    if (!cfg.headless) ui->newFrame();

                    uint32_t renderedFrame = frameNumber;
                    bool ok = renderer.drawFrame(frameNumber);
                    frameNumber++;
                    if (!ok) continue;

                    metrics.endFrame(frameNumber, 0.0, 1, 36);

                    if (cfg.regression && (renderedFrame == 100 || renderedFrame == 400)) {
                        renderer.waitIdle();
                        std::string testName = "frame_" + std::to_string(renderedFrame);
                        if (cfg.saveRef)
                            regression.saveReference(testName, renderer.lastImageIndex(), renderer.commandPool());
                        else
                            regression.captureAndTest(testName, renderer.lastImageIndex(), renderer.commandPool());
                    }
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

        } // swapchain, renderPass, pipeline, renderer, profiler, regression destroyed here

        if (surface != VK_NULL_HANDLE)
            window->destroySurface(ctx.instance(), surface);

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
