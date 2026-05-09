#include "core/VulkanContext.h"
#include "core/Window.h"
#include "core/Swapchain.h"
#include "renderer/RenderPass.h"
#include "renderer/Pipeline.h"
#include "renderer/Renderer.h"
#include "profiling/GPUProfiler.h"
#include "ui/DebugUI.h"
#include "capture/FrameCapture.h"
#include "capture/FrameReplayer.h"
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
    uint32_t    targetFrames   = UINT32_MAX;
    bool        captureEnabled = false;
    bool        regression     = false;
    bool        saveRef        = false;
    bool        headless       = false;
    bool        multiRes       = false;
    uint32_t    captureEvery   = 50;
    int         scene          = 0;
    std::string replayPath;
    std::string meshPath;
    std::string fbxPath;
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
        else if (!strcmp(argv[i], "--multi-res"))                    cfg.multiRes       = true;
        else if (!strcmp(argv[i], "--capture-every") && i+1 < argc) cfg.captureEvery   = std::stoul(argv[++i]);
        else if (!strcmp(argv[i], "--scene")         && i+1 < argc) cfg.scene           = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--replay")        && i+1 < argc) cfg.replayPath       = argv[++i];
        else if (!strcmp(argv[i], "--mesh")          && i+1 < argc) cfg.meshPath         = argv[++i];
        else if (!strcmp(argv[i], "--fbx")           && i+1 < argc) cfg.fbxPath          = argv[++i];
    }
    // --mesh implies --scene 2; --fbx implies --scene 3
    if (!cfg.meshPath.empty()) cfg.scene = 2;
    if (!cfg.fbxPath.empty())  cfg.scene = 3;
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

            tgt::PipelineConfig meshCfg{};
            meshCfg.vertSpvPath = path("shaders/mesh.vert.spv");
            meshCfg.fragSpvPath = path("shaders/mesh.frag.spv");

            // PBR pipeline: Cook-Torrance + 5 material maps + 2 IBL (env + BRDF LUT)
            tgt::PipelineConfig pbrCfg{};
            pbrCfg.vertSpvPath          = path("shaders/pbr.vert.spv");
            pbrCfg.fragSpvPath          = path("shaders/pbr.frag.spv");
            pbrCfg.textureBindingCount  = 7;  // albedo, normal, roughness, metallic, AO, env, brdf_lut
            pbrCfg.usePBRVertex         = true;
            pbrCfg.cullMode             = VK_CULL_MODE_NONE; // double-sided for hair/cloth

            tgt::Pipeline    pipeline(ctx, renderPass, swapchain.extent(), pipelineCfg);
            tgt::Pipeline    meshPipeline(ctx, renderPass, swapchain.extent(), meshCfg);
            tgt::Pipeline    pbrPipeline(ctx, renderPass, swapchain.extent(), pbrCfg);
            tgt::Renderer    renderer(ctx, swapchain, renderPass, pipeline);
            renderer.setScene(cfg.scene);
            renderer.setMeshPipeline(&meshPipeline);
            renderer.setPBRPipeline(&pbrPipeline);
            renderer.loadMesh(cfg.meshPath);  // always loads sphere; OBJ if --mesh given
            if (!cfg.fbxPath.empty())
                renderer.loadPBRModel(cfg.fbxPath);

            tgt::GPUProfiler      profiler(ctx, tgt::Swapchain::kMaxFramesInFlight);
            renderer.setProfiler(&profiler);

            std::unique_ptr<tgt::DebugUI> ui;
            if (!cfg.headless)
                ui = std::make_unique<tgt::DebugUI>(ctx, window->handle(), renderPass.handle(),
                                                     swapchain.imageCount(), renderer.commandPool());

            std::string pendingReplay;  // set by UI replay panel, consumed after drawFrame

            if (ui) {
                ui->setSceneCallback([&](int s) { renderer.setScene(s); });
                ui->setCapturesDir(path("captures"));

                // Replay triggered from the ImGui panel — deferred to after drawFrame
                ui->setReplayCallback([&](const std::string& replayFile) {
                    // Store path; actual replay runs after current drawFrame completes
                    // (captured by reference; pendingReplay is in the outer scope below)
                    pendingReplay = replayFile;
                });
            }

            tgt::MetricsCollector metrics;
            tgt::FrameCapture     capture(path("captures"));
            tgt::RegressionTester regression(ctx, swapchain,
                                              path("references"), path("reports"));

            uint32_t frameNumber = 0;
            float    realCpuFrameMs = 0.0f;
            auto     lastFrameStart = std::chrono::high_resolution_clock::now();

            // Orbit camera state for scene 3 (mouse drag + scroll)
            struct OrbitState {
                float azimuth   = 0.0f;
                float elevation = 0.25f;
                float radius    = 2.5f;
                double lastX = 0, lastY = 0;
                bool   dragging = false;
                bool   userMoved = false;
            } orbit;

            // Scroll callback — stored in a static so the non-capturing lambda can write it
            static float s_scrollDelta = 0.0f;
            if (window)
                glfwSetScrollCallback(window->handle(),
                    [](GLFWwindow*, double, double dy) { s_scrollDelta += (float)dy; });

            if (!cfg.headless) {
                renderer.setFrameCallback(
                    [&](uint32_t /*frame*/, VkCommandBuffer cmd, tgt::FrameDrawStats& stats) {
                        tgt::UIFrameData data;
                        data.fps           = static_cast<float>(metrics.currentFPS());
                        data.cpuFrameMs    = realCpuFrameMs;
                        auto& rep          = profiler.lastReport();
                        data.gpuFrameMs    = static_cast<float>(rep.totalGpuMs);
                        data.jitterMs      = static_cast<float>(rep.jitterMs);
                        data.syncSuspected = rep.syncSuspected;
                        data.spikeCount    = static_cast<uint32_t>(profiler.spikes().size());
                        data.vsInvocations = rep.pipelineStats.vertexShaderInvocations;
                        data.fsInvocations = rep.pipelineStats.fragmentShaderInvocations;
                        data.iaPrimitives  = rep.pipelineStats.inputAssemblyPrimitives;
                        data.clippingPrims = rep.pipelineStats.clippingPrimitives;
                        data.drawCalls    = stats.drawCalls;
                        {
                            uint64_t pixels = (uint64_t)swapchain.extent().width * swapchain.extent().height;
                            data.overdrawRatio = (pixels > 0 && rep.pipelineStats.fragmentShaderInvocations > 0)
                                ? static_cast<float>(rep.pipelineStats.fragmentShaderInvocations) / static_cast<float>(pixels)
                                : 0.0f;
                        }
                        data.pipelineName = (cfg.scene == 3) ? pbrPipeline.name()
                                          : (cfg.scene == 2) ? meshPipeline.name()
                                          : pipeline.name();
                        // VK_EXT_memory_budget: per-heap VRAM usage
                        {
                            auto budget = ctx.queryMemoryBudget();
                            if (budget.supported) {
                                for (uint32_t h = 0; h < budget.heapCount; ++h) {
                                    tgt::UIFrameData::HeapBudget hb{};
                                    hb.budgetMiB = static_cast<float>(budget.budget[h]) / (1024.0f * 1024.0f);
                                    hb.usedMiB   = static_cast<float>(budget.usage[h])  / (1024.0f * 1024.0f);
                                    data.vramHeaps.push_back(hb);
                                }
                            }
                        }
                        // Per-pass breakdown
                        for (auto& pass : rep.passes) {
                            if (pass.name == "barrier") {
                                data.barrierMs = static_cast<float>(pass.gpuTimeMs);
                            } else if (pass.name.size() > 4 && pass.name.substr(0, 4) == "sub:") {
                                tgt::UIFrameData::SubmeshTiming st;
                                st.name  = pass.name;
                                st.gpuMs = static_cast<float>(pass.gpuTimeMs);
                                data.submeshTimings.push_back(st);
                            }
                        }
                        // Per-draw command buffer list for inspector panel
                        auto& draws = renderer.lastDrawCalls();
                        for (uint32_t i = 0; i < static_cast<uint32_t>(draws.size()); ++i)
                            data.drawCallList.push_back({ i, draws[i].pipeline,
                                                          draws[i].vertexCount, draws[i].indexCount });
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
                // --- Frame replay path (CLI --replay) ---
                tgt::FrameReplayer replayer(ctx, swapchain, renderPass, pipeline,
                                             renderer, regression);
                auto result = replayer.replay(cfg.replayPath, renderer.commandPool());
                if (result.totalPixels == 0)
                    std::cerr << "[Replay] Failed to replay: " << cfg.replayPath << "\n";
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

                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        realCpuFrameMs = std::chrono::duration<float, std::milli>(now - lastFrameStart).count();
                        lastFrameStart = now;
                    }

                    ctx.setCurrentFrame(frameNumber);
                    metrics.beginFrame();
                    if (!cfg.headless) ui->newFrame();

                    // Scene 3 orbit camera input
                    if (cfg.scene == 3 && window) {
                        GLFWwindow* win = window->handle();
                        double mx, my;
                        glfwGetCursorPos(win, &mx, &my);

                        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                            if (!orbit.dragging) {
                                orbit.lastX = mx; orbit.lastY = my;
                                orbit.dragging = true;
                            }
                            float dx = static_cast<float>(mx - orbit.lastX) * 0.007f;
                            float dy = static_cast<float>(my - orbit.lastY) * 0.007f;
                            orbit.azimuth   += dx;
                            orbit.elevation  = std::clamp(orbit.elevation - dy, -1.4f, 1.4f);
                            orbit.lastX = mx; orbit.lastY = my;
                            orbit.userMoved = true;
                        } else {
                            orbit.dragging = false;
                        }

                        // Scroll to zoom
                        orbit.radius = std::clamp(orbit.radius - s_scrollDelta * 0.15f, 0.3f, 8.0f);
                        s_scrollDelta = 0.0f;

                        // Auto-rotate until user first drags
                        if (!orbit.userMoved)
                            orbit.azimuth += 0.004f;

                        renderer.setOrbitCamera(orbit.azimuth, orbit.elevation, orbit.radius);
                    }

                    uint32_t renderedFrame = frameNumber;
                    bool ok = renderer.drawFrame(frameNumber);
                    frameNumber++;
                    if (!ok) continue;

                    { auto& fs = renderer.lastFrameStats();
                      metrics.endFrame(frameNumber, 0.0, fs.drawCalls, fs.indexCount); }

                    // UI-triggered replay (deferred from within-render-pass callback)
                    if (!pendingReplay.empty()) {
                        renderer.waitIdle();
                        tgt::FrameReplayer replayer(ctx, swapchain, renderPass, pipeline,
                                                     renderer, regression);
                        auto result = replayer.replay(pendingReplay, renderer.commandPool());
                        if (ui) ui->setLastReplayResult(result.psnrDb, result.passed);
                        pendingReplay.clear();
                    }

                    if (cfg.regression && (renderedFrame == 100 || renderedFrame == 400)) {
                        renderer.waitIdle();
                        std::string testName = "frame_" + std::to_string(renderedFrame);
                        if (cfg.saveRef) {
                            if (cfg.multiRes)
                                regression.saveReferenceMultiRes(testName, renderer.lastImageIndex(), renderer.commandPool());
                            else
                                regression.saveReference(testName, renderer.lastImageIndex(), renderer.commandPool());
                        } else {
                            if (cfg.multiRes)
                                regression.captureAndTestMultiRes(testName, renderer.lastImageIndex(), renderer.commandPool());
                            else
                                regression.captureAndTest(testName, renderer.lastImageIndex(), renderer.commandPool());
                        }
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
