#include "FrameReplayer.h"
#include "renderer/Renderer.h"
#include "validation/RegressionTester.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <glm/glm.hpp>

namespace tgt {

using json = nlohmann::json;

FrameReplayer::FrameReplayer(VulkanContext& ctx, Swapchain& swapchain, RenderPass& renderPass,
                              Pipeline& pipeline, Renderer& renderer, RegressionTester& regression)
    : m_ctx(ctx), m_swapchain(swapchain), m_renderPass(renderPass),
      m_pipeline(pipeline), m_renderer(renderer), m_regression(regression)
{}

ImageDiffResult FrameReplayer::replay(const std::string& captureJsonPath, VkCommandPool cmdPool) {
    ImageDiffResult failure{};
    failure.testName = captureJsonPath;

    std::ifstream jf(captureJsonPath);
    if (!jf) {
        std::cerr << "[FrameReplayer] Cannot open: " << captureJsonPath << "\n";
        return failure;
    }

    json doc;
    try {
        jf >> doc;
    } catch (const std::exception& e) {
        std::cerr << "[FrameReplayer] JSON parse error: " << e.what() << "\n";
        return failure;
    }

    uint32_t frameIndex = doc.value("frame", 0u);

    // Build ReplayFrameData
    ReplayFrameData replayData{};

    // Default identity matrix (column-major)
    float identMat[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    std::memcpy(replayData.view, identMat, 64);
    std::memcpy(replayData.proj, identMat, 64);

    if (doc.contains("draw_calls") && !doc["draw_calls"].empty()) {
        // Extract view and proj from the first draw call's UBO
        auto& firstDc = doc["draw_calls"][0];
        if (firstDc.contains("ubo")) {
            auto& u = firstDc["ubo"];
            auto loadMat = [&](const char* key, float* dst) {
                if (u.contains(key)) {
                    auto& arr = u[key];
                    for (int i = 0; i < 16 && i < (int)arr.size(); ++i)
                        dst[i] = arr[i].get<float>();
                }
            };
            loadMat("view", replayData.view);
            loadMat("proj", replayData.proj);
        }

        // Extract per-draw model matrices
        for (auto& dc : doc["draw_calls"]) {
            ReplayDrawCall rd{};
            rd.indexCount = dc.value("index_count", 36u);
            std::memcpy(rd.model, identMat, 64);

            if (dc.contains("ubo") && dc["ubo"].contains("model")) {
                auto& arr = dc["ubo"]["model"];
                for (int i = 0; i < 16 && i < (int)arr.size(); ++i)
                    rd.model[i] = arr[i].get<float>();
            }
            replayData.draws.push_back(rd);
        }
    }

    // Set replay data and render
    m_renderer.setReplayData(&replayData);

    // Warm-up frames to fill the in-flight pipeline
    for (uint32_t f = 0; f < 4; ++f) {
        m_renderer.drawFrame(f);
    }
    m_renderer.waitIdle();

    // Use RegressionTester to read back, save PNG, and optionally compare
    std::string testName = "replay_" + std::to_string(frameIndex);
    ImageDiffResult result = m_regression.captureAndTest(
        testName,
        m_renderer.lastImageIndex(),
        cmdPool);

    // Restore normal rendering mode
    m_renderer.setReplayData(nullptr);

    std::cout << "[FrameReplayer] Replayed frame " << frameIndex << "\n"
              << "  PSNR  : " << result.psnrDb << " dB\n"
              << "  Result: " << (result.passed ? "PASS" : "FAIL (no reference or below threshold)") << "\n";

    return result;
}

} // namespace tgt
