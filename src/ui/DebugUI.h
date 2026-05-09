#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <deque>
#include <vector>
#include <cstdint>
#include <functional>

struct GLFWwindow;

namespace tgt {

class VulkanContext;

// Lightweight per-draw summary for the command buffer inspector panel
struct DrawCallSummary {
    uint32_t    index;
    std::string pipeline;
    uint32_t    vertexCount;
    uint32_t    indexCount;
};

struct UIFrameData {
    float       fps             = 0.0f;
    float       cpuFrameMs      = 0.0f;
    float       gpuFrameMs      = 0.0f;
    float       barrierMs       = 0.0f;
    float       jitterMs        = 0.0f;
    bool        syncSuspected   = false;
    uint32_t    spikeCount      = 0;
    uint32_t    drawCalls       = 0;
    uint64_t    vsInvocations   = 0;
    uint64_t    fsInvocations   = 0;
    uint64_t    iaPrimitives    = 0;
    uint64_t    clippingPrims   = 0;
    float       overdrawRatio   = 0.0f;
    std::string pipelineName;
    std::vector<DrawCallSummary> drawCallList;

    // Per-submesh GPU timings (scene 3 only)
    struct SubmeshTiming { std::string name; float gpuMs = 0.0f; };
    std::vector<SubmeshTiming> submeshTimings;

    // VK_EXT_memory_budget: per-heap VRAM budget and usage (MiB)
    struct HeapBudget { float budgetMiB = 0.0f; float usedMiB = 0.0f; };
    std::vector<HeapBudget> vramHeaps;
};

class DebugUI {
public:
    using SceneCallback  = std::function<void(int)>;
    using ReplayCallback = std::function<void(const std::string& path)>;

    DebugUI(VulkanContext& ctx, GLFWwindow* window,
            VkRenderPass renderPass, uint32_t imageCount,
            VkCommandPool cmdPool);
    ~DebugUI();

    DebugUI(const DebugUI&) = delete;
    DebugUI& operator=(const DebugUI&) = delete;

    void newFrame();
    void render(VkCommandBuffer cmd, const UIFrameData& data);

    void setSceneCallback(SceneCallback cb)   { m_sceneCallback  = std::move(cb); }
    void setReplayCallback(ReplayCallback cb) { m_replayCallback = std::move(cb); }
    void setCapturesDir(const std::string& d) { m_capturesDir = d; scanCaptureFiles(); }
    void setLastReplayResult(double psnr, bool passed) {
        m_lastPsnr = psnr; m_replayPassed = passed; m_replayResultValid = true;
    }

private:
    void buildPanels(const UIFrameData& data);
    void panelFrameTiming(const UIFrameData& data);
    void panelPipelineStats(const UIFrameData& data);
    void panelPipelineInspector(const UIFrameData& data);
    void panelValidationLog();
    void panelSceneControl();
    void panelCommandBufferInspector(const UIFrameData& data);
    void panelReplayControls();
    void panelSubmeshTimings(const UIFrameData& data);
    void panelVRAMBudget(const UIFrameData& data);
    void scanCaptureFiles();

    VulkanContext&   m_ctx;
    VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;

    static constexpr int kHistorySize = 128;
    std::deque<float> m_gpuHistory;
    std::deque<float> m_cpuHistory;

    SceneCallback  m_sceneCallback;
    ReplayCallback m_replayCallback;
    int            m_activeScene = 0;

    // Replay panel state
    std::string              m_capturesDir;
    std::vector<std::string> m_captureFiles;
    int                      m_selectedCapture = 0;
    double                   m_lastPsnr        = 0.0;
    bool                     m_replayPassed    = false;
    bool                     m_replayResultValid = false;
};

} // namespace tgt
