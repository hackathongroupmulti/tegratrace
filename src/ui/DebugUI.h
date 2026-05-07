#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <deque>
#include <cstdint>
#include <functional>

struct GLFWwindow;

namespace tgt {

class VulkanContext;

struct UIFrameData {
    float       fps             = 0.0f;
    float       cpuFrameMs      = 0.0f;
    float       gpuFrameMs      = 0.0f;
    uint32_t    drawCalls       = 0;
    uint64_t    vsInvocations   = 0;
    uint64_t    fsInvocations   = 0;
    uint64_t    iaPrimitives    = 0;
    uint64_t    clippingPrims   = 0;
    std::string pipelineName;
};

class DebugUI {
public:
    using SceneCallback = std::function<void(int)>;

    DebugUI(VulkanContext& ctx, GLFWwindow* window,
            VkRenderPass renderPass, uint32_t imageCount,
            VkCommandPool cmdPool);
    ~DebugUI();

    DebugUI(const DebugUI&) = delete;
    DebugUI& operator=(const DebugUI&) = delete;

    void newFrame();
    void render(VkCommandBuffer cmd, const UIFrameData& data);

    void setSceneCallback(SceneCallback cb) { m_sceneCallback = std::move(cb); }

private:
    void buildPanels(const UIFrameData& data);

    VulkanContext&   m_ctx;
    VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;

    static constexpr int kHistorySize = 128;
    std::deque<float> m_gpuHistory;
    std::deque<float> m_cpuHistory;

    SceneCallback m_sceneCallback;
    int           m_activeScene = 0;
};

} // namespace tgt
