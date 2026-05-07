#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <array>

namespace tgt {

class VulkanContext;
class Swapchain;
class RenderPass;
class Pipeline;
class Buffer;
class GPUProfiler;

struct FrameDrawStats {
    uint32_t drawCalls    = 0;
    uint32_t vertexCount  = 0;
    uint32_t indexCount   = 0;
};

// Per-draw-call data logged by capture system
struct DrawCallRecord {
    uint32_t    vertexCount;
    uint32_t    instanceCount;
    uint32_t    indexCount;
    uint32_t    firstVertex;
    std::string pipeline;
    std::string vertShader;
    std::string fragShader;
    float       viewportW, viewportH;
    float       model[16];
    float       view[16];
    float       proj[16];
};

using FrameCallback = std::function<void(uint32_t frame, VkCommandBuffer cmd, FrameDrawStats&)>;
using CaptureCallback = std::function<void(uint32_t frame, const std::vector<DrawCallRecord>&)>;

// UBO now only contains view and proj — model is sent via push constants
struct UniformBufferObject {
    float view[16];
    float proj[16];
};

// Replay support
struct ReplayDrawCall {
    float    model[16];
    uint32_t indexCount;
};

struct ReplayFrameData {
    float    view[16];
    float    proj[16];
    std::vector<ReplayDrawCall> draws;
};

class Renderer {
public:
    Renderer(VulkanContext& ctx, Swapchain& swapchain,
             RenderPass& renderPass, Pipeline& pipeline);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Returns false when window was resized (caller should call handleResize)
    bool drawFrame(uint32_t frameNumber);
    void handleResize();
    void waitIdle();

    void setCaptureCallback(CaptureCallback cb) { m_captureCallback = std::move(cb); }
    void setFrameCallback(FrameCallback cb)     { m_frameCallback   = std::move(cb); }
    void setProfiler(GPUProfiler* p)            { m_profiler = p; }

    void setScene(int s)                          { m_scene = s; }
    int  scene()  const                           { return m_scene; }
    void setReplayData(const ReplayFrameData* d)  { m_replayData = d; }

    VkCommandPool commandPool()     const { return m_commandPool; }
    uint32_t      lastImageIndex()  const { return m_lastImageIndex; }

private:
    void createCommandPool();
    void createCommandBuffers();
    void createDescriptorPool();
    void createUniformBuffers();
    void createDescriptorSets();
    void updateUniformBuffer(uint32_t frame, uint32_t frameNumber);
    VkCommandBuffer recordCommandBuffer(uint32_t imageIndex, uint32_t frameNumber,
                                        FrameDrawStats& stats);

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    RenderPass&    m_renderPass;
    Pipeline&      m_pipeline;

    VkCommandPool                     m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer>      m_commandBuffers;
    VkDescriptorPool                  m_descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet>      m_descriptorSets;
    std::vector<std::unique_ptr<Buffer>> m_uniformBuffers;

    // Demo geometry (colored cube)
    std::unique_ptr<Buffer> m_vertexBuffer;
    std::unique_ptr<Buffer> m_indexBuffer;
    uint32_t                m_indexCount = 0;

    CaptureCallback m_captureCallback;
    FrameCallback   m_frameCallback;
    GPUProfiler*    m_profiler       = nullptr;
    int             m_scene          = 0;
    const ReplayFrameData* m_replayData = nullptr;

    float    m_currentView[16]{};
    float    m_currentProj[16]{};

    uint32_t m_lastImageIndex  = 0;
    uint32_t m_frameCount      = 0;
};

} // namespace tgt
