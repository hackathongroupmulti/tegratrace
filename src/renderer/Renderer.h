#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <array>
#include <cmath>

namespace tgt {

class VulkanContext;
class Swapchain;
class RenderPass;
class Pipeline;
class Buffer;
class GPUProfiler;
class PBRModel;

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

    // Returns the draw call records from the most recently completed frame
    const std::vector<DrawCallRecord>& lastDrawCalls()  const { return m_lastDrawCalls; }
    const FrameDrawStats&              lastFrameStats()  const { return m_lastFrameStats; }

    void loadMesh(const std::string& objPath = "");
    void setMeshPipeline(Pipeline* p) { m_meshPipeline = p; }

    // Scene 3: PBR model (FBX via Assimp)
    void loadPBRModel(const std::string& fbxPath);
    void setPBRPipeline(Pipeline* p) { m_pbrPipeline = p; }
    // Set up GPU frustum-culling compute pass (call after loadPBRModel).
    void setupCullPipeline(const std::string& cullSpvPath);
    // Build BLAS/TLAS and enable ray-traced shadows (call after setupCullPipeline).
    void buildRTAccelStructures();
    // Set up mesh shader pipeline + meshlet descriptor sets (call after loadPBRModel).
    void setupMeshShaderPipeline(Pipeline* p);
    void setOrbitCamera(float azimuth, float elevation, float radius) {
        m_orbitAzimuth = azimuth; m_orbitElevation = elevation; m_orbitRadius = radius;
    }

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

    void createTexture();

    // Demo geometry (colored cube)
    std::unique_ptr<Buffer> m_vertexBuffer;
    std::unique_ptr<Buffer> m_indexBuffer;
    uint32_t                m_indexCount = 0;

    // Procedural texture (UV-gradient, uploaded once at init)
    VkImage        m_textureImage  = VK_NULL_HANDLE;
    VkDeviceMemory m_textureMemory = VK_NULL_HANDLE;
    VkImageView    m_textureView   = VK_NULL_HANDLE;
    VkSampler      m_sampler       = VK_NULL_HANDLE;

    CaptureCallback m_captureCallback;
    FrameCallback   m_frameCallback;
    GPUProfiler*    m_profiler       = nullptr;
    int             m_scene          = 0;
    const ReplayFrameData* m_replayData = nullptr;

    float    m_currentView[16]{};
    float    m_currentProj[16]{};

    std::vector<DrawCallRecord> m_lastDrawCalls;
    FrameDrawStats              m_lastFrameStats{};

    // Scene 2: external mesh geometry
    std::unique_ptr<Buffer> m_meshVertexBuffer;
    std::unique_ptr<Buffer> m_meshIndexBuffer;
    uint32_t                m_meshIndexCount = 0;
    Pipeline*               m_meshPipeline   = nullptr;

    // Scene 3: PBR multi-submesh model
    std::unique_ptr<PBRModel> m_pbrModel;
    Pipeline*                 m_pbrPipeline = nullptr;
    float                     m_pbrCameraPos[3] = { 0.0f, 1.5f, 3.0f };
    float m_orbitAzimuth   = 0.0f;
    float m_orbitElevation = 0.25f;
    float m_orbitRadius    = 2.5f;

    // GPU frustum-culling compute pipeline (set up after PBR model is loaded)
    void destroyCullPipeline();
    VkPipeline            m_cullPipeline  = VK_NULL_HANDLE;
    VkPipelineLayout      m_cullLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_cullDSLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      m_cullDescPool  = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet>         m_cullDescSets;       // one per frame-in-flight
    std::vector<std::unique_ptr<Buffer>> m_cullUBOBuffers;     // viewProj + drawCount per frame
    std::vector<std::unique_ptr<Buffer>> m_culledIndirectBufs; // output per frame

    // Async compute: dedicated compute queue dispatch with timeline semaphore
    VkCommandPool                m_computeCmdPool  = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_computeCmdBufs;
    VkSemaphore                  m_cullTimelineSem = VK_NULL_HANDLE;
    uint64_t                     m_cullTimelineVal = 0;

    // Mesh shader pipeline (task+mesh+frag replacing vertex+cull when supported)
    Pipeline* m_meshShaderPipeline = nullptr;

    bool     m_rtEnabled       = false;  // true once TLAS is built and descriptors updated
    uint32_t m_lastImageIndex  = 0;
    uint32_t m_frameCount      = 0;
};

} // namespace tgt
