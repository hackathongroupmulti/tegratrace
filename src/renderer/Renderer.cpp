#include "Renderer.h"
#include "Pipeline.h"
#include "RenderPass.h"
#include "Buffer.h"
#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "profiling/GPUProfiler.h"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace tgt {

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

// Cube geometry: 8 vertices, 12 triangles
static const std::array<Vertex, 8> kCubeVerts = {{
    {{-0.5f,-0.5f,-0.5f}, {1,0,0}, {0,0}},
    {{ 0.5f,-0.5f,-0.5f}, {0,1,0}, {1,0}},
    {{ 0.5f, 0.5f,-0.5f}, {0,0,1}, {1,1}},
    {{-0.5f, 0.5f,-0.5f}, {1,1,0}, {0,1}},
    {{-0.5f,-0.5f, 0.5f}, {0,1,1}, {0,0}},
    {{ 0.5f,-0.5f, 0.5f}, {1,0,1}, {1,0}},
    {{ 0.5f, 0.5f, 0.5f}, {1,1,1}, {1,1}},
    {{-0.5f, 0.5f, 0.5f}, {0,0,0}, {0,1}},
}};

static const std::array<uint16_t, 36> kCubeIdx = {{
    0,1,2, 2,3,0,
    4,5,6, 6,7,4,
    0,4,7, 7,3,0,
    1,5,6, 6,2,1,
    3,2,6, 6,7,3,
    0,1,5, 5,4,0,
}};


Renderer::Renderer(VulkanContext& ctx, Swapchain& swapchain,
                   RenderPass& renderPass, Pipeline& pipeline)
    : m_ctx(ctx), m_swapchain(swapchain),
      m_renderPass(renderPass), m_pipeline(pipeline)
{
    createCommandPool();
    createCommandBuffers();

    m_vertexBuffer = std::make_unique<Buffer>(ctx,
        sizeof(kCubeVerts),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_vertexBuffer->upload(m_commandPool, kCubeVerts.data(), sizeof(kCubeVerts));

    m_indexBuffer = std::make_unique<Buffer>(ctx,
        sizeof(kCubeIdx),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_indexBuffer->upload(m_commandPool, kCubeIdx.data(), sizeof(kCubeIdx));
    m_indexCount = static_cast<uint32_t>(kCubeIdx.size());

    createDescriptorPool();
    createUniformBuffers();
    createDescriptorSets();
}

Renderer::~Renderer() {
    waitIdle();
    vkDestroyDescriptorPool(m_ctx.device(), m_descriptorPool, nullptr);
    m_uniformBuffers.clear();
    m_vertexBuffer.reset();
    m_indexBuffer.reset();
    vkDestroyCommandPool(m_ctx.device(), m_commandPool, nullptr);
}

void Renderer::createCommandPool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = m_ctx.queueFamilies().graphics.value();
    VK_CHECK(vkCreateCommandPool(m_ctx.device(), &ci, nullptr, &m_commandPool));
}

void Renderer::createCommandBuffers() {
    m_commandBuffers.resize(Swapchain::kMaxFramesInFlight);
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = m_commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());
    VK_CHECK(vkAllocateCommandBuffers(m_ctx.device(), &ai, m_commandBuffers.data()));
}

void Renderer::createDescriptorPool() {
    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps.descriptorCount = Swapchain::kMaxFramesInFlight;
    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.poolSizeCount = 1;
    ci.pPoolSizes    = &ps;
    ci.maxSets       = Swapchain::kMaxFramesInFlight;
    VK_CHECK(vkCreateDescriptorPool(m_ctx.device(), &ci, nullptr, &m_descriptorPool));
}

void Renderer::createUniformBuffers() {
    VkDeviceSize size = sizeof(UniformBufferObject);
    for (int i = 0; i < Swapchain::kMaxFramesInFlight; ++i) {
        m_uniformBuffers.push_back(std::make_unique<Buffer>(
            m_ctx, size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    }
}

void Renderer::createDescriptorSets() {
    // Need to get the DSLayout from the pipeline - use pipeline layout query
    // We expose it via a VkDescriptorSetLayout stored in the pipeline
    // For now get it via pipeline's layout (we store it indirectly)
    // Workaround: allocate from pool using the layout baked into the pipeline

    // The descriptor set layout was created in Pipeline::Pipeline and bound to m_layout.
    // We need the raw VkDescriptorSetLayout. To avoid circular deps, we re-create a compatible layout here.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding        = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount= 1;
    binding.stageFlags     = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 1;
    dlci.pBindings    = &binding;
    VkDescriptorSetLayout dsLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx.device(), &dlci, nullptr, &dsLayout));

    std::vector<VkDescriptorSetLayout> layouts(Swapchain::kMaxFramesInFlight, dsLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = m_descriptorPool;
    ai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    ai.pSetLayouts        = layouts.data();
    m_descriptorSets.resize(Swapchain::kMaxFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.device(), &ai, m_descriptorSets.data()));

    vkDestroyDescriptorSetLayout(m_ctx.device(), dsLayout, nullptr);

    for (int i = 0; i < Swapchain::kMaxFramesInFlight; ++i) {
        VkDescriptorBufferInfo bi{};
        bi.buffer = m_uniformBuffers[i]->handle();
        bi.offset = 0;
        bi.range  = sizeof(UniformBufferObject);
        VkWriteDescriptorSet wr{};
        wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet          = m_descriptorSets[i];
        wr.dstBinding      = 0;
        wr.descriptorCount = 1;
        wr.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wr.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(m_ctx.device(), 1, &wr, 0, nullptr);
    }
}

void Renderer::updateUniformBuffer(uint32_t frame, uint32_t frameNumber) {
    if (m_hasUBOOverride) {
        m_lastUBO = m_uboOverride;
        m_uniformBuffers[frame]->writeHostVisible(&m_lastUBO, sizeof(m_lastUBO));
        return;
    }
    UniformBufferObject ubo{};
    float angle = frameNumber * 0.016f;  // ~1 full rotation per 400 frames

    auto model = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.3f, 1.0f, 0.2f));
    model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -2.5f)) * model;

    auto view = glm::mat4(1.0f);

    auto ext    = m_swapchain.extent();
    float aspect = static_cast<float>(ext.width) / static_cast<float>(ext.height);
    auto proj   = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    proj[1][1] *= -1.0f;  // Vulkan Y-flip

    memcpy(ubo.model, glm::value_ptr(model), sizeof(ubo.model));
    memcpy(ubo.view,  glm::value_ptr(view),  sizeof(ubo.view));
    memcpy(ubo.proj,  glm::value_ptr(proj),  sizeof(ubo.proj));

    m_lastUBO = ubo;
    m_uniformBuffers[frame]->writeHostVisible(&ubo, sizeof(ubo));
}

VkCommandBuffer Renderer::recordCommandBuffer(uint32_t imageIndex, uint32_t frameNumber,
                                               FrameDrawStats& stats) {
    uint32_t frameIdx = m_swapchain.currentFrame() == 0 ?
                        Swapchain::kMaxFramesInFlight - 1 : m_swapchain.currentFrame() - 1;

    auto cmd = m_commandBuffers[frameIdx % Swapchain::kMaxFramesInFlight];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    auto ext = m_swapchain.extent();
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{0.05f, 0.05f, 0.08f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    uint32_t profIdx = frameIdx % Swapchain::kMaxFramesInFlight;
    if (m_profiler) {
        m_profiler->beginPass(cmd, profIdx, "main");
        m_profiler->beginPipelineStats(cmd, profIdx);
    }

    VkRenderPassBeginInfo rbi{};
    rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass      = m_renderPass.handle();
    rbi.framebuffer     = m_renderPass.framebuffer(imageIndex);
    rbi.renderArea      = {{0, 0}, ext};
    rbi.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rbi.pClearValues    = clearValues.data();

    Pipeline& activePipeline = m_activePipeline ? *m_activePipeline : m_pipeline;

    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline.handle());

    VkViewport vp{ 0, 0, static_cast<float>(ext.width), static_cast<float>(ext.height), 0, 1 };
    VkRect2D   sc{ {0, 0}, ext };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    VkBuffer vbuf = m_vertexBuffer->handle();
    VkDeviceSize vbOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &vbOffset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer->handle(), 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             activePipeline.layout(), 0, 1,
                             &m_descriptorSets[frameIdx % Swapchain::kMaxFramesInFlight],
                             0, nullptr);

    vkCmdDrawIndexed(cmd, m_indexCount, m_instanceCount, 0, 0, 0);
    stats.drawCalls++;
    stats.indexCount  += m_indexCount * m_instanceCount;
    stats.vertexCount += static_cast<uint32_t>(kCubeVerts.size()) * m_instanceCount;

    if (m_captureCallback) {
        DrawCallRecord rec{};
        rec.vertexCount    = static_cast<uint32_t>(kCubeVerts.size());
        rec.instanceCount  = m_instanceCount;
        rec.indexCount     = m_indexCount * m_instanceCount;
        rec.firstVertex    = 0;
        rec.pipeline       = activePipeline.name();
        rec.vertShader     = activePipeline.vertSpvPath();
        rec.fragShader     = activePipeline.fragSpvPath();
        rec.viewportW      = static_cast<float>(ext.width);
        rec.viewportH      = static_cast<float>(ext.height);
        memcpy(rec.model, m_lastUBO.model, sizeof(rec.model));
        memcpy(rec.view,  m_lastUBO.view,  sizeof(rec.view));
        memcpy(rec.proj,  m_lastUBO.proj,  sizeof(rec.proj));
        m_captureCallback(frameNumber, {rec});
    }

    if (m_frameCallback)
        m_frameCallback(frameNumber, cmd, stats);

    vkCmdEndRenderPass(cmd);

    if (m_profiler) {
        m_profiler->endPass(cmd, profIdx);
        m_profiler->endPipelineStats(cmd, profIdx);
    }

    VK_CHECK(vkEndCommandBuffer(cmd));
    return cmd;
}

bool Renderer::drawFrame(uint32_t frameNumber) {
    uint32_t imageIndex;
    auto result = m_swapchain.acquireNextImage(&imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { handleResize(); return false; }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("acquireNextImage failed");

    m_lastImageIndex = imageIndex;

    uint32_t frameIdx = m_swapchain.currentFrame() == 0 ?
                        Swapchain::kMaxFramesInFlight - 1 : m_swapchain.currentFrame() - 1;

    // Fence for this slot was just waited — GPU is done with previous work in this slot.
    // Safe to read profiler results from that completed submission.
    if (m_profiler && m_frameCount >= static_cast<uint32_t>(Swapchain::kMaxFramesInFlight))
        m_profiler->readResults(frameIdx % Swapchain::kMaxFramesInFlight);

    updateUniformBuffer(frameIdx % Swapchain::kMaxFramesInFlight, frameNumber);

    FrameDrawStats stats{};
    auto cmd = recordCommandBuffer(imageIndex, frameNumber, stats);

    result = m_swapchain.submitAndPresent(imageIndex, cmd);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { handleResize(); return false; }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("submitAndPresent failed");

    m_frameCount++;
    return true;
}

void Renderer::handleResize() {
    waitIdle();
    m_swapchain.recreate();
    m_renderPass.recreateFramebuffers(); // must follow swapchain recreate
}

void Renderer::waitIdle() {
    vkDeviceWaitIdle(m_ctx.device());
}

} // namespace tgt
