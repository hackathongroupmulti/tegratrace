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
#include <vector>
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
    createTexture();          // must precede createDescriptorSets — writes use textureView/sampler
    createDescriptorSets();
}

Renderer::~Renderer() {
    waitIdle();
    vkDestroySampler(m_ctx.device(), m_sampler, nullptr);
    vkDestroyImageView(m_ctx.device(), m_textureView, nullptr);
    vkDestroyImage(m_ctx.device(), m_textureImage, nullptr);
    vkFreeMemory(m_ctx.device(), m_textureMemory, nullptr);
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
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = Swapchain::kMaxFramesInFlight;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = Swapchain::kMaxFramesInFlight;

    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    ci.pPoolSizes    = poolSizes.data();
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
    // Use the layout from the pipeline directly — avoids duplication and ensures consistency
    std::vector<VkDescriptorSetLayout> layouts(Swapchain::kMaxFramesInFlight, m_pipeline.dsLayout());
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = m_descriptorPool;
    ai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    ai.pSetLayouts        = layouts.data();
    m_descriptorSets.resize(Swapchain::kMaxFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.device(), &ai, m_descriptorSets.data()));

    for (int i = 0; i < Swapchain::kMaxFramesInFlight; ++i) {
        VkDescriptorBufferInfo bi{};
        bi.buffer = m_uniformBuffers[i]->handle();
        bi.offset = 0;
        bi.range  = sizeof(UniformBufferObject);

        VkDescriptorImageInfo ii{};
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii.imageView   = m_textureView;
        ii.sampler     = m_sampler;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = m_descriptorSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo     = &bi;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = m_descriptorSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo      = &ii;

        vkUpdateDescriptorSets(m_ctx.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Renderer::createTexture() {
    // Generate a 64×64 UV-gradient texture: R=u, G=v, B=0.5
    constexpr uint32_t W = 64, H = 64;
    std::vector<uint8_t> pixels(W * H * 4);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t i = (y * W + x) * 4;
            pixels[i+0] = static_cast<uint8_t>((x * 255) / (W - 1));
            pixels[i+1] = static_cast<uint8_t>((y * 255) / (H - 1));
            pixels[i+2] = 128;
            pixels[i+3] = 255;
        }
    }
    VkDeviceSize imageSize = W * H * 4;

    // Upload via staging buffer
    Buffer staging(m_ctx, imageSize,
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.writeHostVisible(pixels.data(), imageSize);

    // Create device-local image
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent        = { W, H, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.format        = VK_FORMAT_R8G8B8A8_SRGB;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(m_ctx.device(), &ici, nullptr, &m_textureImage));

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(m_ctx.device(), m_textureImage, &memReq);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = m_ctx.findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_ctx.device(), &mai, nullptr, &m_textureMemory));
    vkBindImageMemory(m_ctx.device(), m_textureImage, m_textureMemory, 0);

    // Transition UNDEFINED → TRANSFER_DST, copy, transition → SHADER_READ_ONLY
    auto cmd = m_ctx.beginSingleTimeCommands(m_commandPool);
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = m_textureImage;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { W, H, 1 };
    vkCmdCopyBufferToImage(cmd, staging.handle(), m_textureImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = m_textureImage;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    m_ctx.endSingleTimeCommands(m_commandPool, cmd);

    // Image view
    VkImageViewCreateInfo ivci{};
    ivci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image                           = m_textureImage;
    ivci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format                          = VK_FORMAT_R8G8B8A8_SRGB;
    ivci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.baseMipLevel   = 0;
    ivci.subresourceRange.levelCount     = 1;
    ivci.subresourceRange.baseArrayLayer = 0;
    ivci.subresourceRange.layerCount     = 1;
    VK_CHECK(vkCreateImageView(m_ctx.device(), &ivci, nullptr, &m_textureView));

    // Sampler
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable        = VK_FALSE;
    sci.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sci.unnormalizedCoordinates = VK_FALSE;
    sci.compareEnable           = VK_FALSE;
    sci.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VK_CHECK(vkCreateSampler(m_ctx.device(), &sci, nullptr, &m_sampler));
}

void Renderer::updateUniformBuffer(uint32_t frame, uint32_t frameNumber) {
    UniformBufferObject ubo{};

    if (m_replayData) {
        // Use view/proj from replay data
        memcpy(ubo.view, m_replayData->view, sizeof(ubo.view));
        memcpy(ubo.proj, m_replayData->proj, sizeof(ubo.proj));
    } else {
        auto view = glm::mat4(1.0f);

        auto ext    = m_swapchain.extent();
        float aspect = static_cast<float>(ext.width) / static_cast<float>(ext.height);
        float fov = (m_scene == 1) ? 60.0f : 45.0f;
        auto proj   = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.0f);
        proj[1][1] *= -1.0f;  // Vulkan Y-flip

        memcpy(ubo.view, glm::value_ptr(view), sizeof(ubo.view));
        memcpy(ubo.proj, glm::value_ptr(proj), sizeof(ubo.proj));
    }

    memcpy(m_currentView, ubo.view, sizeof(m_currentView));
    memcpy(m_currentProj, ubo.proj, sizeof(m_currentProj));

    m_uniformBuffers[frame]->writeHostVisible(&ubo, sizeof(ubo));
    (void)frameNumber;
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

    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline.handle());

    VkViewport vp{ 0, 0, static_cast<float>(ext.width), static_cast<float>(ext.height), 0, 1 };
    VkRect2D   sc{ {0, 0}, ext };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    VkBuffer vbuf = m_vertexBuffer->handle();
    VkDeviceSize vbOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &vbOffset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer->handle(), 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             m_pipeline.layout(), 0, 1,
                             &m_descriptorSets[frameIdx % Swapchain::kMaxFramesInFlight],
                             0, nullptr);

    // Build list of per-object model matrices
    std::vector<glm::mat4> models;
    float angle = frameNumber * 0.016f;  // ~1 full rotation per 400 frames

    if (m_replayData) {
        for (auto& rd : m_replayData->draws) {
            glm::mat4 m;
            memcpy(glm::value_ptr(m), rd.model, 64);
            models.push_back(m);
        }
    } else if (m_scene == 0) {
        // Single cube
        auto model = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.3f, 1.0f, 0.2f));
        model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -2.5f)) * model;
        models.push_back(model);
    } else {
        // Scene 1: 5x5 grid of 25 cubes
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 5; ++col) {
                float x = (col - 2) * 1.5f;
                float y = (row - 2) * 1.5f;
                float z = -8.0f;
                auto model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
                model = model * glm::rotate(glm::mat4(1.0f),
                    angle + row * 0.3f + col * 0.2f, glm::vec3(0.3f, 1.0f, 0.2f));
                models.push_back(model);
            }
        }
    }

    std::vector<DrawCallRecord> records;

    for (auto& model : models) {
        // Push model matrix via push constants
        vkCmdPushConstants(cmd, m_pipeline.layout(), VK_SHADER_STAGE_VERTEX_BIT,
                           0, 64, glm::value_ptr(model));
        vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);

        stats.drawCalls++;
        stats.indexCount  += m_indexCount;
        stats.vertexCount += static_cast<uint32_t>(kCubeVerts.size());

        if (m_captureCallback) {
            DrawCallRecord rec{};
            rec.vertexCount    = static_cast<uint32_t>(kCubeVerts.size());
            rec.instanceCount  = 1;
            rec.indexCount     = m_indexCount;
            rec.firstVertex    = 0;
            rec.pipeline       = m_pipeline.name();
            rec.vertShader     = m_pipeline.vertSpvPath();
            rec.fragShader     = m_pipeline.fragSpvPath();
            rec.viewportW      = static_cast<float>(ext.width);
            rec.viewportH      = static_cast<float>(ext.height);
            memcpy(rec.model, glm::value_ptr(model), sizeof(rec.model));
            memcpy(rec.view,  m_currentView, sizeof(rec.view));
            memcpy(rec.proj,  m_currentProj, sizeof(rec.proj));
            records.push_back(rec);
        }
    }

    if (m_captureCallback && !records.empty()) {
        m_captureCallback(frameNumber, records);
    }

    // Cache draw list before frame callback so UI can read it via lastDrawCalls()
    m_lastDrawCalls = records;

    if (m_frameCallback)
        m_frameCallback(frameNumber, cmd, stats);

    vkCmdEndRenderPass(cmd);

    // End main pass timing
    if (m_profiler) m_profiler->endPass(cmd, profIdx);

    // Barrier probe: time a post-render pipeline sync point to measure synchronization overhead
    if (m_profiler) m_profiler->beginPass(cmd, profIdx, "barrier");
    {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    if (m_profiler) m_profiler->endPass(cmd, profIdx);

    if (m_profiler) m_profiler->endPipelineStats(cmd, profIdx);

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
