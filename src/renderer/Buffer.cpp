#include "Buffer.h"
#include "core/VulkanContext.h"
#include <cstring>
#include <stdexcept>

namespace tgt {

Buffer::Buffer(VulkanContext& ctx, VkDeviceSize size,
               VkBufferUsageFlags usage, VkMemoryPropertyFlags props)
    : m_ctx(ctx), m_size(size)
{
    VkBufferCreateInfo bi{};
    bi.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size        = size;
    bi.usage       = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device(), &bi, nullptr, &m_buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device(), m_buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx.device(), &ai, nullptr, &m_memory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate buffer memory");

    vkBindBufferMemory(ctx.device(), m_buffer, m_memory, 0);
}

Buffer::~Buffer() {
    if (m_buffer) vkDestroyBuffer(m_ctx.device(), m_buffer, nullptr);
    if (m_memory) vkFreeMemory(m_ctx.device(), m_memory, nullptr);
}

void Buffer::upload(VkCommandPool pool, const void* data, VkDeviceSize size) {
    // Stage through a host-visible staging buffer
    Buffer staging(m_ctx, size,
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.writeHostVisible(data, size);

    auto cmd = m_ctx.beginSingleTimeCommands(pool);
    VkBufferCopy region{ 0, 0, size };
    vkCmdCopyBuffer(cmd, staging.handle(), m_buffer, 1, &region);
    m_ctx.endSingleTimeCommands(pool, cmd);
}

void Buffer::writeHostVisible(const void* data, VkDeviceSize size) {
    void* mapped;
    vkMapMemory(m_ctx.device(), m_memory, 0, size, 0, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(m_ctx.device(), m_memory);
}

VkVertexInputBindingDescription Vertex::bindingDescription() {
    return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
}

std::vector<VkVertexInputAttributeDescription> Vertex::attributeDescriptions() {
    return {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)   },
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
        { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)    },
    };
}

VkVertexInputBindingDescription PBRVertex::bindingDescription() {
    return { 0, sizeof(PBRVertex), VK_VERTEX_INPUT_RATE_VERTEX };
}

std::vector<VkVertexInputAttributeDescription> PBRVertex::attributeDescriptions() {
    return {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PBRVertex, pos)       },
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PBRVertex, normal)    },
        { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(PBRVertex, uv)        },
        { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PBRVertex, tangent)   },
        { 4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PBRVertex, bitangent) },
    };
}

} // namespace tgt
