#pragma once
#include <vulkan/vulkan.h>
#include <vector>

namespace tgt {

class VulkanContext;

class Buffer {
public:
    Buffer(VulkanContext& ctx, VkDeviceSize size,
           VkBufferUsageFlags usage, VkMemoryPropertyFlags props);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void upload(VkCommandPool pool, const void* data, VkDeviceSize size);
    void writeHostVisible(const void* data, VkDeviceSize size);

    VkBuffer       handle() const { return m_buffer; }
    VkDeviceSize   size()   const { return m_size; }

private:
    VulkanContext& m_ctx;
    VkBuffer       m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceSize   m_size;
};

struct Vertex {
    float pos[3];
    float color[3];
    float uv[2];

    static VkVertexInputBindingDescription   bindingDescription();
    static std::vector<VkVertexInputAttributeDescription> attributeDescriptions();
};

} // namespace tgt
