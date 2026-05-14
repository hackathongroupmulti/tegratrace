#include "Renderer.h"
#include "Pipeline.h"
#include "RenderPass.h"
#include "Buffer.h"
#include "PBRModel.h"
#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "profiling/GPUProfiler.h"
#include <tiny_obj_loader.h>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <fstream>
#include <array>
#include <vector>
#include <iostream>
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
    destroyCullPipeline();
    vkDestroySampler(m_ctx.device(), m_sampler, nullptr);
    vkDestroyImageView(m_ctx.device(), m_textureView, nullptr);
    vkDestroyImage(m_ctx.device(), m_textureImage, nullptr);
    vkFreeMemory(m_ctx.device(), m_textureMemory, nullptr);
    vkDestroyDescriptorPool(m_ctx.device(), m_descriptorPool, nullptr);
    m_uniformBuffers.clear();
    m_vertexBuffer.reset();
    m_indexBuffer.reset();
    m_meshVertexBuffer.reset();
    m_meshIndexBuffer.reset();
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

void Renderer::loadMesh(const std::string& objPath) {
    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;

    if (!objPath.empty()) {
        tinyobj::attrib_t                attrib;
        std::vector<tinyobj::shape_t>    shapes;
        std::vector<tinyobj::material_t> materials;
        std::string errMsg;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &errMsg, objPath.c_str())) {
            std::cerr << "[Mesh] Failed to load OBJ '" << objPath << "': " << errMsg << "\n";
        } else {
            if (!errMsg.empty()) std::cerr << "[Mesh] OBJ: " << errMsg << "\n";

            // Expand to unindexed flat triangles then re-index naively
            for (auto& shape : shapes) {
                for (auto& idx : shape.mesh.indices) {
                    Vertex v{};
                    int pi = idx.vertex_index * 3;
                    v.pos[0] = attrib.vertices[pi+0];
                    v.pos[1] = attrib.vertices[pi+1];
                    v.pos[2] = attrib.vertices[pi+2];

                    if (idx.normal_index >= 0) {
                        // Map normal to color for visual interest
                        int ni = idx.normal_index * 3;
                        v.color[0] = std::abs(attrib.normals[ni+0]);
                        v.color[1] = std::abs(attrib.normals[ni+1]);
                        v.color[2] = std::abs(attrib.normals[ni+2]);
                    } else {
                        v.color[0] = v.color[1] = v.color[2] = 0.8f;
                    }

                    if (idx.texcoord_index >= 0) {
                        int ti = idx.texcoord_index * 2;
                        v.uv[0] = attrib.texcoords[ti+0];
                        v.uv[1] = 1.0f - attrib.texcoords[ti+1]; // flip V for Vulkan
                    }

                    indices.push_back(static_cast<uint32_t>(verts.size()));
                    verts.push_back(v);
                }
            }
            std::cout << "[Mesh] Loaded OBJ '" << objPath << "': "
                      << verts.size()/3 << " triangles\n";
        }
    }

    // Fall back to procedural UV sphere if OBJ failed or none given
    if (verts.empty()) {
        constexpr uint32_t SLICES = 120, STACKS = 120;
        constexpr float    R      = 1.0f;
        const float PI = 3.14159265358979323846f;

        for (uint32_t i = 0; i <= STACKS; ++i) {
            float phi = PI * i / STACKS;
            for (uint32_t j = 0; j <= SLICES; ++j) {
                float theta = 2.0f * PI * j / SLICES;
                float x = R * std::sin(phi) * std::cos(theta);
                float y = R * std::cos(phi);
                float z = R * std::sin(phi) * std::sin(theta);
                Vertex v{};
                v.pos[0] = x; v.pos[1] = y; v.pos[2] = z;
                v.color[0] = std::abs(x); v.color[1] = std::abs(y); v.color[2] = std::abs(z);
                v.uv[0] = static_cast<float>(j) / SLICES;
                v.uv[1] = static_cast<float>(i) / STACKS;
                verts.push_back(v);
            }
        }
        for (uint32_t i = 0; i < STACKS; ++i) {
            for (uint32_t j = 0; j < SLICES; ++j) {
                uint32_t a = i * (SLICES+1) + j;
                uint32_t b = a + 1;
                uint32_t c = a + (SLICES+1);
                uint32_t d = c + 1;
                indices.push_back(a); indices.push_back(c); indices.push_back(b);
                indices.push_back(b); indices.push_back(c); indices.push_back(d);
            }
        }
        std::cout << "[Mesh] Generated UV sphere: " << indices.size()/3 << " triangles\n";
    }

    waitIdle();
    m_meshIndexCount = static_cast<uint32_t>(indices.size());

    VkDeviceSize vsize = verts.size()   * sizeof(Vertex);
    VkDeviceSize isize = indices.size() * sizeof(uint32_t);

    m_meshVertexBuffer = std::make_unique<Buffer>(m_ctx, vsize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_meshVertexBuffer->upload(m_commandPool, verts.data(), vsize);

    m_meshIndexBuffer = std::make_unique<Buffer>(m_ctx, isize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_meshIndexBuffer->upload(m_commandPool, indices.data(), isize);
}

void Renderer::loadPBRModel(const std::string& fbxPath) {
    waitIdle();
    m_pbrModel = std::make_unique<PBRModel>(m_ctx);
    if (!m_pbrModel->load(fbxPath, m_commandPool)) {
        std::cerr << "[Renderer] PBR model load failed: " << fbxPath << "\n";
        m_pbrModel.reset();
        return;
    }
    if (m_pbrPipeline) {
        m_pbrModel->createDescriptorResources(
            m_pbrPipeline->dsLayout(), Swapchain::kMaxFramesInFlight,
            m_pbrPipeline->bindless());
    }
}

// ---------------------------------------------------------------------------
// GPU frustum culling compute pipeline
// ---------------------------------------------------------------------------
static VkShaderModule loadSpvModule(VkDevice device, const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open shader: " + path);
    size_t sz = static_cast<size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = sz;
    ci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
    return mod;
}

void Renderer::destroyCullPipeline() {
    if (m_cullTimelineSem) { vkDestroySemaphore(m_ctx.device(), m_cullTimelineSem, nullptr); m_cullTimelineSem = VK_NULL_HANDLE; }
    if (m_computeCmdPool) { vkDestroyCommandPool(m_ctx.device(), m_computeCmdPool, nullptr); m_computeCmdPool = VK_NULL_HANDLE; m_computeCmdBufs.clear(); }
    m_cullDescSets.clear();
    m_cullUBOBuffers.clear();
    m_culledIndirectBufs.clear();
    if (m_cullDescPool) { vkDestroyDescriptorPool(m_ctx.device(), m_cullDescPool, nullptr); m_cullDescPool = VK_NULL_HANDLE; }
    if (m_cullPipeline) { vkDestroyPipeline(m_ctx.device(), m_cullPipeline, nullptr); m_cullPipeline = VK_NULL_HANDLE; }
    if (m_cullLayout)   { vkDestroyPipelineLayout(m_ctx.device(), m_cullLayout, nullptr); m_cullLayout = VK_NULL_HANDLE; }
    if (m_cullDSLayout) { vkDestroyDescriptorSetLayout(m_ctx.device(), m_cullDSLayout, nullptr); m_cullDSLayout = VK_NULL_HANDLE; }
}

void Renderer::setupCullPipeline(const std::string& cullSpvPath) {
    if (!m_pbrModel || !m_pbrModel->isLoaded()) return;
    destroyCullPipeline();

    // Descriptor set layout: 3 SSBOs + 1 UBO
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding         = static_cast<uint32_t>(i);
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[3].binding        = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].descriptorCount= 1;
    bindings[3].stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 4;
    dlci.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx.device(), &dlci, nullptr, &m_cullDSLayout));

    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &m_cullDSLayout;
    VK_CHECK(vkCreatePipelineLayout(m_ctx.device(), &plci, nullptr, &m_cullLayout));

    VkShaderModule compMod = loadSpvModule(m_ctx.device(), cullSpvPath);
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = compMod;
    cpci.stage.pName  = "main";
    cpci.layout       = m_cullLayout;
    VK_CHECK(vkCreateComputePipelines(m_ctx.device(), m_ctx.pipelineCache(), 1, &cpci, nullptr, &m_cullPipeline));
    vkDestroyShaderModule(m_ctx.device(), compMod, nullptr);

    constexpr int FIF = Swapchain::kMaxFramesInFlight;
    uint32_t drawCount = m_pbrModel->drawCount();
    VkDeviceSize indSize = drawCount * sizeof(VkDrawIndexedIndirectCommand);

    // Descriptor pool (3 SSBO + 1 UBO) x FIF
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3u * FIF };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u * FIF };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.poolSizeCount = 2;
    pci.pPoolSizes    = poolSizes.data();
    pci.maxSets       = static_cast<uint32_t>(FIF);
    VK_CHECK(vkCreateDescriptorPool(m_ctx.device(), &pci, nullptr, &m_cullDescPool));

    std::vector<VkDescriptorSetLayout> layouts(FIF, m_cullDSLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = m_cullDescPool;
    ai.descriptorSetCount = static_cast<uint32_t>(FIF);
    ai.pSetLayouts        = layouts.data();
    m_cullDescSets.resize(FIF);
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.device(), &ai, m_cullDescSets.data()));

    // For async compute, culled indirect buffers need concurrent sharing between compute and graphics
    std::vector<uint32_t> indirectSharing;
    if (m_ctx.hasAsyncCompute()) {
        indirectSharing = { m_ctx.queueFamilies().graphics.value(),
                            m_ctx.queueFamilies().compute.value() };
    }

    // Per-frame UBO and culled indirect buffers; write descriptor sets
    struct CullParams { float viewProj[16]; uint32_t drawCount; uint32_t pad[3]; };
    for (int f = 0; f < FIF; ++f) {
        m_cullUBOBuffers.push_back(std::make_unique<Buffer>(m_ctx, sizeof(CullParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
        m_culledIndirectBufs.push_back(std::make_unique<Buffer>(m_ctx, indSize,
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indirectSharing));

        VkDescriptorBufferInfo srcInfo{ m_pbrModel->indirectBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo dstInfo{ m_culledIndirectBufs[f]->handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo sphInfo{ m_pbrModel->sphereBuffer(),   0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo uboInfo{ m_cullUBOBuffers[f]->handle(), 0, sizeof(CullParams) };

        std::array<VkWriteDescriptorSet, 4> writes{};
        for (int i = 0; i < 4; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = m_cullDescSets[f];
            writes[i].dstBinding      = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[0].pBufferInfo = &srcInfo;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[1].pBufferInfo = &dstInfo;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[2].pBufferInfo = &sphInfo;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[3].pBufferInfo = &uboInfo;
        vkUpdateDescriptorSets(m_ctx.device(), 4, writes.data(), 0, nullptr);
    }
    // Async compute: create dedicated compute command pool, command buffers, and timeline semaphore
    if (m_ctx.hasAsyncCompute()) {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = m_ctx.queueFamilies().compute.value();
        cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(m_ctx.device(), &cpci, nullptr, &m_computeCmdPool));

        m_computeCmdBufs.resize(FIF);
        VkCommandBufferAllocateInfo cai{};
        cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool        = m_computeCmdPool;
        cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = FIF;
        VK_CHECK(vkAllocateCommandBuffers(m_ctx.device(), &cai, m_computeCmdBufs.data()));

        VkSemaphoreTypeCreateInfo tsci{};
        tsci.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        tsci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        tsci.initialValue  = 0;
        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sci.pNext = &tsci;
        VK_CHECK(vkCreateSemaphore(m_ctx.device(), &sci, nullptr, &m_cullTimelineSem));
        m_cullTimelineVal = 0;
        std::cout << "[Renderer] Async compute cull enabled (timeline semaphore, queue family "
                  << m_ctx.queueFamilies().compute.value() << ")\n";
    }

    std::cout << "[Renderer] GPU frustum-culling compute pipeline ready ("
              << drawCount << " submeshes)\n";
}

void Renderer::updateUniformBuffer(uint32_t frame, uint32_t frameNumber) {
    UniformBufferObject ubo{};

    if (m_replayData) {
        // Use view/proj from replay data
        memcpy(ubo.view, m_replayData->view, sizeof(ubo.view));
        memcpy(ubo.proj, m_replayData->proj, sizeof(ubo.proj));
    } else {
        auto ext    = m_swapchain.extent();
        float aspect = static_cast<float>(ext.width) / static_cast<float>(ext.height);

        glm::mat4 view;
        if (m_scene == 3) {
            float cosEl = std::cos(m_orbitElevation);
            glm::vec3 eye = {
                m_orbitRadius * cosEl * std::sin(m_orbitAzimuth),
                m_orbitRadius * std::sin(m_orbitElevation),
                m_orbitRadius * cosEl * std::cos(m_orbitAzimuth)
            };
            glm::vec3 at = { 0.0f, 0.9f, 0.0f };
            view = glm::lookAt(eye, at, { 0.0f, 1.0f, 0.0f });
            m_pbrCameraPos[0] = eye.x;
            m_pbrCameraPos[1] = eye.y;
            m_pbrCameraPos[2] = eye.z;
        } else {
            view = glm::mat4(1.0f);
        }

        float fov = (m_scene == 1) ? 60.0f : (m_scene == 2) ? 50.0f :
                    (m_scene == 3) ? 45.0f  : 45.0f;
        auto proj   = glm::perspective(glm::radians(fov), aspect, 0.01f, 200.0f);
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
    const bool isPBRScene = (m_scene == 3) && m_pbrModel && m_pbrModel->isLoaded() && m_pbrPipeline;
    if (m_profiler) {
        if (!isPBRScene) m_profiler->beginPass(cmd, profIdx, "main");
        m_profiler->beginPipelineStats(cmd, profIdx);
    }

    VkRenderPassBeginInfo rbi{};
    rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass      = m_renderPass.handle();
    rbi.framebuffer     = m_renderPass.framebuffer(imageIndex);
    rbi.renderArea      = {{0, 0}, ext};
    rbi.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rbi.pClearValues    = clearValues.data();

    // PRE-PASS: inline frustum culling — must be OUTSIDE the render pass.
    // Skipped when async compute is active (cull was dispatched on compute queue in drawFrame).
    if (isPBRScene && m_cullPipeline && !m_culledIndirectBufs.empty() && !m_cullTimelineSem) {
        uint32_t fi2 = profIdx;
        struct CullParams { float viewProj[16]; uint32_t drawCount; uint32_t pad[3]; };
        glm::mat4 vp2 = glm::make_mat4(m_currentProj) * glm::make_mat4(m_currentView);
        CullParams cp{};
        std::memcpy(cp.viewProj, glm::value_ptr(vp2), 64);
        cp.drawCount = m_pbrModel->drawCount();
        m_cullUBOBuffers[fi2]->writeHostVisible(&cp, sizeof(cp));

        uint32_t groups = (cp.drawCount + 63) / 64;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cullPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_cullLayout, 0, 1, &m_cullDescSets[fi2], 0, nullptr);
        vkCmdDispatch(cmd, groups, 1, 1);

        VkMemoryBarrier cullBarrier{};
        cullBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        cullBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        cullBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &cullBarrier, 0, nullptr, 0, nullptr);
    }

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
    } else if (m_scene == 1) {
        // 5x5 grid of 25 cubes
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
    } else {
        // Scene 2: single mesh (sphere or loaded OBJ)
        auto model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));
        model = model * glm::rotate(glm::mat4(1.0f), angle * 0.4f, glm::vec3(0.0f, 1.0f, 0.0f));
        models.push_back(model);
    }

    // Scene 2 uses its own vertex/index buffers and optionally a different pipeline
    const bool isMeshScene = (m_scene == 2) && m_meshVertexBuffer && m_meshIndexBuffer;
    if (isMeshScene) {
        if (m_meshPipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshPipeline->handle());
            // Rebind descriptor sets under the mesh pipeline layout
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_meshPipeline->layout(), 0, 1,
                &m_descriptorSets[frameIdx % Swapchain::kMaxFramesInFlight], 0, nullptr);
        }
        VkBuffer     mvbuf  = m_meshVertexBuffer->handle();
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mvbuf, &offset);
        vkCmdBindIndexBuffer(cmd, m_meshIndexBuffer->handle(), 0, VK_INDEX_TYPE_UINT32);
    }

    // ---- Scene 3: PBR multi-submesh model — mesh shader OR GPU-driven indirect ----
    if (isPBRScene) {
        m_lastDrawCalls.clear();
        const bool useMeshShader = m_meshShaderPipeline &&
                                   m_pbrModel->meshletCount() > 0 &&
                                   m_ctx.fnCmdDrawMeshTasks;

        if (useMeshShader) {
            // Task+mesh pipeline: one DrawMeshTasksEXT per submesh for per-submesh material indices
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshShaderPipeline->handle());

            glm::mat4 modelMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f));
            modelMat = glm::rotate(modelMat, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

            uint32_t fi = frameIdx % Swapchain::kMaxFramesInFlight;
            VkDescriptorSet meshSet = m_pbrModel->meshDescriptorSet(static_cast<int>(fi));
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_meshShaderPipeline->layout(), 0, 1, &meshSet, 0, nullptr);

            m_ctx.beginDebugLabel(cmd, "PBR_Mesh", 0.8f, 0.4f, 0.8f);

            struct MeshDrawPC {
                float    model[16];
                uint32_t albedoIdx, normalIdx, roughIdx, metallicIdx, aoIdx, brdfLutIdx;
                uint32_t rtEnabled;
                uint32_t meshletOffset;
                uint32_t meshletCount;
            };

            for (int s = 0; s < static_cast<int>(m_pbrModel->submeshes().size()); ++s) {
                const auto& sub = m_pbrModel->submeshes()[s];
                uint32_t mc = m_pbrModel->submeshMeshletCount(s);
                if (mc == 0) continue;

                MeshDrawPC dpc{};
                std::memcpy(dpc.model, glm::value_ptr(modelMat), 64);
                const auto& ti = m_pbrModel->texIndices(s);
                dpc.albedoIdx    = ti.albedoIdx;
                dpc.normalIdx    = ti.normalIdx;
                dpc.roughIdx     = ti.roughIdx;
                dpc.metallicIdx  = ti.metallicIdx;
                dpc.aoIdx        = ti.aoIdx;
                dpc.brdfLutIdx   = ti.brdfLutIdx;
                dpc.rtEnabled    = m_rtEnabled ? 1u : 0u;
                dpc.meshletOffset = m_pbrModel->submeshMeshletOffset(s);
                dpc.meshletCount  = mc;

                vkCmdPushConstants(cmd, m_meshShaderPipeline->layout(),
                    VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                    VK_SHADER_STAGE_FRAGMENT_BIT, 0, 100, &dpc);

                uint32_t numGroups = (mc + 31u) / 32u;
                m_ctx.fnCmdDrawMeshTasks(cmd, numGroups, 1, 1);

                stats.drawCalls++;
                stats.indexCount += sub.indexCount;
            }
            m_ctx.endDebugLabel(cmd);

            if (m_frameCallback) m_frameCallback(frameNumber, cmd, stats);
            vkCmdEndRenderPass(cmd);
            VK_CHECK(vkEndCommandBuffer(cmd));
            return cmd;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pbrPipeline->handle());

        // XPS FBX is Z-up, centimetre units → rotate -90° on X, scale to metres
        glm::mat4 modelMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f));
        modelMat = glm::rotate(modelMat, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        // Bind consolidated VBO/IBO once for all submeshes (GPU-driven geometry)
        VkBuffer     cvbo    = m_pbrModel->consolidatedVBO();
        VkDeviceSize vbOff   = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &cvbo, &vbOff);
        vkCmdBindIndexBuffer(cmd, m_pbrModel->consolidatedIBO(), 0, VK_INDEX_TYPE_UINT32);

        uint32_t fi = frameIdx % Swapchain::kMaxFramesInFlight;

        m_ctx.beginDebugLabel(cmd, "PBR_Model", 0.2f, 0.8f, 0.4f);

        for (int s = 0; s < static_cast<int>(m_pbrModel->submeshes().size()); ++s) {
            const auto& sub = m_pbrModel->submeshes()[s];

            std::string passName = "sub:" + (sub.name.empty() ? std::to_string(s) : sub.name);
            if (m_profiler) m_profiler->beginPass(cmd, profIdx, passName);

            // Nsight/RenderDoc: each submesh gets its own colour-coded label
            m_ctx.beginDebugLabel(cmd, passName.c_str(), 0.4f, 0.6f, 1.0f);

            // Bindless: bind once before the loop; classic: bind per-submesh
            if (!m_pbrPipeline->bindless() || s == 0) {
                VkDescriptorSet ds = m_pbrModel->descriptorSet(s, fi);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pbrPipeline->layout(), 0, 1, &ds, 0, nullptr);
            }

            if (m_pbrPipeline->bindless()) {
                // Push model matrix + 6 tex indices + rtEnabled (92 bytes, vert|frag stages)
                struct PBRDrawPC {
                    float    model[16];
                    uint32_t albedoIdx, normalIdx, roughIdx, metallicIdx, aoIdx, brdfLutIdx;
                    uint32_t rtEnabled;
                } dpc{};
                std::memcpy(dpc.model, glm::value_ptr(modelMat), 64);
                const auto& ti = m_pbrModel->texIndices(s);
                dpc.albedoIdx   = ti.albedoIdx;
                dpc.normalIdx   = ti.normalIdx;
                dpc.roughIdx    = ti.roughIdx;
                dpc.metallicIdx = ti.metallicIdx;
                dpc.aoIdx       = ti.aoIdx;
                dpc.brdfLutIdx  = ti.brdfLutIdx;
                dpc.rtEnabled   = m_rtEnabled ? 1u : 0u;
                vkCmdPushConstants(cmd, m_pbrPipeline->layout(),
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 92, &dpc);
            } else {
                vkCmdPushConstants(cmd, m_pbrPipeline->layout(),
                    VK_SHADER_STAGE_VERTEX_BIT, 0, 64, glm::value_ptr(modelMat));
            }

            // GPU-driven: draw from culled buffer if compute cull is active
            VkBuffer indBuf = (!m_culledIndirectBufs.empty())
                ? m_culledIndirectBufs[profIdx]->handle()
                : m_pbrModel->indirectBuffer();
            vkCmdDrawIndexedIndirect(cmd, indBuf,
                static_cast<VkDeviceSize>(s) * sizeof(VkDrawIndexedIndirectCommand),
                1, sizeof(VkDrawIndexedIndirectCommand));

            m_ctx.endDebugLabel(cmd);

            if (m_profiler) m_profiler->endPass(cmd, profIdx);

            stats.drawCalls++;
            stats.indexCount += sub.indexCount;

            if (m_captureCallback) {
                DrawCallRecord rec{};
                rec.vertexCount   = sub.indexCount;
                rec.instanceCount = 1;
                rec.indexCount    = sub.indexCount;
                rec.pipeline      = m_pbrPipeline->name();
                rec.vertShader    = m_pbrPipeline->vertSpvPath();
                rec.fragShader    = m_pbrPipeline->fragSpvPath();
                rec.viewportW     = static_cast<float>(ext.width);
                rec.viewportH     = static_cast<float>(ext.height);
                memcpy(rec.model, glm::value_ptr(modelMat), 64);
                memcpy(rec.view,  m_currentView, 64);
                memcpy(rec.proj,  m_currentProj, 64);
                m_lastDrawCalls.push_back(rec);
            }
        }

        m_ctx.endDebugLabel(cmd);  // end PBR_Model

        if (m_captureCallback && !m_lastDrawCalls.empty())
            m_captureCallback(frameNumber, m_lastDrawCalls);

        if (m_frameCallback) m_frameCallback(frameNumber, cmd, stats);
        vkCmdEndRenderPass(cmd);

        if (m_profiler) m_profiler->beginPass(cmd, profIdx, "barrier");
        {
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
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

    std::vector<DrawCallRecord> records;
    Pipeline& activePipelineRef = (isMeshScene && m_meshPipeline) ? *m_meshPipeline : m_pipeline;

    for (auto& model : models) {
        vkCmdPushConstants(cmd, activePipelineRef.layout(), VK_SHADER_STAGE_VERTEX_BIT,
                           0, 64, glm::value_ptr(model));

        if (isMeshScene) {
            vkCmdDrawIndexed(cmd, m_meshIndexCount, 1, 0, 0, 0);
            stats.drawCalls++;
            stats.indexCount  += m_meshIndexCount;
        } else {
            vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
            stats.drawCalls++;
            stats.indexCount  += m_indexCount;
            stats.vertexCount += static_cast<uint32_t>(kCubeVerts.size());
        }

        if (m_captureCallback) {
            DrawCallRecord rec{};
            rec.vertexCount    = isMeshScene ? m_meshIndexCount : static_cast<uint32_t>(kCubeVerts.size());
            rec.instanceCount  = 1;
            rec.indexCount     = isMeshScene ? m_meshIndexCount : m_indexCount;
            rec.firstVertex    = 0;
            rec.pipeline       = activePipelineRef.name();
            rec.vertShader     = activePipelineRef.vertSpvPath();
            rec.fragShader     = activePipelineRef.fragSpvPath();
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

    // Update PBR model UBO when in scene 3
    if (m_scene == 3 && m_pbrModel && m_pbrModel->isLoaded()) {
        static const float kLightDir[4]   = { 0.577f, 0.577f, 0.577f, 0.0f };
        static const float kLightColor[4] = { 1.0f, 0.98f, 0.95f, 3.5f };
        float camPos4[4] = { m_pbrCameraPos[0], m_pbrCameraPos[1], m_pbrCameraPos[2], 1.0f };
        m_pbrModel->updateUBO(frameIdx % Swapchain::kMaxFramesInFlight,
                              m_currentView, m_currentProj,
                              camPos4, kLightDir, kLightColor);
    }

    // Async compute: dispatch frustum cull on dedicated compute queue before graphics work.
    // The timeline semaphore ensures graphics DRAW_INDIRECT stage waits for cull completion.
    if (m_ctx.hasAsyncCompute() && m_cullTimelineSem && m_scene == 3 &&
        m_pbrModel && m_pbrModel->isLoaded() && m_cullPipeline && !m_culledIndirectBufs.empty()) {
        uint32_t fi2 = frameIdx % Swapchain::kMaxFramesInFlight;
        struct CullParams { float viewProj[16]; uint32_t drawCount; uint32_t pad[3]; };
        glm::mat4 mvp = glm::make_mat4(m_currentProj) * glm::make_mat4(m_currentView);
        CullParams cp{};
        std::memcpy(cp.viewProj, glm::value_ptr(mvp), 64);
        cp.drawCount = m_pbrModel->drawCount();
        m_cullUBOBuffers[fi2]->writeHostVisible(&cp, sizeof(cp));

        VkCommandBuffer ccmd = m_computeCmdBufs[fi2];
        vkResetCommandBuffer(ccmd, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ccmd, &bi);

        uint32_t groups = (cp.drawCount + 63) / 64;
        vkCmdBindPipeline(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cullPipeline);
        vkCmdBindDescriptorSets(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_cullLayout, 0, 1, &m_cullDescSets[fi2], 0, nullptr);
        vkCmdDispatch(ccmd, groups, 1, 1);
        vkEndCommandBuffer(ccmd);

        uint64_t signalVal = ++m_cullTimelineVal;
        VkTimelineSemaphoreSubmitInfo tssi{};
        tssi.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tssi.signalSemaphoreValueCount = 1;
        tssi.pSignalSemaphoreValues    = &signalVal;
        VkSubmitInfo csi{};
        csi.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        csi.pNext                = &tssi;
        csi.commandBufferCount   = 1;
        csi.pCommandBuffers      = &ccmd;
        csi.signalSemaphoreCount = 1;
        csi.pSignalSemaphores    = &m_cullTimelineSem;
        VK_CHECK(vkQueueSubmit(m_ctx.computeQueue(), 1, &csi, VK_NULL_HANDLE));
    }

    FrameDrawStats stats{};
    auto cmd = recordCommandBuffer(imageIndex, frameNumber, stats);
    m_lastFrameStats = stats;

    VkSemaphore passSem = (m_cullTimelineSem && m_scene == 3) ? m_cullTimelineSem : VK_NULL_HANDLE;
    result = m_swapchain.submitAndPresent(imageIndex, cmd, passSem, m_cullTimelineVal);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { handleResize(); return false; }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("submitAndPresent failed");

    m_frameCount++;
    return true;
}

void Renderer::setupMeshShaderPipeline(Pipeline* p) {
    if (!m_pbrModel || !m_pbrModel->meshletCount() || !p) return;
    m_meshShaderPipeline = p;
    m_pbrModel->createMeshDescriptorResources(p->dsLayout(), Swapchain::kMaxFramesInFlight);
    std::cout << "[Renderer] Mesh shader pipeline ready ("
              << m_pbrModel->meshletCount() << " meshlets across "
              << m_pbrModel->submeshes().size() << " submeshes)\n";
}

void Renderer::buildRTAccelStructures() {
    if (!m_pbrModel || !m_pbrModel->isLoaded() || !m_ctx.rayQuerySupported()) return;

    // Use the same model transform applied in recordCommandBuffer for scene 3
    glm::mat4 modelMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f));
    modelMat = glm::rotate(modelMat, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    m_pbrModel->buildAccelerationStructures(m_commandPool, glm::value_ptr(modelMat));
    if (m_pbrModel->rtBuilt()) {
        m_pbrModel->writeTLASDescriptors();
        m_rtEnabled = true;
        std::cout << "[Renderer] Ray-traced shadows enabled (VK_KHR_ray_query)\n";
    }
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
