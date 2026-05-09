#include "PBRModel.h"
#include "Buffer.h"
#include "core/VulkanContext.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <stb_image.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cmath>

namespace fs = std::filesystem;

namespace tgt {

#define VK_CHECK(x) \
    do { VkResult _r = (x); \
         if (_r != VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

PBRModel::PBRModel(VulkanContext& ctx) : m_ctx(ctx) {}

PBRModel::~PBRModel() {
    vkDeviceWaitIdle(m_ctx.device());

    if (m_descPool) vkDestroyDescriptorPool(m_ctx.device(), m_descPool, nullptr);
    m_uboBuffers.clear();

    destroyMaterials();
    destroyTextureCache();
    m_submeshes.clear();

    m_consolidatedVBO.reset();
    m_consolidatedIBO.reset();
    m_indirectBuffer.reset();
    m_sphereBuffer.reset();

    // Fallback textures
    if (m_fbWhiteView) vkDestroyImageView(m_ctx.device(), m_fbWhiteView, nullptr);
    if (m_fbWhiteImg)  vkDestroyImage(m_ctx.device(), m_fbWhiteImg, nullptr);
    if (m_fbWhiteMem)  vkFreeMemory(m_ctx.device(), m_fbWhiteMem, nullptr);

    if (m_fbNormView)  vkDestroyImageView(m_ctx.device(), m_fbNormView, nullptr);
    if (m_fbNormImg)   vkDestroyImage(m_ctx.device(), m_fbNormImg, nullptr);
    if (m_fbNormMem)   vkFreeMemory(m_ctx.device(), m_fbNormMem, nullptr);

    // IBL textures
    if (m_envEquirectView) vkDestroyImageView(m_ctx.device(), m_envEquirectView, nullptr);
    if (m_envEquirectImg)  vkDestroyImage(m_ctx.device(), m_envEquirectImg, nullptr);
    if (m_envEquirectMem)  vkFreeMemory(m_ctx.device(), m_envEquirectMem, nullptr);

    if (m_envCubeView) vkDestroyImageView(m_ctx.device(), m_envCubeView, nullptr);
    if (m_envCubeImg)  vkDestroyImage(m_ctx.device(), m_envCubeImg, nullptr);
    if (m_envCubeMem)  vkFreeMemory(m_ctx.device(), m_envCubeMem, nullptr);

    for (auto v : m_prefilterMipViews) vkDestroyImageView(m_ctx.device(), v, nullptr);
    if (m_prefilterView) vkDestroyImageView(m_ctx.device(), m_prefilterView, nullptr);
    if (m_prefilterImg)  vkDestroyImage(m_ctx.device(), m_prefilterImg, nullptr);
    if (m_prefilterMem)  vkFreeMemory(m_ctx.device(), m_prefilterMem, nullptr);

    if (m_brdfLutView) vkDestroyImageView(m_ctx.device(), m_brdfLutView, nullptr);
    if (m_brdfLutImg)  vkDestroyImage(m_ctx.device(), m_brdfLutImg, nullptr);
    if (m_brdfLutMem)  vkFreeMemory(m_ctx.device(), m_brdfLutMem, nullptr);

    if (m_irradianceView) vkDestroyImageView(m_ctx.device(), m_irradianceView, nullptr);
    if (m_irradianceImg)  vkDestroyImage(m_ctx.device(), m_irradianceImg, nullptr);
    if (m_irradianceMem)  vkFreeMemory(m_ctx.device(), m_irradianceMem, nullptr);

    if (m_sampler) vkDestroySampler(m_ctx.device(), m_sampler, nullptr);
}

void PBRModel::destroyMaterials() {
    m_materials.clear();  // views borrowed from cache; cache owns the GPU objects
}

void PBRModel::destroyTextureCache() {
    for (auto& [path, tex] : m_texCache) {
        if (tex.view)   vkDestroyImageView(m_ctx.device(), tex.view, nullptr);
        if (tex.image)  vkDestroyImage(m_ctx.device(), tex.image, nullptr);
        if (tex.memory) vkFreeMemory(m_ctx.device(), tex.memory, nullptr);
    }
    m_texCache.clear();
}

// ---------------------------------------------------------------------------
// Upload a 1×1 RGBA8 pixel to a new device-local image (for fallback textures)
// ---------------------------------------------------------------------------
static VkImageView makeSmallTexture(VulkanContext& ctx, VkCommandPool pool,
                                    uint8_t pr, uint8_t pg, uint8_t pb, uint8_t pa,
                                    VkImage& outImg, VkDeviceMemory& outMem) {
    uint8_t pixels[4] = { pr, pg, pb, pa };

    Buffer staging(ctx, 4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.writeHostVisible(pixels, 4);

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent        = { 1, 1, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateImage(ctx.device(), &ici, nullptr, &outImg);
    if (r != VK_SUCCESS) throw std::runtime_error("vkCreateImage failed");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx.device(), outImg, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(ctx.device(), &mai, nullptr, &outMem);
    vkBindImageMemory(ctx.device(), outImg, outMem, 0);

    auto cmd = ctx.beginSingleTimeCommands(pool);
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = outImg;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,nullptr,0,nullptr,1,&barrier);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { 1, 1, 1 };
    vkCmdCopyBufferToImage(cmd, staging.handle(), outImg,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = outImg;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,nullptr,0,nullptr,1,&barrier);
    }
    ctx.endSingleTimeCommands(pool, cmd);

    VkImageViewCreateInfo ivci{};
    ivci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image                           = outImg;
    ivci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format                          = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount     = 1;
    ivci.subresourceRange.layerCount     = 1;
    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(ctx.device(), &ivci, nullptr, &view);
    return view;
}

void PBRModel::createFallbackTextures(VkCommandPool pool) {
    m_fbWhiteView = makeSmallTexture(m_ctx, pool,
        255, 255, 255, 255, m_fbWhiteImg, m_fbWhiteMem);
    m_fbNormView = makeSmallTexture(m_ctx, pool,
        128, 128, 255, 255, m_fbNormImg, m_fbNormMem);
}

// ---------------------------------------------------------------------------
// Texture loading from disk — generates full mip chain via vkCmdBlitImage
// ---------------------------------------------------------------------------
VkImageView PBRModel::loadTextureFile(const std::string& path, bool srgb,
                                      VkImage& outImg, VkDeviceMemory& outMem,
                                      VkCommandPool pool) {
    int w, h, ch;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "[PBR] stb_image failed: " << path << " — " << stbi_failure_reason() << "\n";
        return VK_NULL_HANDLE;
    }

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(w) * h * 4;
    VkFormat fmt = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1;

    Buffer staging(m_ctx, imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.writeHostVisible(pixels, imageSize);
    stbi_image_free(pixels);

    // TRANSFER_SRC needed on each mip level so the blit can read from level i-1
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent        = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    ici.mipLevels     = mipLevels;
    ici.arrayLayers   = 1;
    ici.format        = fmt;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                       | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(m_ctx.device(), &ici, nullptr, &outImg));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_ctx.device(), outImg, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = m_ctx.findMemoryType(req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_ctx.device(), &mai, nullptr, &outMem));
    vkBindImageMemory(m_ctx.device(), outImg, outMem, 0);

    auto cmd = m_ctx.beginSingleTimeCommands(pool);

    // Transition all levels UNDEFINED → TRANSFER_DST_OPTIMAL
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = outImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1 };
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&b);
    }

    // Copy staging data into mip level 0
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    vkCmdCopyBufferToImage(cmd, staging.handle(), outImg,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Generate mip chain: blit level i-1 → level i, then transition i-1 to SHADER_READ_ONLY
    int32_t mipW = w, mipH = h;
    for (uint32_t i = 1; i < mipLevels; ++i) {
        // Level i-1: TRANSFER_DST → TRANSFER_SRC
        VkImageMemoryBarrier toSrc{};
        toSrc.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSrc.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image               = outImg;
        toSrc.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1 };
        toSrc.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSrc.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&toSrc);

        int32_t nextW = std::max(1, mipW / 2);
        int32_t nextH = std::max(1, mipH / 2);

        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
        blit.srcOffsets[0]  = { 0, 0, 0 };
        blit.srcOffsets[1]  = { mipW, mipH, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
        blit.dstOffsets[0]  = { 0, 0, 0 };
        blit.dstOffsets[1]  = { nextW, nextH, 1 };
        vkCmdBlitImage(cmd,
            outImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            outImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // Level i-1: TRANSFER_SRC → SHADER_READ_ONLY
        VkImageMemoryBarrier toRead{};
        toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toRead.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toRead.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image               = outImg;
        toRead.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1 };
        toRead.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        toRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&toRead);

        mipW = nextW;
        mipH = nextH;
    }

    // Transition last mip level: TRANSFER_DST → SHADER_READ_ONLY
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = outImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1, 0, 1 };
        b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&b);
    }

    m_ctx.endSingleTimeCommands(pool, cmd);

    VkImageViewCreateInfo ivci{};
    ivci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image                           = outImg;
    ivci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format                          = fmt;
    ivci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount     = mipLevels;
    ivci.subresourceRange.layerCount     = 1;
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(m_ctx.device(), &ivci, nullptr, &view));

    std::cout << "[PBR] Loaded texture: " << fs::path(path).filename().string()
              << " (" << w << "x" << h << ", " << mipLevels << " mips, "
              << (srgb ? "sRGB" : "linear") << ")\n";
    return view;
}

// ---------------------------------------------------------------------------
// IBL: GGX BRDF integration LUT (Hammersley + Smith-GGX split-sum)
// Stores (scale, bias) → R8G8 of RGBA8 image at (NdotV, roughness)
// ---------------------------------------------------------------------------
static float radicalInverse_VdC(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

void PBRModel::generateBRDFLut(VkCommandPool pool) {
    constexpr uint32_t SZ  = 256;
    constexpr uint32_t NSP = 512;
    constexpr float PI = 3.14159265358979f;

    std::vector<uint8_t> lutData(SZ * SZ * 4);

    for (uint32_t y = 0; y < SZ; ++y) {
        float roughness = (y + 0.5f) / float(SZ);
        float a  = roughness * roughness;
        float a2 = a * a;

        for (uint32_t x = 0; x < SZ; ++x) {
            float NdotV = (x + 0.5f) / float(SZ);
            // View vector in tangent space (N = Z axis)
            float Vx = std::sqrt(1.0f - NdotV * NdotV);
            float Vz = NdotV;

            float A = 0.0f, B = 0.0f;
            for (uint32_t i = 0; i < NSP; ++i) {
                float u1 = float(i) / float(NSP);
                float u2 = radicalInverse_VdC(i);

                // GGX importance sample half-vector
                float phi      = 2.0f * PI * u1;
                float cosTheta = std::sqrt((1.0f - u2) / (1.0f + (a2 - 1.0f) * u2));
                float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
                float Hx = std::cos(phi) * sinTheta;
                float Hz = cosTheta;

                // L = reflect(-V, H)
                float VdotH = Vx * Hx + Vz * Hz;  // Vy=0
                float Lz = 2.0f * VdotH * Hz - Vz;
                float NdotL = std::max(Lz, 0.0f);
                float NdotH = std::max(Hz, 0.0f);
                float vdotH = std::max(VdotH, 0.0f);

                if (NdotL > 0.0f) {
                    // IBL roughness remapping: k = a/2 (vs direct: k=(r+1)^2/8)
                    float k   = a / 2.0f;
                    float G_V = NdotV / (NdotV * (1.0f - k) + k);
                    float G_L = NdotL / (NdotL * (1.0f - k) + k);
                    float G_vis = G_V * G_L * vdotH / (NdotH * NdotV + 1e-6f);
                    float Fc    = std::pow(1.0f - vdotH, 5.0f);
                    A += (1.0f - Fc) * G_vis;
                    B += Fc          * G_vis;
                }
            }
            A /= float(NSP);
            B /= float(NSP);

            uint32_t idx = (y * SZ + x) * 4;
            lutData[idx+0] = static_cast<uint8_t>(std::min(std::max(A, 0.0f), 1.0f) * 255.0f);
            lutData[idx+1] = static_cast<uint8_t>(std::min(std::max(B, 0.0f), 1.0f) * 255.0f);
            lutData[idx+2] = 0;
            lutData[idx+3] = 255;
        }
    }

    VkDeviceSize sz = SZ * SZ * 4;
    Buffer staging(m_ctx, sz,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.writeHostVisible(lutData.data(), sz);

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent        = { SZ, SZ, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(m_ctx.device(), &ici, nullptr, &m_brdfLutImg));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_ctx.device(), m_brdfLutImg, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = m_ctx.findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_ctx.device(), &mai, nullptr, &m_brdfLutMem));
    vkBindImageMemory(m_ctx.device(), m_brdfLutImg, m_brdfLutMem, 0);

    auto cmd = m_ctx.beginSingleTimeCommands(pool);
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = m_brdfLutImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&b);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { SZ, SZ, 1 };
    vkCmdCopyBufferToImage(cmd, staging.handle(), m_brdfLutImg,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = m_brdfLutImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&b);
    }
    m_ctx.endSingleTimeCommands(pool, cmd);

    VkImageViewCreateInfo ivci{};
    ivci.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image                       = m_brdfLutImg;
    ivci.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format                      = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(m_ctx.device(), &ivci, nullptr, &m_brdfLutView));
    std::cout << "[PBR] Generated BRDF LUT (" << SZ << "x" << SZ << ", "
              << NSP << " samples)\n";
}

// ---------------------------------------------------------------------------
// IBL: Load equirectangular env map from the model directory
// Looks for any *env* or *Env* file (TGA, PNG, HDR) in the FBX directory.
// ---------------------------------------------------------------------------
void PBRModel::loadEnvMap(VkCommandPool pool) {
    static const char* kExts[] = { ".tga", ".png", ".jpg", ".hdr", nullptr };

    // First try canonical name used by XPS exports
    for (const char** ext = kExts; *ext; ++ext) {
        std::string candidate = m_dir + "/Gen_Env" + *ext;
        if (fs::exists(candidate)) {
            m_envEquirectView = loadTextureFile(candidate, false,
                                                m_envEquirectImg, m_envEquirectMem, pool);
            if (m_envEquirectView) return;
        }
    }

    // Broad search: any filename containing "env" (case-insensitive)
    try {
        for (auto& entry : fs::directory_iterator(m_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string fn = entry.path().filename().string();
            std::string fnLow = fn;
            std::transform(fnLow.begin(), fnLow.end(), fnLow.begin(), ::tolower);
            if (fnLow.find("env") == std::string::npos) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            bool validExt = (ext == ".tga" || ext == ".png" || ext == ".jpg" || ext == ".hdr");
            if (!validExt) continue;

            m_envEquirectView = loadTextureFile(entry.path().string(), false,
                                                m_envEquirectImg, m_envEquirectMem, pool);
            if (m_envEquirectView) return;
        }
    } catch (...) {}

    std::cout << "[PBR] No env map found in '" << m_dir << "' — IBL uses white fallback\n";
}

// ---------------------------------------------------------------------------
// SPV loader used by IBL compute passes
// ---------------------------------------------------------------------------
static VkShaderModule loadSpvModulePBR(VkDevice device, const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open IBL shader: " + path);
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

// ---------------------------------------------------------------------------
// IBL: Run GPU compute passes to build prefiltered + irradiance cubemaps
// ---------------------------------------------------------------------------
void PBRModel::buildIBLCubemaps(VkCommandPool pool) {
    if (!m_envEquirectView) {
        std::cout << "[PBR] No env map — skipping IBL cubemap generation\n";
        return;
    }

    const std::string kEnvToCubeSpv  = "shaders/env_to_cube.comp.spv";
    const std::string kPrefilterSpv  = "shaders/ibl_prefilter.comp.spv";
    const std::string kIrradianceSpv = "shaders/ibl_irradiance.comp.spv";

    for (const auto& p : { kEnvToCubeSpv, kPrefilterSpv, kIrradianceSpv }) {
        if (!fs::exists(p)) {
            std::cerr << "[PBR] IBL SPV not found: " << p << " — skipping\n";
            return;
        }
    }

    VkDevice dev = m_ctx.device();
    constexpr uint32_t kCubeSize = 512;
    constexpr uint32_t kPrefSize = 256;
    constexpr uint32_t kPrefMips = 8;   // roughness 0..1 across 8 mip levels
    constexpr uint32_t kIrrSize  = 32;
    constexpr VkFormat kFmt      = VK_FORMAT_R16G16B16A16_SFLOAT;

    // --- Create cubemap image helper ---
    auto createCubeImage = [&](uint32_t size, uint32_t mips,
                                VkImage& img, VkDeviceMemory& mem) {
        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.extent        = { size, size, 1 };
        ici.mipLevels     = mips;
        ici.arrayLayers   = 6;
        ici.format        = kFmt;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateImage(dev, &ici, nullptr, &img));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, img, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = m_ctx.findMemoryType(req.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        VK_CHECK(vkBindImageMemory(dev, img, mem, 0));
    };

    // --- Create image view helper ---
    auto makeView = [&](VkImage img, VkImageViewType type,
                        uint32_t baseMip, uint32_t mipCount) -> VkImageView {
        VkImageViewCreateInfo ivci{};
        ivci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image                           = img;
        ivci.viewType                        = type;
        ivci.format                          = kFmt;
        ivci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.baseMipLevel   = baseMip;
        ivci.subresourceRange.levelCount     = mipCount;
        ivci.subresourceRange.baseArrayLayer = 0;
        ivci.subresourceRange.layerCount     = 6;
        VkImageView v = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(dev, &ivci, nullptr, &v));
        return v;
    };

    // Allocate all cubemap images
    createCubeImage(kCubeSize, 1,        m_envCubeImg,    m_envCubeMem);
    createCubeImage(kPrefSize, kPrefMips, m_prefilterImg,  m_prefilterMem);
    createCubeImage(kIrrSize,  1,         m_irradianceImg, m_irradianceMem);

    // Persistent views (shader-facing samplerCube views)
    m_envCubeView    = makeView(m_envCubeImg,    VK_IMAGE_VIEW_TYPE_CUBE,       0, 1);
    m_prefilterView  = makeView(m_prefilterImg,  VK_IMAGE_VIEW_TYPE_CUBE,       0, kPrefMips);
    m_irradianceView = makeView(m_irradianceImg, VK_IMAGE_VIEW_TYPE_CUBE,       0, 1);
    for (uint32_t mip = 0; mip < kPrefMips; ++mip)
        m_prefilterMipViews.push_back(makeView(m_prefilterImg,
                                               VK_IMAGE_VIEW_TYPE_2D_ARRAY, mip, 1));

    // Temporary storage-image array views (for compute writes)
    VkImageView envCubeArrayView = makeView(m_envCubeImg,    VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, 1);
    VkImageView irrArrayView     = makeView(m_irradianceImg, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, 1);

    // Cubemap sampler: clamp-to-edge to avoid seams at cube face boundaries
    VkSampler cubeSampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &cubeSampler));
    }

    // --- Create compute pipeline helper (2 bindings: samplerXxx + storageImage) ---
    struct IBLPipeline {
        VkDescriptorSetLayout dsLayout  = VK_NULL_HANDLE;
        VkPipelineLayout      layout    = VK_NULL_HANDLE;
        VkPipeline            pipeline  = VK_NULL_HANDLE;
    };

    auto makeIBLPipeline = [&](const std::string& spvPath, bool pushConst) -> IBLPipeline {
        IBLPipeline p;
        // DS layout: binding 0 = sampledImage, binding 1 = storageImage
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings    = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &p.dsLayout));

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = sizeof(float);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &p.dsLayout;
        plci.pushConstantRangeCount = pushConst ? 1u : 0u;
        plci.pPushConstantRanges    = pushConst ? &pc : nullptr;
        VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &p.layout));

        VkShaderModule mod = loadSpvModulePBR(dev, spvPath);
        VkComputePipelineCreateInfo cpci{};
        cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = mod;
        cpci.stage.pName  = "main";
        cpci.layout       = p.layout;
        VK_CHECK(vkCreateComputePipelines(dev, m_ctx.pipelineCache(), 1, &cpci, nullptr,
                                          &p.pipeline));
        vkDestroyShaderModule(dev, mod, nullptr);
        return p;
    };

    auto destroyIBLPipeline = [&](IBLPipeline& p) {
        if (p.pipeline) vkDestroyPipeline(dev, p.pipeline, nullptr);
        if (p.layout)   vkDestroyPipelineLayout(dev, p.layout, nullptr);
        if (p.dsLayout) vkDestroyDescriptorSetLayout(dev, p.dsLayout, nullptr);
        p = {};
    };

    IBLPipeline envToCubePL  = makeIBLPipeline(kEnvToCubeSpv,  false);
    IBLPipeline prefilterPL  = makeIBLPipeline(kPrefilterSpv,   true);
    IBLPipeline irradiancePL = makeIBLPipeline(kIrradianceSpv,  false);

    // --- Descriptor pools and sets ---
    // env_to_cube: 1 set; irradiance: 1 set; prefilter: kPrefMips sets (one per mip)
    auto makePool = [&](VkDescriptorSetLayout layout, uint32_t nSets) -> VkDescriptorPool {
        std::array<VkDescriptorPoolSize, 2> ps{{ { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nSets },
                                                  { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, nSets } }};
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.poolSizeCount = 2;
        pci.pPoolSizes    = ps.data();
        pci.maxSets       = nSets;
        VkDescriptorPool pool2 = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &pool2));
        (void)layout;
        return pool2;
    };

    VkDescriptorPool envToCubePool  = makePool(envToCubePL.dsLayout,  1);
    VkDescriptorPool prefilterPool  = makePool(prefilterPL.dsLayout,  kPrefMips);
    VkDescriptorPool irradiancePool = makePool(irradiancePL.dsLayout, 1);

    auto allocSets = [&](VkDescriptorPool p, VkDescriptorSetLayout layout,
                         uint32_t n, std::vector<VkDescriptorSet>& sets) {
        sets.resize(n);
        std::vector<VkDescriptorSetLayout> layouts(n, layout);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = p;
        ai.descriptorSetCount = n;
        ai.pSetLayouts        = layouts.data();
        VK_CHECK(vkAllocateDescriptorSets(dev, &ai, sets.data()));
    };

    std::vector<VkDescriptorSet> envToCubeSets, prefilterSets, irradianceSets;
    allocSets(envToCubePool,  envToCubePL.dsLayout,  1,         envToCubeSets);
    allocSets(prefilterPool,  prefilterPL.dsLayout,  kPrefMips, prefilterSets);
    allocSets(irradiancePool, irradiancePL.dsLayout, 1,         irradianceSets);

    // Write env_to_cube descriptor (equirect → cube array storage)
    {
        VkDescriptorImageInfo in  = { m_sampler, m_envEquirectView,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo out = { VK_NULL_HANDLE, envCubeArrayView,
                                      VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = envToCubeSets[0]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &in;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = envToCubeSets[0]; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &out;
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    }

    // Write prefilter descriptors (one per mip, each with its own mip-level storage view)
    for (uint32_t mip = 0; mip < kPrefMips; ++mip) {
        VkDescriptorImageInfo in  = { cubeSampler, m_envCubeView,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo out = { VK_NULL_HANDLE, m_prefilterMipViews[mip],
                                      VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = prefilterSets[mip]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &in;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = prefilterSets[mip]; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &out;
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    }

    // Write irradiance descriptor
    {
        VkDescriptorImageInfo in  = { cubeSampler, m_envCubeView,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo out = { VK_NULL_HANDLE, irrArrayView,
                                      VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = irradianceSets[0]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &in;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = irradianceSets[0]; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &out;
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    }

    // --- Record and submit one command buffer for all IBL passes ---
    auto cmd = m_ctx.beginSingleTimeCommands(pool);

    auto imgBarrier = [&](VkImage img, uint32_t mips,
                          VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                          VkImageLayout oldLayout, VkImageLayout newLayout,
                          VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = srcAccess;
        b.dstAccessMask       = dstAccess;
        b.oldLayout           = oldLayout;
        b.newLayout           = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 6 };
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // Transition all outputs UNDEFINED → GENERAL for compute writes
    imgBarrier(m_envCubeImg,    1,        0, VK_ACCESS_SHADER_WRITE_BIT,
               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    imgBarrier(m_prefilterImg,  kPrefMips, 0, VK_ACCESS_SHADER_WRITE_BIT,
               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    imgBarrier(m_irradianceImg, 1,        0, VK_ACCESS_SHADER_WRITE_BIT,
               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Pass 1: env_to_cube — equirectangular → source cubemap
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, envToCubePL.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, envToCubePL.layout,
                            0, 1, &envToCubeSets[0], 0, nullptr);
    vkCmdDispatch(cmd, (kCubeSize + 15) / 16, (kCubeSize + 15) / 16, 6);

    // Source cubemap GENERAL → SHADER_READ_ONLY for prefilter + irradiance reads
    imgBarrier(m_envCubeImg, 1,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Pass 2: GGX prefilter — one dispatch per roughness mip
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterPL.pipeline);
    for (uint32_t mip = 0; mip < kPrefMips; ++mip) {
        float    roughness = float(mip) / float(kPrefMips - 1);
        uint32_t mipSize   = std::max(1u, kPrefSize >> mip);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterPL.layout,
                                0, 1, &prefilterSets[mip], 0, nullptr);
        vkCmdPushConstants(cmd, prefilterPL.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(float), &roughness);
        vkCmdDispatch(cmd, std::max(1u, (mipSize + 15) / 16),
                           std::max(1u, (mipSize + 15) / 16), 6);
    }

    // Prefiltered cubemap GENERAL → SHADER_READ_ONLY
    imgBarrier(m_prefilterImg, kPrefMips,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    // Pass 3: diffuse irradiance — hemisphere integration
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradiancePL.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradiancePL.layout,
                            0, 1, &irradianceSets[0], 0, nullptr);
    vkCmdDispatch(cmd, (kIrrSize + 15) / 16, (kIrrSize + 15) / 16, 6);

    // Irradiance cubemap GENERAL → SHADER_READ_ONLY
    imgBarrier(m_irradianceImg, 1,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    m_ctx.endSingleTimeCommands(pool, cmd);

    // Cleanup temporary objects
    vkDestroyDescriptorPool(dev, envToCubePool,  nullptr);
    vkDestroyDescriptorPool(dev, prefilterPool,  nullptr);
    vkDestroyDescriptorPool(dev, irradiancePool, nullptr);
    vkDestroyImageView(dev, envCubeArrayView, nullptr);
    vkDestroyImageView(dev, irrArrayView,     nullptr);
    vkDestroySampler(dev, cubeSampler, nullptr);
    destroyIBLPipeline(envToCubePL);
    destroyIBLPipeline(prefilterPL);
    destroyIBLPipeline(irradiancePL);

    std::cout << "[PBR] IBL cubemaps built: "
              << kCubeSize << "x" << kCubeSize << " source cube, "
              << kPrefSize << "x" << kPrefSize << " prefiltered (" << kPrefMips << " mips), "
              << kIrrSize << "x" << kIrrSize << " irradiance\n";
}

// ---------------------------------------------------------------------------
// Texture path resolution
// ---------------------------------------------------------------------------
std::string PBRModel::resolveTexPath(const std::string& raw) const {
    if (raw.empty()) return {};
    if (fs::exists(raw)) return raw;

    std::string filename = fs::path(raw).filename().string();
    std::string candidate = m_dir + "/" + filename;
    if (fs::exists(candidate)) return candidate;

    candidate = m_dir + "/textures/" + filename;
    if (fs::exists(candidate)) return candidate;

    candidate = m_dir + "/Texture/" + filename;
    if (fs::exists(candidate)) return candidate;

    return {};
}

static int detectSlotFromFilename(const std::string& filename) {
    std::string upper = filename;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    if (upper.find("_AO")   != std::string::npos) return kPBRAO;
    if (upper.find("_ATOC") != std::string::npos) return kPBRAlbedo;
    if (upper.find("_D.")   != std::string::npos ||
        upper.rfind("_D")   == upper.size() - 2)  return kPBRAlbedo;
    if (upper.find("_N.")   != std::string::npos ||
        upper.rfind("_N")   == upper.size() - 2)  return kPBRNormal;
    if (upper.find("_R.")   != std::string::npos ||
        upper.rfind("_R")   == upper.size() - 2)  return kPBRRoughness;
    if (upper.find("_M.")   != std::string::npos ||
        upper.rfind("_M")   == upper.size() - 2)  return kPBRMetallic;
    if (upper.find("_S.")   != std::string::npos ||
        upper.rfind("_S")   == upper.size() - 2)  return kPBRRoughness;
    return -1;
}

// ---------------------------------------------------------------------------
// Main load function
// ---------------------------------------------------------------------------
bool PBRModel::load(const std::string& fbxPath, VkCommandPool pool) {
    m_dir = fs::path(fbxPath).parent_path().string();

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(fbxPath,
        aiProcess_Triangulate          |
        aiProcess_FlipUVs              |
        aiProcess_CalcTangentSpace     |
        aiProcess_GenSmoothNormals     |
        aiProcess_JoinIdenticalVertices);

    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        std::cerr << "[PBR] Assimp error: " << importer.GetErrorString() << "\n";
        return false;
    }

    createFallbackTextures(pool);

    // Sampler: linear filtering + full mip range + repeat addressing
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sci.maxLod       = VK_LOD_CLAMP_NONE;  // exposes all generated mip levels
    VK_CHECK(vkCreateSampler(m_ctx.device(), &sci, nullptr, &m_sampler));

    // -----------------------------------------------------------------------
    // Process materials
    // -----------------------------------------------------------------------
    m_materials.resize(scene->mNumMaterials);
    for (unsigned mi = 0; mi < scene->mNumMaterials; ++mi) {
        aiMaterial* aiMat = scene->mMaterials[mi];
        aiString name;
        aiMat->Get(AI_MATKEY_NAME, name);
        m_materials[mi].name = name.C_Str();

        static const aiTextureType kAllTypes[] = {
            aiTextureType_DIFFUSE, aiTextureType_SPECULAR, aiTextureType_AMBIENT,
            aiTextureType_NORMALS, aiTextureType_HEIGHT,
            aiTextureType_SHININESS, aiTextureType_OPACITY,
            aiTextureType_METALNESS, aiTextureType_DIFFUSE_ROUGHNESS,
            aiTextureType_AMBIENT_OCCLUSION, aiTextureType_UNKNOWN
        };

        std::vector<std::string> slotPaths(kPBRTexCount);

        for (aiTextureType type : kAllTypes) {
            unsigned count = aiMat->GetTextureCount(type);
            for (unsigned ti = 0; ti < count; ++ti) {
                aiString texPath;
                if (aiMat->GetTexture(type, ti, &texPath) != AI_SUCCESS) continue;
                if (texPath.length == 0 || texPath.data[0] == '*') continue;

                std::string resolved = resolveTexPath(texPath.C_Str());
                if (resolved.empty()) {
                    std::cerr << "[PBR] Texture not found: " << texPath.C_Str() << "\n";
                    continue;
                }

                int slot = detectSlotFromFilename(fs::path(resolved).filename().string());
                if (slot < 0) {
                    if      (type == aiTextureType_DIFFUSE)           slot = kPBRAlbedo;
                    else if (type == aiTextureType_NORMALS ||
                             type == aiTextureType_HEIGHT)             slot = kPBRNormal;
                    else if (type == aiTextureType_SHININESS ||
                             type == aiTextureType_DIFFUSE_ROUGHNESS)  slot = kPBRRoughness;
                    else if (type == aiTextureType_METALNESS ||
                             type == aiTextureType_SPECULAR)           slot = kPBRMetallic;
                    else if (type == aiTextureType_AMBIENT_OCCLUSION ||
                             type == aiTextureType_AMBIENT)            slot = kPBRAO;
                    else continue;
                }
                if (m_materials[mi].hasMap[slot]) continue;

                bool srgb = (slot == kPBRAlbedo);
                VkImageView view = VK_NULL_HANDLE;
                auto cacheIt = m_texCache.find(resolved);
                if (cacheIt != m_texCache.end()) {
                    view = cacheIt->second.view;
                } else {
                    VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
                    view = loadTextureFile(resolved, srgb, img, mem, pool);
                    if (view != VK_NULL_HANDLE)
                        m_texCache[resolved] = { img, mem, view };
                }
                if (view != VK_NULL_HANDLE) {
                    m_materials[mi].views[slot]  = view;
                    m_materials[mi].hasMap[slot] = true;
                    slotPaths[slot]              = resolved;
                }
            }
        }

        // Base-name inference: recover missing slots from stems of loaded textures
        static const char* kStripSuffixes[] = {
            "_N", "_AO", "_R", "_M", "_S", "_D", "_D-cut", "_ATOC", nullptr
        };
        struct SlotProbe {
            int         slot;
            const char* suffixes[4];
            bool        srgb;
        };
        static const SlotProbe kProbes[] = {
            { kPBRAlbedo,    {"_D.png",  "_D.tga",  "_D-cut.png", "_ATOC.tga"}, true  },
            { kPBRRoughness, {"_R.png",  "_R.tga",  nullptr,      nullptr     }, false },
            { kPBRMetallic,  {"_M.png",  "_M.tga",  nullptr,      nullptr     }, false },
            { kPBRAO,        {"_AO.png", "_AO.tga", nullptr,      nullptr     }, false },
            { kPBRNormal,    {"_N.png",  "_N.tga",  nullptr,      nullptr     }, false },
        };

        std::vector<std::string> bases;
        for (int t = 0; t < kPBRTexCount; ++t) {
            if (slotPaths[t].empty()) continue;
            std::string stem = fs::path(slotPaths[t]).stem().string();
            for (const char** s = kStripSuffixes; *s; ++s) {
                std::string suf(*s);
                if (stem.size() > suf.size() &&
                    stem.substr(stem.size() - suf.size()) == suf) {
                    std::string base = stem.substr(0, stem.size() - suf.size());
                    if (base.size() >= 4) bases.push_back(base);
                    break;
                }
            }
        }

        for (auto& probe : kProbes) {
            if (m_materials[mi].hasMap[probe.slot]) continue;
            for (auto& base : bases) {
                bool found = false;
                for (int ci = 0; probe.suffixes[ci] && !found; ++ci) {
                    std::string candidate = m_dir + "/" + base + probe.suffixes[ci];
                    if (!fs::exists(candidate)) continue;

                    VkImageView view = VK_NULL_HANDLE;
                    auto cacheIt = m_texCache.find(candidate);
                    if (cacheIt != m_texCache.end()) {
                        view = cacheIt->second.view;
                    } else {
                        VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
                        view = loadTextureFile(candidate, probe.srgb, img, mem, pool);
                        if (view != VK_NULL_HANDLE)
                            m_texCache[candidate] = { img, mem, view };
                    }
                    if (view != VK_NULL_HANDLE) {
                        m_materials[mi].views[probe.slot]  = view;
                        m_materials[mi].hasMap[probe.slot] = true;
                        found = true;
                    }
                }
                if (m_materials[mi].hasMap[probe.slot]) break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Process mesh nodes — accumulate into consolidated VBO/IBO for GPU-driven indirect
    // -----------------------------------------------------------------------
    std::vector<PBRVertex> allVerts;
    std::vector<uint32_t>  allIndices;

    std::function<void(const aiNode*)> processNode = [&](const aiNode* node) {
        for (unsigned mi = 0; mi < node->mNumMeshes; ++mi) {
            const aiMesh* mesh = scene->mMeshes[node->mMeshes[mi]];

            std::vector<PBRVertex> verts;
            std::vector<uint32_t>  indices;
            verts.reserve(mesh->mNumVertices);

            for (unsigned vi = 0; vi < mesh->mNumVertices; ++vi) {
                PBRVertex v{};
                v.pos[0] = mesh->mVertices[vi].x;
                v.pos[1] = mesh->mVertices[vi].y;
                v.pos[2] = mesh->mVertices[vi].z;

                if (mesh->HasNormals()) {
                    v.normal[0] = mesh->mNormals[vi].x;
                    v.normal[1] = mesh->mNormals[vi].y;
                    v.normal[2] = mesh->mNormals[vi].z;
                }
                if (mesh->mTextureCoords[0]) {
                    v.uv[0] = mesh->mTextureCoords[0][vi].x;
                    v.uv[1] = mesh->mTextureCoords[0][vi].y;
                }
                if (mesh->HasTangentsAndBitangents()) {
                    v.tangent[0]   = mesh->mTangents[vi].x;
                    v.tangent[1]   = mesh->mTangents[vi].y;
                    v.tangent[2]   = mesh->mTangents[vi].z;
                    v.bitangent[0] = mesh->mBitangents[vi].x;
                    v.bitangent[1] = mesh->mBitangents[vi].y;
                    v.bitangent[2] = mesh->mBitangents[vi].z;
                } else {
                    v.tangent[0] = 1.0f;
                    v.bitangent[1] = 1.0f;
                }
                verts.push_back(v);
            }

            for (unsigned fi = 0; fi < mesh->mNumFaces; ++fi) {
                const aiFace& face = mesh->mFaces[fi];
                for (unsigned ii = 0; ii < face.mNumIndices; ++ii)
                    indices.push_back(face.mIndices[ii]);
            }

            if (verts.empty() || indices.empty()) continue;

            PBRSubmesh submesh;
            submesh.name         = mesh->mName.C_Str();
            submesh.materialIdx  = static_cast<int>(mesh->mMaterialIndex);
            submesh.indexCount   = static_cast<uint32_t>(indices.size());
            submesh.firstIndex   = static_cast<uint32_t>(allIndices.size());
            submesh.vertexOffset = static_cast<int32_t>(allVerts.size());

            allVerts.insert(allVerts.end(), verts.begin(), verts.end());
            allIndices.insert(allIndices.end(), indices.begin(), indices.end());

            std::cout << "[PBR] Submesh '" << submesh.name << "': "
                      << verts.size() << " verts, "
                      << indices.size() / 3 << " tris, mat=" << submesh.materialIdx << "\n";

            m_submeshes.push_back(std::move(submesh));
        }
        for (unsigned ci = 0; ci < node->mNumChildren; ++ci)
            processNode(node->mChildren[ci]);
    };

    processNode(scene->mRootNode);

    if (m_submeshes.empty()) {
        std::cerr << "[PBR] No submeshes loaded from '" << fbxPath << "'\n";
        return false;
    }

    // Upload consolidated geometry buffers
    {
        VkDeviceSize vsize = allVerts.size()   * sizeof(PBRVertex);
        VkDeviceSize isize = allIndices.size() * sizeof(uint32_t);

        m_consolidatedVBO = std::make_unique<Buffer>(m_ctx, vsize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        m_consolidatedVBO->upload(pool, allVerts.data(), vsize);

        m_consolidatedIBO = std::make_unique<Buffer>(m_ctx, isize,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        m_consolidatedIBO->upload(pool, allIndices.data(), isize);
    }

    // Build and upload GPU-driven indirect command buffer
    // STORAGE_BUFFER_BIT: cull compute shader reads from this as an SSBO
    {
        std::vector<VkDrawIndexedIndirectCommand> indirects(m_submeshes.size());
        for (size_t s = 0; s < m_submeshes.size(); ++s) {
            indirects[s].indexCount    = m_submeshes[s].indexCount;
            indirects[s].instanceCount = 1;
            indirects[s].firstIndex    = m_submeshes[s].firstIndex;
            indirects[s].vertexOffset  = m_submeshes[s].vertexOffset;
            indirects[s].firstInstance = 0;
        }
        VkDeviceSize indSize = m_submeshes.size() * sizeof(VkDrawIndexedIndirectCommand);
        m_indirectBuffer = std::make_unique<Buffer>(m_ctx, indSize,
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        m_indirectBuffer->upload(pool, indirects.data(), indSize);
    }

    // Build per-submesh world-space bounding spheres for GPU frustum culling.
    // Apply the constant XPS model matrix (scale 0.01, rotate -90° X) at load time
    // so the culling compute shader only needs the VP matrix.
    {
        // Model matrix transform (applied to local AABB centre and radius):
        //   world.x = 0.01 * local.x
        //   world.y = 0.01 * local.z    (Rx(-90): y' = z)
        //   world.z = 0.01 * (-local.y) (Rx(-90): z' = -y)
        const float kScale = 0.01f;
        struct Sphere { float x, y, z, r; };
        std::vector<Sphere> spheres(m_submeshes.size());

        size_t globalVertBase = 0;
        for (size_t s = 0; s < m_submeshes.size(); ++s) {
            int32_t  vo   = m_submeshes[s].vertexOffset;
            uint32_t ic   = m_submeshes[s].indexCount;
            uint32_t fi   = m_submeshes[s].firstIndex;

            float minX =  1e30f, maxX = -1e30f;
            float minY =  1e30f, maxY = -1e30f;
            float minZ =  1e30f, maxZ = -1e30f;

            for (uint32_t idx = fi; idx < fi + ic; ++idx) {
                uint32_t vi = allIndices[idx] + static_cast<uint32_t>(vo);
                const float* p = allVerts[vi].pos;
                if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
                if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
                if (p[2] < minZ) minZ = p[2]; if (p[2] > maxZ) maxZ = p[2];
            }
            float cx = (minX + maxX) * 0.5f;
            float cy = (minY + maxY) * 0.5f;
            float cz = (minZ + maxZ) * 0.5f;

            float radius = 0.0f;
            for (uint32_t idx = fi; idx < fi + ic; ++idx) {
                uint32_t vi = allIndices[idx] + static_cast<uint32_t>(vo);
                const float* p = allVerts[vi].pos;
                float dx = p[0]-cx, dy = p[1]-cy, dz = p[2]-cz;
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 > radius) radius = d2;
            }
            radius = std::sqrt(radius);

            // Transform to world space using the fixed model matrix
            spheres[s].x = kScale * cx;
            spheres[s].y = kScale * cz;    // Rx(-90): y' = z
            spheres[s].z = kScale * (-cy); // Rx(-90): z' = -y
            spheres[s].r = kScale * radius;
        }

        VkDeviceSize sphereSize = m_submeshes.size() * sizeof(Sphere);
        m_sphereBuffer = std::make_unique<Buffer>(m_ctx, sphereSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        m_sphereBuffer->upload(pool, spheres.data(), sphereSize);
        (void)globalVertBase;
    }

    // IBL resources (generated after sampler is ready)
    generateBRDFLut(pool);
    loadEnvMap(pool);
    buildIBLCubemaps(pool);

    std::cout << "[PBR] Loaded '" << fs::path(fbxPath).filename().string() << "': "
              << m_submeshes.size() << " submeshes, "
              << allVerts.size() << " total verts, "
              << m_materials.size() << " materials\n";
    return true;
}

// ---------------------------------------------------------------------------
// Descriptor resources (call after load())
// ---------------------------------------------------------------------------
void PBRModel::createDescriptorResources(VkDescriptorSetLayout pbrDSLayout, int framesInFlight) {
    m_framesInFlight = framesInFlight;
    uint32_t nSets   = static_cast<uint32_t>(m_submeshes.size() * framesInFlight);

    for (int f = 0; f < framesInFlight; ++f) {
        m_uboBuffers.push_back(std::make_unique<Buffer>(m_ctx,
            sizeof(PBRUBOData),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    }

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = nSets;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = nSets * kTotalTexBindings;  // 5 material + 3 IBL

    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    pci.pPoolSizes    = poolSizes.data();
    pci.maxSets       = nSets;
    VK_CHECK(vkCreateDescriptorPool(m_ctx.device(), &pci, nullptr, &m_descPool));

    std::vector<VkDescriptorSetLayout> layouts(nSets, pbrDSLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = m_descPool;
    ai.descriptorSetCount = nSets;
    ai.pSetLayouts        = layouts.data();
    m_descSets.resize(nSets);
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.device(), &ai, m_descSets.data()));

    // IBL views — fall back to white if cubemap generation was skipped
    VkImageView prefilterView  = m_prefilterView  ? m_prefilterView  : m_fbWhiteView;
    VkImageView brdfView       = m_brdfLutView    ? m_brdfLutView    : m_fbWhiteView;
    VkImageView irradianceView = m_irradianceView ? m_irradianceView : m_fbWhiteView;

    for (int s = 0; s < static_cast<int>(m_submeshes.size()); ++s) {
        int matIdx = m_submeshes[s].materialIdx;

        for (int f = 0; f < framesInFlight; ++f) {
            VkDescriptorSet ds = m_descSets[s * framesInFlight + f];

            VkDescriptorBufferInfo bi{};
            bi.buffer = m_uboBuffers[f]->handle();
            bi.offset = 0;
            bi.range  = sizeof(PBRUBOData);

            // Bindings 1-5: per-material textures
            VkDescriptorImageInfo imgInfos[kTotalTexBindings]{};
            for (int t = 0; t < kPBRTexCount; ++t) {
                imgInfos[t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imgInfos[t].sampler     = m_sampler;
                VkImageView view = VK_NULL_HANDLE;
                if (matIdx >= 0 && matIdx < static_cast<int>(m_materials.size())
                    && m_materials[matIdx].hasMap[t]) {
                    view = m_materials[matIdx].views[t];
                }
                imgInfos[t].imageView = (view != VK_NULL_HANDLE)
                    ? view
                    : (t == kPBRNormal ? m_fbNormView : m_fbWhiteView);
            }
            // Bindings 6-8: IBL (prefiltered specular cube, BRDF LUT, diffuse irradiance cube)
            imgInfos[kPBRTexCount + 0] = { m_sampler, prefilterView,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            imgInfos[kPBRTexCount + 1] = { m_sampler, brdfView,       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            imgInfos[kPBRTexCount + 2] = { m_sampler, irradianceView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            std::array<VkWriteDescriptorSet, 1 + kTotalTexBindings> writes{};
            writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet          = ds;
            writes[0].dstBinding      = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo     = &bi;

            for (int t = 0; t < kTotalTexBindings; ++t) {
                writes[1 + t].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1 + t].dstSet          = ds;
                writes[1 + t].dstBinding      = static_cast<uint32_t>(1 + t);
                writes[1 + t].descriptorCount = 1;
                writes[1 + t].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[1 + t].pImageInfo      = &imgInfos[t];
            }

            vkUpdateDescriptorSets(m_ctx.device(),
                static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame UBO update
// ---------------------------------------------------------------------------
void PBRModel::updateUBO(int frameIdx, const float* view16, const float* proj16,
                          const float* cameraPos4, const float* lightDir4,
                          const float* lightColor4) {
    PBRUBOData data{};
    std::memcpy(data.view,       view16,      64);
    std::memcpy(data.proj,       proj16,      64);
    std::memcpy(data.cameraPos,  cameraPos4,  16);
    std::memcpy(data.lightDir,   lightDir4,   16);
    std::memcpy(data.lightColor, lightColor4, 16);
    m_uboBuffers[frameIdx]->writeHostVisible(&data, sizeof(data));
}

VkDescriptorSet PBRModel::descriptorSet(int submeshIdx, int frameIdx) const {
    return m_descSets[submeshIdx * m_framesInFlight + frameIdx];
}

VkBuffer PBRModel::consolidatedVBO() const { return m_consolidatedVBO->handle(); }
VkBuffer PBRModel::consolidatedIBO() const { return m_consolidatedIBO->handle(); }
VkBuffer PBRModel::indirectBuffer()  const { return m_indirectBuffer->handle();  }
VkBuffer PBRModel::sphereBuffer()    const { return m_sphereBuffer->handle();    }

} // namespace tgt
