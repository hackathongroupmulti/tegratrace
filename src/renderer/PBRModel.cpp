#include "PBRModel.h"
#include "Buffer.h"
#include "core/VulkanContext.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <stb_image.h>

#include <filesystem>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cctype>

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
    // Wait for GPU before releasing any resource
    vkDeviceWaitIdle(m_ctx.device());

    if (m_descPool) vkDestroyDescriptorPool(m_ctx.device(), m_descPool, nullptr);
    m_uboBuffers.clear();

    destroyMaterials();
    m_submeshes.clear();

    // Fallback textures
    if (m_fbWhiteView) vkDestroyImageView(m_ctx.device(), m_fbWhiteView, nullptr);
    if (m_fbWhiteImg)  vkDestroyImage(m_ctx.device(), m_fbWhiteImg, nullptr);
    if (m_fbWhiteMem)  vkFreeMemory(m_ctx.device(), m_fbWhiteMem, nullptr);

    if (m_fbNormView)  vkDestroyImageView(m_ctx.device(), m_fbNormView, nullptr);
    if (m_fbNormImg)   vkDestroyImage(m_ctx.device(), m_fbNormImg, nullptr);
    if (m_fbNormMem)   vkFreeMemory(m_ctx.device(), m_fbNormMem, nullptr);

    if (m_sampler) vkDestroySampler(m_ctx.device(), m_sampler, nullptr);
}

void PBRModel::destroyMaterials() {
    for (auto& mat : m_materials) {
        for (int i = 0; i < kPBRTexCount; ++i) {
            if (mat.views[i])    vkDestroyImageView(m_ctx.device(), mat.views[i], nullptr);
            if (mat.images[i])   vkDestroyImage(m_ctx.device(), mat.images[i], nullptr);
            if (mat.memories[i]) vkFreeMemory(m_ctx.device(), mat.memories[i], nullptr);
        }
    }
    m_materials.clear();
}

// ---------------------------------------------------------------------------
// Small helper: upload a 1×1 RGBA8 pixel to a new device-local image
// ---------------------------------------------------------------------------
static VkImageView makeSmallTexture(VulkanContext& ctx, VkCommandPool pool,
                                    uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                                    VkImage& outImg, VkDeviceMemory& outMem) {
    uint8_t pixels[4] = { r, g, b, a };

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
    VK_CHECK(vkCreateImage(ctx.device(), &ici, nullptr, &outImg));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx.device(), outImg, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = ctx.findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(ctx.device(), &mai, nullptr, &outMem));
    vkBindImageMemory(ctx.device(), outImg, outMem, 0);

    auto cmd = ctx.beginSingleTimeCommands(pool);
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = outImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,nullptr,0,nullptr,1,&b);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { 1, 1, 1 };
    vkCmdCopyBufferToImage(cmd, staging.handle(), outImg,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = outImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,nullptr,0,nullptr,1,&b);
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
    VK_CHECK(vkCreateImageView(ctx.device(), &ivci, nullptr, &view));
    return view;
}

// ---------------------------------------------------------------------------
// Fallback texture creation
// ---------------------------------------------------------------------------
void PBRModel::createFallbackTextures(VkCommandPool pool) {
    // White: albedo/roughness/metallic/AO fallback
    m_fbWhiteView = makeSmallTexture(m_ctx, pool,
        255, 255, 255, 255, m_fbWhiteImg, m_fbWhiteMem);

    // Flat normal: (0,0,1) encoded as (128,128,255)
    m_fbNormView = makeSmallTexture(m_ctx, pool,
        128, 128, 255, 255, m_fbNormImg, m_fbNormMem);
}

// ---------------------------------------------------------------------------
// Texture loading from disk via stb_image
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

    Buffer staging(m_ctx, imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.writeHostVisible(pixels, imageSize);
    stbi_image_free(pixels);

    // Create device-local image
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent        = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.format        = fmt;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = outImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&b);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    vkCmdCopyBufferToImage(cmd, staging.handle(), outImg,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = outImg;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
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
    ivci.subresourceRange.levelCount     = 1;
    ivci.subresourceRange.layerCount     = 1;
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(m_ctx.device(), &ivci, nullptr, &view));

    std::cout << "[PBR] Loaded texture: " << fs::path(path).filename().string()
              << " (" << w << "x" << h << ", " << (srgb ? "sRGB" : "linear") << ")\n";
    return view;
}

// ---------------------------------------------------------------------------
// Texture path resolution
// Assimp may give us: absolute path, relative path, or just a filename.
// We try several locations relative to the FBX's directory.
// ---------------------------------------------------------------------------
std::string PBRModel::resolveTexPath(const std::string& raw) const {
    if (raw.empty()) return {};

    // 1. Try as-is
    if (fs::exists(raw)) return raw;

    // 2. Just the filename, next to the FBX
    std::string filename = fs::path(raw).filename().string();
    std::string candidate = m_dir + "/" + filename;
    if (fs::exists(candidate)) return candidate;

    // 3. In a "textures" subdirectory
    candidate = m_dir + "/textures/" + filename;
    if (fs::exists(candidate)) return candidate;

    // 4. In a "Texture" subdirectory (common in XPS exports)
    candidate = m_dir + "/Texture/" + filename;
    if (fs::exists(candidate)) return candidate;

    return {};
}

// ---------------------------------------------------------------------------
// Suffix-based texture slot detection
// Returns the PBR slot index [0..kPBRTexCount) or -1 if unrecognised.
// ---------------------------------------------------------------------------
static int detectSlotFromFilename(const std::string& filename) {
    // Convert to upper-case for case-insensitive matching
    std::string upper = filename;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    // Test from most specific to least specific
    if (upper.find("_AO")   != std::string::npos) return kPBRAO;
    if (upper.find("_ATOC") != std::string::npos) return kPBRAlbedo; // ATOC treated as albedo alpha
    if (upper.find("_D.")   != std::string::npos ||
        upper.rfind("_D")   == upper.size() - 2)  return kPBRAlbedo;
    if (upper.find("_N.")   != std::string::npos ||
        upper.rfind("_N")   == upper.size() - 2)  return kPBRNormal;
    if (upper.find("_R.")   != std::string::npos ||
        upper.rfind("_R")   == upper.size() - 2)  return kPBRRoughness;
    if (upper.find("_M.")   != std::string::npos ||
        upper.rfind("_M")   == upper.size() - 2)  return kPBRMetallic;
    if (upper.find("_S.")   != std::string::npos ||
        upper.rfind("_S")   == upper.size() - 2)  return kPBRRoughness; // _S specular → roughness
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

    // Create sampler (shared across all textures)
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
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

        // Gather every referenced texture regardless of Assimp semantic type,
        // then classify by filename suffix.
        static const aiTextureType kAllTypes[] = {
            aiTextureType_DIFFUSE, aiTextureType_SPECULAR, aiTextureType_AMBIENT,
            aiTextureType_NORMALS, aiTextureType_HEIGHT,
            aiTextureType_SHININESS, aiTextureType_OPACITY,
            aiTextureType_METALNESS, aiTextureType_DIFFUSE_ROUGHNESS,
            aiTextureType_AMBIENT_OCCLUSION, aiTextureType_UNKNOWN
        };

        for (aiTextureType type : kAllTypes) {
            unsigned count = aiMat->GetTextureCount(type);
            for (unsigned ti = 0; ti < count; ++ti) {
                aiString texPath;
                if (aiMat->GetTexture(type, ti, &texPath) != AI_SUCCESS) continue;
                if (texPath.length == 0) continue;

                // Skip embedded texture references (begin with '*')
                if (texPath.data[0] == '*') continue;

                std::string rawPath = texPath.C_Str();
                std::string resolved = resolveTexPath(rawPath);
                if (resolved.empty()) {
                    std::cerr << "[PBR] Texture not found: " << rawPath << "\n";
                    continue;
                }

                int slot = detectSlotFromFilename(fs::path(resolved).filename().string());
                if (slot < 0) {
                    // Fallback: map Assimp type to slot
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

                if (m_materials[mi].hasMap[slot]) continue; // first match wins

                bool srgb = (slot == kPBRAlbedo);
                VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
                VkImageView view = loadTextureFile(resolved, srgb, img, mem, pool);
                if (view != VK_NULL_HANDLE) {
                    m_materials[mi].images[slot]   = img;
                    m_materials[mi].memories[slot] = mem;
                    m_materials[mi].views[slot]    = view;
                    m_materials[mi].hasMap[slot]   = true;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Process mesh nodes (depth-first traversal)
    // -----------------------------------------------------------------------
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
                    // Fallback tangent frame: T=(1,0,0), B=(0,1,0)
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
            submesh.name        = mesh->mName.C_Str();
            submesh.materialIdx = static_cast<int>(mesh->mMaterialIndex);
            submesh.indexCount  = static_cast<uint32_t>(indices.size());

            VkDeviceSize vsize = verts.size()   * sizeof(PBRVertex);
            VkDeviceSize isize = indices.size()  * sizeof(uint32_t);

            submesh.vertexBuffer = std::make_unique<Buffer>(m_ctx, vsize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            submesh.vertexBuffer->upload(pool, verts.data(), vsize);

            submesh.indexBuffer = std::make_unique<Buffer>(m_ctx, isize,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            submesh.indexBuffer->upload(pool, indices.data(), isize);

            std::cout << "[PBR] Submesh '" << submesh.name << "': "
                      << verts.size() << " verts, "
                      << indices.size() / 3 << " tris, mat=" << submesh.materialIdx << "\n";

            m_submeshes.push_back(std::move(submesh));
        }
        for (unsigned ci = 0; ci < node->mNumChildren; ++ci)
            processNode(node->mChildren[ci]);
    };

    processNode(scene->mRootNode);

    std::cout << "[PBR] Loaded '" << fs::path(fbxPath).filename().string() << "': "
              << m_submeshes.size() << " submeshes, "
              << m_materials.size() << " materials\n";
    return !m_submeshes.empty();
}

// ---------------------------------------------------------------------------
// Descriptor resources (call after load())
// ---------------------------------------------------------------------------
void PBRModel::createDescriptorResources(VkDescriptorSetLayout pbrDSLayout, int framesInFlight) {
    m_framesInFlight = framesInFlight;
    uint32_t nSets   = static_cast<uint32_t>(m_submeshes.size() * framesInFlight);

    // --- Per-frame UBO buffers ---
    for (int f = 0; f < framesInFlight; ++f) {
        m_uboBuffers.push_back(std::make_unique<Buffer>(m_ctx,
            sizeof(PBRUBOData),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    }

    // --- Descriptor pool ---
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = nSets;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = nSets * kPBRTexCount;

    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    pci.pPoolSizes    = poolSizes.data();
    pci.maxSets       = nSets;
    VK_CHECK(vkCreateDescriptorPool(m_ctx.device(), &pci, nullptr, &m_descPool));

    // --- Allocate sets ---
    std::vector<VkDescriptorSetLayout> layouts(nSets, pbrDSLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = m_descPool;
    ai.descriptorSetCount = nSets;
    ai.pSetLayouts        = layouts.data();
    m_descSets.resize(nSets);
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.device(), &ai, m_descSets.data()));

    // --- Write descriptors ---
    for (int s = 0; s < static_cast<int>(m_submeshes.size()); ++s) {
        int matIdx = m_submeshes[s].materialIdx;

        for (int f = 0; f < framesInFlight; ++f) {
            VkDescriptorSet ds = m_descSets[s * framesInFlight + f];

            // Binding 0: UBO for this frame
            VkDescriptorBufferInfo bi{};
            bi.buffer = m_uboBuffers[f]->handle();
            bi.offset = 0;
            bi.range  = sizeof(PBRUBOData);

            // Bindings 1-5: texture maps (use fallbacks if absent)
            VkDescriptorImageInfo imgInfos[kPBRTexCount]{};
            for (int t = 0; t < kPBRTexCount; ++t) {
                imgInfos[t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imgInfos[t].sampler     = m_sampler;

                VkImageView view = VK_NULL_HANDLE;
                if (matIdx >= 0 && matIdx < static_cast<int>(m_materials.size())
                    && m_materials[matIdx].hasMap[t]) {
                    view = m_materials[matIdx].views[t];
                }
                if (view == VK_NULL_HANDLE) {
                    view = (t == kPBRNormal) ? m_fbNormView : m_fbWhiteView;
                }
                imgInfos[t].imageView = view;
            }

            std::array<VkWriteDescriptorSet, 1 + kPBRTexCount> writes{};
            writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet          = ds;
            writes[0].dstBinding      = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo     = &bi;

            for (int t = 0; t < kPBRTexCount; ++t) {
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

// ---------------------------------------------------------------------------
// Descriptor set accessor
// ---------------------------------------------------------------------------
VkDescriptorSet PBRModel::descriptorSet(int submeshIdx, int frameIdx) const {
    return m_descSets[submeshIdx * m_framesInFlight + frameIdx];
}

} // namespace tgt
