#include "Pipeline.h"
#include "Shader.h"
#include "Buffer.h"
#include "RenderPass.h"
#include "core/VulkanContext.h"
#include <array>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <fstream>

namespace tgt {

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

Pipeline::Pipeline(VulkanContext& ctx, RenderPass& renderPass,
                   VkExtent2D extent, const PipelineConfig& cfg)
    : m_ctx(ctx),
      m_name(std::filesystem::path(cfg.vertSpvPath).stem().string()),
      m_vertSpvPath(cfg.vertSpvPath),
      m_fragSpvPath(cfg.fragSpvPath),
      m_renderPassHandle(renderPass.handle()),
      m_extent(extent),
      m_cfg(cfg)
{
    VkPushConstantRange pcRange{};

    if (cfg.useMeshShader) {
        // Mesh shader pipeline: task+mesh+fragment with extended bindless layout (9 bindings)
        // Bindings 0-4: same as bindless PBR; bindings 5-8: meshlet SSBOs
        static constexpr uint32_t kMaxBindless = 256;
        VkDescriptorSetLayoutBinding msBindings[9]{};
        msBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        msBindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindless,
            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        msBindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        msBindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        msBindings[4] = { 4, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        msBindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr };
        msBindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_MESH_BIT_EXT, nullptr };
        msBindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_MESH_BIT_EXT, nullptr };
        msBindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_MESH_BIT_EXT, nullptr };
        VkDescriptorBindingFlags msFlags[9] = {
            0,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{};
        bfci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bfci.bindingCount  = 9;
        bfci.pBindingFlags = msFlags;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        dlci.pNext        = &bfci;
        dlci.bindingCount = 9;
        dlci.pBindings    = msBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device(), &dlci, nullptr, &m_dsLayout));
        // PC: mat4 model (64) + 7 uints (PBR indices + rtEnabled) + 2 uints (meshlet offset/count) = 100
        pcRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                             VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset = 0;
        pcRange.size   = 100;
    } else if (cfg.useBindless) {
        // Bindless layout:
        //   0 = UBO,             1 = sampler2D[256] (partially bound, update-after-bind)
        //   2 = samplerCube prefilt,  3 = samplerCube irrad
        //   4 = accelerationStructureEXT (TLAS, partially bound — omitted if RT not built)
        static constexpr uint32_t kMaxBindless = 256;
        VkDescriptorSetLayoutBinding blBindings[5]{};
        blBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,               1,           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        blBindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindless,       VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        blBindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,       1,            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        blBindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,       1,            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        blBindings[4] = { 4, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,   1,            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorBindingFlags blFlags[5] = {
            0,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{};
        bfci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bfci.bindingCount  = 5;
        bfci.pBindingFlags = blFlags;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        dlci.pNext        = &bfci;
        dlci.bindingCount = 5;
        dlci.pBindings    = blBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device(), &dlci, nullptr, &m_dsLayout));
        // Push constants: mat4 model (64) + 6 tex indices + rtEnabled (28) = 92 bytes, vert|frag
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 92;
    } else {
        // Classic per-binding layout
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding         = 0;
        uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(uboBinding);
        for (uint32_t t = 0; t < cfg.textureBindingCount; ++t) {
            VkDescriptorSetLayoutBinding sb{};
            sb.binding         = t + 1;
            sb.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sb.descriptorCount = 1;
            sb.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings.push_back(sb);
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device(), &dlci, nullptr, &m_dsLayout));
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 64;
    }

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &m_dsLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(ctx.device(), &plci, nullptr, &m_layout));

    m_pipeline = buildGraphicsPipeline();
}

VkPipeline Pipeline::buildGraphicsPipeline() const {
    VkPipeline out = VK_NULL_HANDLE;

    if (m_cfg.useMeshShader) {
        Shader task(m_ctx.device(), m_cfg.taskSpvPath);
        Shader mesh(m_ctx.device(), m_cfg.meshSpvPath);
        Shader frag(m_ctx.device(), m_cfg.fragSpvPath);

        VkPipelineShaderStageCreateInfo stages[3]{};
        stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_TASK_BIT_EXT, task.module(), "main", nullptr };
        stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_MESH_BIT_EXT, mesh.module(), "main", nullptr };
        stages[2] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_FRAGMENT_BIT, frag.module(), "main", nullptr };

        VkViewport vp{ 0, 0, static_cast<float>(m_extent.width), static_cast<float>(m_extent.height), 0, 1 };
        VkRect2D   sc{ {0,0}, m_extent };
        VkPipelineViewportStateCreateInfo vps{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                               nullptr, 0, 1, &vp, 1, &sc };
        VkPipelineRasterizationStateCreateInfo rast{};
        rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rast.polygonMode = m_cfg.polyMode; rast.cullMode = m_cfg.cullMode;
        rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rast.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                 nullptr, 0, VK_SAMPLE_COUNT_1_BIT };
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = m_cfg.depthTest ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = m_cfg.depthWrite ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cbs{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                 nullptr, 0, VK_FALSE, {}, 1, &blend };
        std::array<VkDynamicState,2> dyn2 = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                              nullptr, 0, 2, dyn2.data() };
        VkGraphicsPipelineCreateInfo gci{};
        gci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount = 3; gci.pStages = stages;
        gci.pVertexInputState = nullptr; gci.pInputAssemblyState = nullptr;
        gci.pViewportState = &vps; gci.pRasterizationState = &rast;
        gci.pMultisampleState = &ms; gci.pDepthStencilState = &ds;
        gci.pColorBlendState = &cbs; gci.pDynamicState = &dyn;
        gci.layout = m_layout; gci.renderPass = m_renderPassHandle; gci.subpass = 0;
        VK_CHECK(vkCreateGraphicsPipelines(m_ctx.device(), m_ctx.pipelineCache(), 1, &gci, nullptr, &out));
        return out;
    }

    Shader vert(m_ctx.device(), m_cfg.vertSpvPath);
    Shader frag(m_ctx.device(), m_cfg.fragSpvPath);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_VERTEX_BIT, vert.module(), "main", nullptr };
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_FRAGMENT_BIT, frag.module(), "main", nullptr };

    VkVertexInputBindingDescription binding;
    std::vector<VkVertexInputAttributeDescription> attribs;
    if (m_cfg.usePBRVertex) {
        binding = PBRVertex::bindingDescription();
        attribs = PBRVertex::attributeDescriptions();
    } else {
        binding = Vertex::bindingDescription();
        attribs = Vertex::attributeDescriptions();
    }
    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vis.vertexBindingDescriptionCount = 1; vis.pVertexBindingDescriptions = &binding;
    vis.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size());
    vis.pVertexAttributeDescriptions = attribs.data();
    VkPipelineInputAssemblyStateCreateInfo ias{};
    ias.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{ 0, 0, static_cast<float>(m_extent.width), static_cast<float>(m_extent.height), 0, 1 };
    VkRect2D   sc{ {0,0}, m_extent };
    VkPipelineViewportStateCreateInfo vps{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                           nullptr, 0, 1, &vp, 1, &sc };
    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = m_cfg.polyMode; rast.cullMode = m_cfg.cullMode;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rast.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                             nullptr, 0, VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = m_cfg.depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = m_cfg.depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cbs{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                             nullptr, 0, VK_FALSE, {}, 1, &blend };
    std::array<VkDynamicState,2> dyn2 = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                          nullptr, 0, 2, dyn2.data() };
    VkGraphicsPipelineCreateInfo gci{};
    gci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gci.stageCount = 2; gci.pStages = stages;
    gci.pVertexInputState = &vis; gci.pInputAssemblyState = &ias;
    gci.pViewportState = &vps; gci.pRasterizationState = &rast;
    gci.pMultisampleState = &ms; gci.pDepthStencilState = &ds;
    gci.pColorBlendState = &cbs; gci.pDynamicState = &dyn;
    gci.layout = m_layout; gci.renderPass = m_renderPassHandle; gci.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(m_ctx.device(), m_ctx.pipelineCache(), 1, &gci, nullptr, &out));
    return out;
}

Pipeline::~Pipeline() {
    vkDestroyPipeline(m_ctx.device(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_ctx.device(), m_layout, nullptr);
    vkDestroyDescriptorSetLayout(m_ctx.device(), m_dsLayout, nullptr);
}

bool Pipeline::tryReload() {
    // Verify all SPIR-V files are readable before touching GPU state
    if (m_cfg.useMeshShader) {
        std::ifstream t(m_cfg.taskSpvPath, std::ios::binary);
        std::ifstream m(m_cfg.meshSpvPath, std::ios::binary);
        std::ifstream f(m_cfg.fragSpvPath, std::ios::binary);
        if (!t || !m || !f) return false;
    } else {
        std::ifstream v(m_cfg.vertSpvPath, std::ios::binary);
        std::ifstream f(m_cfg.fragSpvPath, std::ios::binary);
        if (!v || !f) return false;
    }
    try {
        vkDeviceWaitIdle(m_ctx.device());
        VkPipeline newPipeline = buildGraphicsPipeline();
        vkDestroyPipeline(m_ctx.device(), m_pipeline, nullptr);
        m_pipeline = newPipeline;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace tgt
