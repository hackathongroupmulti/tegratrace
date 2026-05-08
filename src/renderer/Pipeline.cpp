#include "Pipeline.h"
#include "Shader.h"
#include "Buffer.h"
#include "RenderPass.h"
#include "core/VulkanContext.h"
#include <array>
#include <vector>
#include <stdexcept>
#include <filesystem>

namespace tgt {

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

Pipeline::Pipeline(VulkanContext& ctx, RenderPass& renderPass,
                   VkExtent2D extent, const PipelineConfig& cfg)
    : m_ctx(ctx),
      m_name(std::filesystem::path(cfg.vertSpvPath).stem().string()),
      m_vertSpvPath(cfg.vertSpvPath),
      m_fragSpvPath(cfg.fragSpvPath)
{
    // Binding 0: UBO (vertex + fragment stages for PBR; vertex-only for legacy)
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding         = 0;
    uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 1..textureBindingCount: combined-image-samplers (fragment stage)
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

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset     = 0;
    pcRange.size       = 64; // mat4

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &m_dsLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(ctx.device(), &plci, nullptr, &m_layout));

    Shader vert(ctx.device(), cfg.vertSpvPath);
    Shader frag(ctx.device(), cfg.fragSpvPath);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert.module();
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag.module();
    stages[1].pName  = "main";

    VkVertexInputBindingDescription binding;
    std::vector<VkVertexInputAttributeDescription> attribs;
    if (cfg.usePBRVertex) {
        binding = PBRVertex::bindingDescription();
        attribs = PBRVertex::attributeDescriptions();
    } else {
        binding = Vertex::bindingDescription();
        attribs = Vertex::attributeDescriptions();
    }
    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vis.vertexBindingDescriptionCount   = 1;
    vis.pVertexBindingDescriptions      = &binding;
    vis.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size());
    vis.pVertexAttributeDescriptions    = attribs.data();

    VkPipelineInputAssemblyStateCreateInfo ias{};
    ias.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{ 0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1 };
    VkRect2D   scissor{ {0,0}, extent };
    VkPipelineViewportStateCreateInfo vps{};
    vps.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports    = &viewport;
    vps.scissorCount  = 1;
    vps.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = cfg.polyMode;
    rast.cullMode    = cfg.cullMode;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = cfg.depthTest  ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = cfg.depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cbs{};
    cbs.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbs.attachmentCount = 1;
    cbs.pAttachments    = &blend;

    // Dynamic viewport/scissor for resize support
    std::array<VkDynamicState, 2> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dyn.pDynamicStates    = dynStates.data();

    VkGraphicsPipelineCreateInfo gci{};
    gci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gci.stageCount          = 2;
    gci.pStages             = stages;
    gci.pVertexInputState   = &vis;
    gci.pInputAssemblyState = &ias;
    gci.pViewportState      = &vps;
    gci.pRasterizationState = &rast;
    gci.pMultisampleState   = &ms;
    gci.pDepthStencilState  = &ds;
    gci.pColorBlendState    = &cbs;
    gci.pDynamicState       = &dyn;
    gci.layout              = m_layout;
    gci.renderPass          = renderPass.handle();
    gci.subpass             = 0;

    VK_CHECK(vkCreateGraphicsPipelines(ctx.device(), VK_NULL_HANDLE, 1, &gci, nullptr, &m_pipeline));
}

Pipeline::~Pipeline() {
    vkDestroyPipeline(m_ctx.device(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_ctx.device(), m_layout, nullptr);
    vkDestroyDescriptorSetLayout(m_ctx.device(), m_dsLayout, nullptr);
}

} // namespace tgt
