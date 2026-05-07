#include "RenderPass.h"
#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include <stdexcept>
#include <array>

namespace tgt {

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

RenderPass::RenderPass(VulkanContext& ctx, Swapchain& swapchain)
    : m_ctx(ctx), m_swapchain(swapchain)
{
    m_depthFormat = ctx.findSupportedFormat(
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

    // Color attachment
    VkAttachmentDescription color{};
    color.format         = swapchain.imageFormat();
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = swapchain.colorFinalLayout();

    // Depth attachment
    VkAttachmentDescription depth{};
    depth.format         = m_depthFormat;
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = { color, depth };
    VkRenderPassCreateInfo rci{};
    rci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rci.attachmentCount = static_cast<uint32_t>(attachments.size());
    rci.pAttachments    = attachments.data();
    rci.subpassCount    = 1;
    rci.pSubpasses      = &subpass;
    rci.dependencyCount = 1;
    rci.pDependencies   = &dep;

    VK_CHECK(vkCreateRenderPass(ctx.device(), &rci, nullptr, &m_renderPass));

    createDepthResources();
    createFramebuffers();
}

RenderPass::~RenderPass() {
    cleanupFramebuffers();
    vkDestroyImageView(m_ctx.device(), m_depthView,   nullptr);
    vkDestroyImage(m_ctx.device(),     m_depthImage,  nullptr);
    vkFreeMemory(m_ctx.device(),       m_depthMemory, nullptr);
    vkDestroyRenderPass(m_ctx.device(), m_renderPass, nullptr);
}

void RenderPass::createDepthResources() {
    auto ext = m_swapchain.extent();

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent        = { ext.width, ext.height, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.format        = m_depthFormat;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(m_ctx.device(), &ici, nullptr, &m_depthImage));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_ctx.device(), m_depthImage, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = m_ctx.findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_ctx.device(), &ai, nullptr, &m_depthMemory));
    vkBindImageMemory(m_ctx.device(), m_depthImage, m_depthMemory, 0);

    VkImageViewCreateInfo ivi{};
    ivi.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivi.image                           = m_depthImage;
    ivi.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    ivi.format                          = m_depthFormat;
    ivi.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    ivi.subresourceRange.baseMipLevel   = 0;
    ivi.subresourceRange.levelCount     = 1;
    ivi.subresourceRange.baseArrayLayer = 0;
    ivi.subresourceRange.layerCount     = 1;
    VK_CHECK(vkCreateImageView(m_ctx.device(), &ivi, nullptr, &m_depthView));
}

void RenderPass::createFramebuffers() {
    auto ext   = m_swapchain.extent();
    auto count = m_swapchain.imageCount();
    m_framebuffers.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        std::array<VkImageView, 2> attachments = { m_swapchain.imageView(i), m_depthView };
        VkFramebufferCreateInfo fci{};
        fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass      = m_renderPass;
        fci.attachmentCount = static_cast<uint32_t>(attachments.size());
        fci.pAttachments    = attachments.data();
        fci.width           = ext.width;
        fci.height          = ext.height;
        fci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(m_ctx.device(), &fci, nullptr, &m_framebuffers[i]));
    }
}

void RenderPass::cleanupFramebuffers() {
    for (auto fb : m_framebuffers) vkDestroyFramebuffer(m_ctx.device(), fb, nullptr);
    m_framebuffers.clear();
}

void RenderPass::recreateFramebuffers() {
    vkDestroyImageView(m_ctx.device(), m_depthView,   nullptr);
    vkDestroyImage(m_ctx.device(),     m_depthImage,  nullptr);
    vkFreeMemory(m_ctx.device(),       m_depthMemory, nullptr);
    cleanupFramebuffers();
    createDepthResources();
    createFramebuffers();
}

} // namespace tgt
