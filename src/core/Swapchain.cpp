#include "Swapchain.h"
#include "VulkanContext.h"
#include "Window.h"
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace tgt {

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

Swapchain::Swapchain(VulkanContext& ctx, Window& window, VkSurfaceKHR surface)
    : m_ctx(ctx), m_window(window), m_surface(surface)
{
    create();
}

Swapchain::~Swapchain() {
    cleanup();
}

void Swapchain::create() {
    auto support = m_ctx.querySwapchainSupport(m_surface);
    auto format  = chooseSurfaceFormat(support.formats);
    auto mode    = choosePresentMode(support.presentModes);
    auto extent  = chooseExtent(support.capabilities);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0)
        imageCount = std::min(imageCount, support.capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = m_surface;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = format.format;
    ci.imageColorSpace  = format.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    auto qf = m_ctx.queueFamilies();
    uint32_t qfIndices[] = { qf.graphics.value(), qf.present.value() };
    if (qfIndices[0] != qfIndices[1]) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = qfIndices;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform   = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = mode;
    ci.clipped        = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(m_ctx.device(), &ci, nullptr, &m_swapchain));

    m_imageFormat = format.format;
    m_extent      = extent;

    uint32_t count;
    vkGetSwapchainImagesKHR(m_ctx.device(), m_swapchain, &count, nullptr);
    m_images.resize(count);
    vkGetSwapchainImagesKHR(m_ctx.device(), m_swapchain, &count, m_images.data());

    m_imageViews.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo ivi{};
        ivi.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivi.image                           = m_images[i];
        ivi.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        ivi.format                          = m_imageFormat;
        ivi.components                      = { VK_COMPONENT_SWIZZLE_IDENTITY };
        ivi.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ivi.subresourceRange.baseMipLevel   = 0;
        ivi.subresourceRange.levelCount     = 1;
        ivi.subresourceRange.baseArrayLayer = 0;
        ivi.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(m_ctx.device(), &ivi, nullptr, &m_imageViews[i]));
    }

    // Sync objects
    m_imageAvailableSemaphores.resize(kMaxFramesInFlight);
    m_renderFinishedSemaphores.resize(kMaxFramesInFlight);
    m_inFlightFences.resize(kMaxFramesInFlight);
    m_imagesInFlight.resize(count, VK_NULL_HANDLE);

    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        VK_CHECK(vkCreateSemaphore(m_ctx.device(), &si, nullptr, &m_imageAvailableSemaphores[i]));
        VK_CHECK(vkCreateSemaphore(m_ctx.device(), &si, nullptr, &m_renderFinishedSemaphores[i]));
        VK_CHECK(vkCreateFence(m_ctx.device(), &fi, nullptr, &m_inFlightFences[i]));
    }
}

void Swapchain::cleanup() {
    for (auto iv : m_imageViews)  vkDestroyImageView(m_ctx.device(), iv, nullptr);
    if (m_swapchain) vkDestroySwapchainKHR(m_ctx.device(), m_swapchain, nullptr);
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        vkDestroySemaphore(m_ctx.device(), m_imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(m_ctx.device(), m_renderFinishedSemaphores[i], nullptr);
        vkDestroyFence(m_ctx.device(), m_inFlightFences[i], nullptr);
    }
    m_imageViews.clear(); m_images.clear();
    m_imageAvailableSemaphores.clear();
    m_renderFinishedSemaphores.clear();
    m_inFlightFences.clear();
    m_imagesInFlight.clear();
}

void Swapchain::recreate() {
    vkDeviceWaitIdle(m_ctx.device());
    cleanup();
    create();
}

VkResult Swapchain::acquireNextImage(uint32_t* imageIndex) {
    vkWaitForFences(m_ctx.device(), 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);
    return vkAcquireNextImageKHR(m_ctx.device(), m_swapchain, UINT64_MAX,
                                  m_imageAvailableSemaphores[m_currentFrame],
                                  VK_NULL_HANDLE, imageIndex);
}

VkResult Swapchain::submitAndPresent(uint32_t imageIndex, VkCommandBuffer cmd) {
    if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(m_ctx.device(), 1, &m_imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    m_imagesInFlight[imageIndex] = m_inFlightFences[m_currentFrame];

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &m_imageAvailableSemaphores[m_currentFrame];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &m_renderFinishedSemaphores[m_currentFrame];

    vkResetFences(m_ctx.device(), 1, &m_inFlightFences[m_currentFrame]);
    VK_CHECK(vkQueueSubmit(m_ctx.graphicsQueue(), 1, &si, m_inFlightFences[m_currentFrame]));

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &m_renderFinishedSemaphores[m_currentFrame];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &m_swapchain;
    pi.pImageIndices      = &imageIndex;

    m_currentFrame = (m_currentFrame + 1) % kMaxFramesInFlight;
    return vkQueuePresentKHR(m_ctx.presentQueue(), &pi);
}

VkSurfaceFormatKHR Swapchain::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
    return formats[0];
}

VkPresentModeKHR Swapchain::choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Swapchain::chooseExtent(const VkSurfaceCapabilitiesKHR& caps) {
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return caps.currentExtent;
    VkExtent2D ext = { static_cast<uint32_t>(m_window.width()),
                       static_cast<uint32_t>(m_window.height()) };
    ext.width  = std::clamp(ext.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    ext.height = std::clamp(ext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return ext;
}

} // namespace tgt
