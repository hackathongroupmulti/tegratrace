#include "VulkanContext.h"
#include <stdexcept>
#include <set>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace tgt {

const std::vector<const char*> VulkanContext::kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> VulkanContext::kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#define VK_CHECK(x) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) throw std::runtime_error(std::string(#x " failed: ") + std::to_string(_r)); \
} while(0)

VulkanContext::VulkanContext(bool enableValidation)
    : m_validation(enableValidation)
{
    createInstance();
    if (m_validation) setupDebugMessenger();
}

VulkanContext::~VulkanContext() {
    if (m_device) vkDestroyDevice(m_device, nullptr);
    if (m_validation && m_debugMessenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, m_debugMessenger, nullptr);
    }
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}

void VulkanContext::initSurface(VkSurfaceKHR surface) {
    pickPhysicalDevice();   // needs surface for present support check - store it temporarily
    // Re-query with surface
    m_queueFamilies = findQueueFamilies(m_physicalDevice, surface);
    createLogicalDevice();
}

void VulkanContext::createInstance() {
    if (m_validation && !checkValidationLayerSupport())
        throw std::runtime_error("Requested validation layers not available");

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "TegraTrace";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName        = "TegraTrace";
    appInfo.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_2;

    std::vector<const char*> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef _WIN32
        "VK_KHR_win32_surface",
#elif defined(__linux__)
        "VK_KHR_xcb_surface",
#endif
    };
    if (m_validation)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &appInfo;
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    if (m_validation) {
        ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
    }

    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));
}

void VulkanContext::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;

    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) throw std::runtime_error("vkCreateDebugUtilsMessengerEXT not found");
    VK_CHECK(fn(m_instance, &ci, nullptr, &m_debugMessenger));
}

void VulkanContext::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    int bestScore = -1;
    for (auto dev : devices) {
        int score = scoreDevice(dev);
        if (score > bestScore) { bestScore = score; m_physicalDevice = dev; }
    }
    if (m_physicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("No suitable GPU found");

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    m_timestampPeriod = props.limits.timestampPeriod;
    std::cout << "[TegraTrace] GPU: " << props.deviceName
              << " | timestamp period: " << m_timestampPeriod << " ns\n";
}

int VulkanContext::scoreDevice(VkPhysicalDevice dev) const {
    if (!checkDeviceExtensionSupport(dev)) return -1;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures   feats;
    vkGetPhysicalDeviceProperties(dev, &props);
    vkGetPhysicalDeviceFeatures(dev, &feats);
    if (!feats.samplerAnisotropy) return -1;
    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 10000;
    score += static_cast<int>(props.limits.maxImageDimension2D);
    return score;
}

bool VulkanContext::checkDeviceExtensionSupport(VkPhysicalDevice dev) const {
    uint32_t count;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, available.data());
    std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (auto& ext : available) required.erase(ext.extensionName);
    return required.empty();
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice dev, VkSurfaceKHR surface) const {
    auto indices = findQueueFamilies(dev, surface);
    if (!indices.isComplete()) return false;
    auto support = querySwapchainSupport(surface);
    return !support.formats.empty() && !support.presentModes.empty();
}

QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface) const {
    QueueFamilyIndices indices;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphics = i;
        VkBool32 presentSupport = VK_FALSE;
        if (surface != VK_NULL_HANDLE)
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
        if (presentSupport) indices.present = i;
        if (indices.isComplete()) break;
    }
    return indices;
}

void VulkanContext::createLogicalDevice() {
    std::set<uint32_t> uniqueQueues = {
        m_queueFamilies.graphics.value(),
        m_queueFamilies.present.value()
    };

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    for (uint32_t qf : uniqueQueues) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = qf;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &priority;
        queueCIs.push_back(qi);
    }

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy    = VK_TRUE;
    features.pipelineStatisticsQuery = VK_TRUE;

    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount    = static_cast<uint32_t>(queueCIs.size());
    ci.pQueueCreateInfos       = queueCIs.data();
    ci.enabledExtensionCount   = static_cast<uint32_t>(kDeviceExtensions.size());
    ci.ppEnabledExtensionNames = kDeviceExtensions.data();
    ci.pEnabledFeatures        = &features;
    if (m_validation) {
        ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
    }

    VK_CHECK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device));
    vkGetDeviceQueue(m_device, m_queueFamilies.graphics.value(), 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_queueFamilies.present.value(), 0, &m_presentQueue);
}

SwapchainSupportDetails VulkanContext::querySwapchainSupport(VkSurfaceKHR surface) const {
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, surface, &details.capabilities);
    uint32_t count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, surface, &count, nullptr);
    details.formats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, surface, &count, details.formats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, surface, &count, nullptr);
    details.presentModes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, surface, &count, details.presentModes.data());
    return details;
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("Failed to find suitable memory type");
}

VkFormat VulkanContext::findSupportedFormat(const std::vector<VkFormat>& candidates,
                                             VkImageTiling tiling,
                                             VkFormatFeatureFlags features) const {
    for (auto fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR  && (props.linearTilingFeatures  & features) == features) return fmt;
        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) return fmt;
    }
    throw std::runtime_error("Failed to find supported format");
}

VkCommandBuffer VulkanContext::beginSingleTimeCommands(VkCommandPool pool) const {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool        = pool;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandPool pool, VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(m_graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQueue);
    vkFreeCommandBuffers(m_device, pool, 1, &cmd);
}

bool VulkanContext::checkValidationLayerSupport() const {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (auto* name : kValidationLayers) {
        bool found = false;
        for (auto& l : layers) if (strcmp(l.layerName, name) == 0) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* pData,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << pData->pMessage << "\n";
    return VK_FALSE;
}

} // namespace tgt
