#include "VulkanContext.h"
#include <stdexcept>
#include <set>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace tgt {

const std::vector<const char*> VulkanContext::kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> VulkanContext::kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Optional device extensions
static constexpr const char* kMemBudgetExt    = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
static constexpr const char* kPerfQueryExt    = VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME;
// Ray tracing (all four required together)
static constexpr const char* kRayQueryExt     = VK_KHR_RAY_QUERY_EXTENSION_NAME;
static constexpr const char* kAccelStructExt  = VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME;
static constexpr const char* kDeferredHostExt = VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
static constexpr const char* kBufDevAddrExt   = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
// Mesh shading (task + mesh stages replacing vertex+cull)
static constexpr const char* kMeshShaderExt   = VK_EXT_MESH_SHADER_EXTENSION_NAME;

#define VK_CHECK(x) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) throw std::runtime_error(std::string(#x " failed: ") + std::to_string(_r)); \
} while(0)

VulkanContext::VulkanContext(bool enableValidation, bool headless)
    : m_validation(enableValidation), m_headless(headless)
{
    createInstance();
    if (m_validation) setupDebugMessenger();
}

VulkanContext::~VulkanContext() {
    if (m_pipelineCache) vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
    if (m_device) vkDestroyDevice(m_device, nullptr);
    if (m_validation && m_debugMessenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, m_debugMessenger, nullptr);
    }
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}

void VulkanContext::initSurface(VkSurfaceKHR surface) {
    pickPhysicalDevice();
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

    std::vector<const char*> extensions;
    if (!m_headless) {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
        extensions.push_back("VK_KHR_win32_surface");
#elif defined(__linux__)
        extensions.push_back("VK_KHR_xcb_surface");
#endif
    }
    // Always request debug utils so Nsight/RenderDoc labels work even in release builds
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
    ci.pUserData       = this;

    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) throw std::runtime_error("vkCreateDebugUtilsMessengerEXT not found");
    VK_CHECK(fn(m_instance, &ci, nullptr, &m_debugMessenger));
}

void VulkanContext::loadDebugLabelFunctions() {
    m_fnBeginLabel  = (PFN_vkCmdBeginDebugUtilsLabelEXT)
        vkGetDeviceProcAddr(m_device, "vkCmdBeginDebugUtilsLabelEXT");
    m_fnEndLabel    = (PFN_vkCmdEndDebugUtilsLabelEXT)
        vkGetDeviceProcAddr(m_device, "vkCmdEndDebugUtilsLabelEXT");
    m_fnInsertLabel = (PFN_vkCmdInsertDebugUtilsLabelEXT)
        vkGetDeviceProcAddr(m_device, "vkCmdInsertDebugUtilsLabelEXT");
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
    // Offscreen headless path needs no device extensions beyond what's always present
    if (m_headless) return true;

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
    // In headless mode (no surface), the graphics queue handles everything
    if (surface == VK_NULL_HANDLE && indices.graphics.has_value())
        indices.present = indices.graphics;

    // Second pass: find dedicated async compute family (COMPUTE without GRAPHICS)
    for (uint32_t i = 0; i < count; ++i) {
        bool hasCompute  = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)  != 0;
        bool hasGraphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (hasCompute && !hasGraphics) { indices.compute = i; break; }
    }
    return indices;
}

void VulkanContext::createLogicalDevice() {
    std::set<uint32_t> uniqueQueues = {
        m_queueFamilies.graphics.value(),
        m_queueFamilies.present.value()
    };
    if (m_queueFamilies.compute.has_value())
        uniqueQueues.insert(m_queueFamilies.compute.value());

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
    features.samplerAnisotropy       = VK_TRUE;
    features.pipelineStatisticsQuery = VK_TRUE;

    // Build device extension list: required + optional extensions
    std::vector<const char*> devExts = m_headless
        ? std::vector<const char*>{}
        : kDeviceExtensions;

    // VkPhysicalDevicePerformanceQueryFeaturesKHR for VK_KHR_performance_query
    VkPhysicalDevicePerformanceQueryFeaturesKHR perfFeatures{};
    perfFeatures.sType                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR;
    perfFeatures.performanceCounterQueryPools = VK_FALSE; // set true if supported

    {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &count, available.data());
        bool hasRayQuery = false, hasAccelStruct = false, hasDeferredHost = false, hasBufDevAddr = false;
        bool hasMeshShader = false;
        for (auto& ext : available) {
            std::string name = ext.extensionName;
            if (name == kMemBudgetExt)    { devExts.push_back(kMemBudgetExt);    m_memBudgetSupported = true; }
            if (name == kPerfQueryExt)    { devExts.push_back(kPerfQueryExt);    perfFeatures.performanceCounterQueryPools = VK_TRUE; m_perfQuerySupported = true; }
            if (name == kRayQueryExt)     hasRayQuery    = true;
            if (name == kAccelStructExt)  hasAccelStruct = true;
            if (name == kDeferredHostExt) hasDeferredHost = true;
            if (name == kBufDevAddrExt)   hasBufDevAddr  = true;
            if (name == kMeshShaderExt)   hasMeshShader  = true;
        }
        if (hasRayQuery && hasAccelStruct && hasDeferredHost && hasBufDevAddr) {
            devExts.push_back(kRayQueryExt);
            devExts.push_back(kAccelStructExt);
            devExts.push_back(kDeferredHostExt);
            devExts.push_back(kBufDevAddrExt);
            m_rayQuerySupported = true;
        }
        if (hasMeshShader) {
            devExts.push_back(kMeshShaderExt);
            m_meshShaderSupported = true;
        }
    }

    // Vulkan 1.2 features: timeline semaphores (core) + descriptor indexing (for bindless)
    VkPhysicalDeviceVulkan12Features vk12Features{};
    vk12Features.sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12Features.timelineSemaphore                         = VK_TRUE;
    vk12Features.descriptorIndexing                        = VK_TRUE;
    vk12Features.runtimeDescriptorArray                    = VK_TRUE;
    vk12Features.descriptorBindingPartiallyBound           = VK_TRUE;
    vk12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    vk12Features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    vk12Features.bufferDeviceAddress                       = VK_TRUE;  // required for AS builds

    // Optional RT feature structs — only populated when all four extensions are present
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{};
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeat{};
    if (m_rayQuerySupported) {
        asFeat.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        asFeat.accelerationStructure = VK_TRUE;
        rqFeat.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        rqFeat.rayQuery = VK_TRUE;
        asFeat.pNext    = &rqFeat;
    }

    // Mesh shader feature struct (VK_EXT_mesh_shader)
    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeat{};
    if (m_meshShaderSupported) {
        meshFeat.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        meshFeat.meshShader = VK_TRUE;
        meshFeat.taskShader = VK_TRUE;
    }

    // Chain: vk12Features → perfFeatures (opt) → asFeat (opt) → meshFeat (opt) → end
    void* chainTail = m_meshShaderSupported ? static_cast<void*>(&meshFeat) : nullptr;
    if (m_rayQuerySupported) {
        asFeat.pNext = chainTail;   // RT tail links into mesh (or end)
        chainTail    = &asFeat;
    }
    if (m_perfQuerySupported) {
        vk12Features.pNext = &perfFeatures;
        perfFeatures.pNext = chainTail;
    } else {
        vk12Features.pNext = chainTail;
    }

    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext                   = &vk12Features;
    ci.queueCreateInfoCount    = static_cast<uint32_t>(queueCIs.size());
    ci.pQueueCreateInfos       = queueCIs.data();
    ci.enabledExtensionCount   = static_cast<uint32_t>(devExts.size());
    ci.ppEnabledExtensionNames = devExts.empty() ? nullptr : devExts.data();
    ci.pEnabledFeatures        = &features;
    if (m_validation) {
        ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
    }

    VK_CHECK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device));
    vkGetDeviceQueue(m_device, m_queueFamilies.graphics.value(), 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_queueFamilies.present.value(), 0, &m_presentQueue);
    if (m_queueFamilies.compute.has_value()) {
        vkGetDeviceQueue(m_device, m_queueFamilies.compute.value(), 0, &m_computeQueue);
        std::cout << "[TegraTrace] Async compute queue: family " << m_queueFamilies.compute.value() << "\n";
    }

    loadDebugLabelFunctions();

    // Create an initially-empty pipeline cache; caller can merge saved data via loadPipelineCache()
    {
        VkPipelineCacheCreateInfo pcci{};
        pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        VK_CHECK(vkCreatePipelineCache(m_device, &pcci, nullptr, &m_pipelineCache));
    }

    if (m_perfQuerySupported) {
        m_fnEnumPerfCounters =
            (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)
            vkGetInstanceProcAddr(m_instance,
                "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
        enumeratePerfCounters();
        std::cout << "[TegraTrace] VK_KHR_performance_query: " << m_perfCounters.size() << " counters\n";
    }
    if (m_memBudgetSupported)
        std::cout << "[TegraTrace] VK_EXT_memory_budget enabled\n";
    if (m_fnBeginLabel)
        std::cout << "[TegraTrace] VkCmdDebugUtilsLabel functions loaded (Nsight/RenderDoc ready)\n";
    if (m_rayQuerySupported) {
        loadRTFunctions();
        std::cout << "[TegraTrace] VK_KHR_ray_query + acceleration structures enabled\n";
    }
    if (m_meshShaderSupported) {
        fnCmdDrawMeshTasks = (PFN_vkCmdDrawMeshTasksEXT)
            vkGetDeviceProcAddr(m_device, "vkCmdDrawMeshTasksEXT");
        std::cout << "[TegraTrace] VK_EXT_mesh_shader enabled (task + mesh stages)\n";
    }
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

static std::string deriveSuggestion(const std::string& msg) {
    if (msg.find("image layout") != std::string::npos || msg.find("imageLayout") != std::string::npos)
        return "Check image layout transitions and pipeline barriers";
    if (msg.find("descriptor set") != std::string::npos || msg.find("descriptor binding") != std::string::npos)
        return "Verify descriptor set is bound and fully updated before draw";
    if (msg.find("synchronization") != std::string::npos || msg.find("hazard") != std::string::npos)
        return "Insert pipeline barrier with correct srcStage/dstStage and access masks";
    if (msg.find("srcAccessMask") != std::string::npos || msg.find("dstAccessMask") != std::string::npos)
        return "Ensure srcAccessMask/dstAccessMask cover all resource access types";
    if (msg.find("render pass") != std::string::npos || msg.find("renderPass") != std::string::npos)
        return "Check render pass begin/end pairing and subpass dependencies";
    if (msg.find("vertex buffer") != std::string::npos || msg.find("vertexBuffer") != std::string::npos)
        return "Bind vertex buffer before vkCmdDraw/vkCmdDrawIndexed";
    if (msg.find("index buffer") != std::string::npos)
        return "Bind index buffer before vkCmdDrawIndexed";
    if (msg.find("pipeline") != std::string::npos && msg.find("bound") != std::string::npos)
        return "Call vkCmdBindPipeline before issuing draw commands";
    if (msg.find("memory") != std::string::npos && msg.find("access") != std::string::npos)
        return "Add memory barrier to flush writes before next read";
    if (msg.find("push constant") != std::string::npos)
        return "Verify push constant range in pipeline layout covers all shader accesses";
    return "";
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* pData,
    void* pUser)
{
    ValidationSeverity sev = ValidationSeverity::Info;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        sev = ValidationSeverity::Error;
    else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        sev = ValidationSeverity::Warning;

    if (sev != ValidationSeverity::Info)
        std::cerr << "[Vulkan] " << pData->pMessage << "\n";

    if (pUser) {
        auto* ctx = static_cast<VulkanContext*>(pUser);
        ValidationMessage msg{};
        msg.severity   = sev;
        msg.text       = pData->pMessage;
        msg.suggestion = deriveSuggestion(pData->pMessage);
        msg.frame      = ctx->m_currentFrame;
        std::lock_guard<std::mutex> lock(ctx->m_logMutex);
        ctx->m_validationLog.push_back(std::move(msg));
    }
    return VK_FALSE;
}

// ---------------------------------------------------------------------------
// VK_EXT_memory_budget: per-heap VRAM budget and usage
// ---------------------------------------------------------------------------
MemoryBudget VulkanContext::queryMemoryBudget() const {
    MemoryBudget result{};
    if (!m_memBudgetSupported) return result;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps{};
    budgetProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

    VkPhysicalDeviceMemoryProperties2 memProps2{};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memProps2.pNext = &budgetProps;
    vkGetPhysicalDeviceMemoryProperties2(m_physicalDevice, &memProps2);

    result.supported  = true;
    result.heapCount  = memProps2.memoryProperties.memoryHeapCount;
    for (uint32_t i = 0; i < result.heapCount; ++i) {
        result.budget[i] = budgetProps.heapBudget[i];
        result.usage[i]  = budgetProps.heapUsage[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// Debug label wrappers — null-safe; no-ops when extension is not present
// ---------------------------------------------------------------------------
void VulkanContext::beginDebugLabel(VkCommandBuffer cmd, const char* name,
                                    float r, float g, float b) const {
    if (!m_fnBeginLabel) return;
    VkDebugUtilsLabelEXT label{};
    label.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0]   = r; label.color[1] = g;
    label.color[2]   = b; label.color[3] = 1.0f;
    m_fnBeginLabel(cmd, &label);
}

void VulkanContext::endDebugLabel(VkCommandBuffer cmd) const {
    if (m_fnEndLabel) m_fnEndLabel(cmd);
}

void VulkanContext::insertDebugLabel(VkCommandBuffer cmd, const char* name,
                                     float r, float g, float b) const {
    if (!m_fnInsertLabel) return;
    VkDebugUtilsLabelEXT label{};
    label.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0]   = r; label.color[1] = g;
    label.color[2]   = b; label.color[3] = 1.0f;
    m_fnInsertLabel(cmd, &label);
}

// ---------------------------------------------------------------------------
// Pipeline cache: persist compiled SPIR-V → driver binary across runs
// ---------------------------------------------------------------------------
void VulkanContext::loadPipelineCache(const std::string& path) {
    std::vector<uint8_t> data;
    if (std::ifstream f(path, std::ios::binary | std::ios::ate); f) {
        data.resize(static_cast<size_t>(f.tellg()));
        f.seekg(0);
        f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        std::cout << "[TegraTrace] Pipeline cache loaded (" << data.size() << " bytes)\n";
    }
    // Recreate with initial data (driver validates header; silently creates empty if invalid)
    if (m_pipelineCache) vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
    VkPipelineCacheCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = data.size();
    ci.pInitialData    = data.empty() ? nullptr : data.data();
    VK_CHECK(vkCreatePipelineCache(m_device, &ci, nullptr, &m_pipelineCache));
}

void VulkanContext::savePipelineCache(const std::string& path) const {
    if (!m_pipelineCache) return;
    size_t size = 0;
    vkGetPipelineCacheData(m_device, m_pipelineCache, &size, nullptr);
    if (size == 0) return;
    std::vector<uint8_t> data(size);
    vkGetPipelineCacheData(m_device, m_pipelineCache, &size, data.data());
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(size));
    std::cout << "[TegraTrace] Pipeline cache saved (" << size << " bytes)\n";
}

// ---------------------------------------------------------------------------
// VK_KHR_performance_query: enumerate available GPU hardware counters
// ---------------------------------------------------------------------------
void VulkanContext::enumeratePerfCounters() {
    if (!m_fnEnumPerfCounters) return;
    uint32_t queueFamilyIndex = m_queueFamilies.graphics.value();
    uint32_t counterCount = 0;
    m_fnEnumPerfCounters(m_physicalDevice, queueFamilyIndex, &counterCount, nullptr, nullptr);
    if (counterCount == 0) return;

    std::vector<VkPerformanceCounterKHR>            counters(counterCount);
    std::vector<VkPerformanceCounterDescriptionKHR> descs(counterCount);
    for (auto& c : counters)
        c.sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR;
    for (auto& d : descs)
        d.sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR;

    m_fnEnumPerfCounters(m_physicalDevice, queueFamilyIndex,
                         &counterCount, counters.data(), descs.data());

    m_perfCounters.clear();
    m_perfCounters.reserve(counterCount);
    for (uint32_t i = 0; i < counterCount; ++i) {
        PerfCounter pc{};
        pc.index       = i;
        pc.name        = descs[i].name;
        pc.category    = descs[i].category;
        pc.description = descs[i].description;
        m_perfCounters.push_back(std::move(pc));
    }
}

// ---------------------------------------------------------------------------
// VK_KHR_ray_query: load extension function pointers + BDA helper
// ---------------------------------------------------------------------------
void VulkanContext::loadRTFunctions() {
    fnCreateAccelStruct    = (PFN_vkCreateAccelerationStructureKHR)
        vkGetDeviceProcAddr(m_device, "vkCreateAccelerationStructureKHR");
    fnDestroyAccelStruct   = (PFN_vkDestroyAccelerationStructureKHR)
        vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR");
    fnGetAccelBuildSizes   = (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureBuildSizesKHR");
    fnCmdBuildAccelStructs = (PFN_vkCmdBuildAccelerationStructuresKHR)
        vkGetDeviceProcAddr(m_device, "vkCmdBuildAccelerationStructuresKHR");
    fnGetAccelDevAddr      = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
        vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureDeviceAddressKHR");
}

VkDeviceAddress VulkanContext::getBufferDeviceAddress(VkBuffer buf) const {
    VkBufferDeviceAddressInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buf;
    return vkGetBufferDeviceAddress(m_device, &info);
}

} // namespace tgt
