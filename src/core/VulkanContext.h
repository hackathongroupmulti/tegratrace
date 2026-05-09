#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <string>
#include <mutex>

namespace tgt {

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool isComplete() const { return graphics.has_value() && present.has_value(); }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

enum class ValidationSeverity { Info, Warning, Error };

struct ValidationMessage {
    ValidationSeverity severity;
    std::string        text;
    std::string        suggestion;
    uint32_t           frame = 0;
};

// Per-heap VRAM budget and usage (from VK_EXT_memory_budget)
struct MemoryBudget {
    uint32_t     heapCount = 0;
    VkDeviceSize budget[VK_MAX_MEMORY_HEAPS] = {};  // available budget per heap
    VkDeviceSize usage[VK_MAX_MEMORY_HEAPS]  = {};  // current usage per heap
    bool         supported = false;
};

class VulkanContext {
public:
    explicit VulkanContext(bool enableValidation = true, bool headless = false);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    void initSurface(VkSurfaceKHR surface);

    VkInstance         instance()        const { return m_instance; }
    VkPhysicalDevice   physicalDevice()  const { return m_physicalDevice; }
    VkDevice           device()          const { return m_device; }
    VkQueue            graphicsQueue()   const { return m_graphicsQueue; }
    VkQueue            presentQueue()    const { return m_presentQueue; }
    QueueFamilyIndices queueFamilies()   const { return m_queueFamilies; }

    SwapchainSupportDetails querySwapchainSupport(VkSurfaceKHR surface) const;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const;
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                  VkImageTiling tiling, VkFormatFeatureFlags features) const;
    float timestampPeriod() const { return m_timestampPeriod; }

    VkCommandBuffer beginSingleTimeCommands(VkCommandPool pool) const;
    void            endSingleTimeCommands(VkCommandPool pool, VkCommandBuffer cmd) const;

    // Validation log
    void setCurrentFrame(uint32_t frame) { m_currentFrame = frame; }
    const std::vector<ValidationMessage>& validationLog() const { return m_validationLog; }

    // VK_EXT_memory_budget: per-heap VRAM budget and usage
    MemoryBudget queryMemoryBudget() const;

    // Nsight/RenderDoc debug labels (VK_EXT_debug_utils)
    void beginDebugLabel(VkCommandBuffer cmd, const char* name,
                         float r = 0.5f, float g = 0.5f, float b = 1.0f) const;
    void endDebugLabel(VkCommandBuffer cmd) const;
    void insertDebugLabel(VkCommandBuffer cmd, const char* name,
                          float r = 1.0f, float g = 1.0f, float b = 0.0f) const;

    // Pipeline cache (eliminates warmup compilation spikes across runs)
    VkPipelineCache pipelineCache() const { return m_pipelineCache; }
    void loadPipelineCache(const std::string& path);
    void savePipelineCache(const std::string& path) const;

    // VK_KHR_performance_query: hardware counter enumeration (optional, read-only)
    struct PerfCounter {
        std::string name;
        std::string category;
        std::string description;
        uint32_t    index = 0;
    };
    bool perfQuerySupported()               const { return m_perfQuerySupported; }
    const std::vector<PerfCounter>& perfCounters() const { return m_perfCounters; }

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void loadDebugLabelFunctions();
    void enumeratePerfCounters();

    bool checkValidationLayerSupport() const;
    bool isDeviceSuitable(VkPhysicalDevice dev, VkSurfaceKHR surface) const;
    int  scoreDevice(VkPhysicalDevice dev) const;
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface) const;
    bool checkDeviceExtensionSupport(VkPhysicalDevice dev) const;

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* pData,
        void* pUser);

    VkInstance               m_instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physicalDevice = VK_NULL_HANDLE;
    VkDevice                 m_device         = VK_NULL_HANDLE;
    VkQueue                  m_graphicsQueue  = VK_NULL_HANDLE;
    VkQueue                  m_presentQueue   = VK_NULL_HANDLE;
    QueueFamilyIndices       m_queueFamilies;
    float                    m_timestampPeriod = 1.0f;
    bool                     m_validation;
    bool                     m_headless;
    bool                     m_memBudgetSupported = false;

    // Nsight/RenderDoc: vkCmdBegin/EndDebugUtilsLabelEXT function pointers
    PFN_vkCmdBeginDebugUtilsLabelEXT  m_fnBeginLabel  = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT    m_fnEndLabel    = nullptr;
    PFN_vkCmdInsertDebugUtilsLabelEXT m_fnInsertLabel = nullptr;

    // Pipeline cache
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

    // VK_KHR_performance_query
    bool                    m_perfQuerySupported = false;
    std::vector<PerfCounter> m_perfCounters;
    PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR m_fnEnumPerfCounters = nullptr;

    uint32_t                       m_currentFrame = 0;
    mutable std::mutex             m_logMutex;
    std::vector<ValidationMessage> m_validationLog;

    static const std::vector<const char*> kValidationLayers;
    static const std::vector<const char*> kDeviceExtensions;
};

} // namespace tgt
