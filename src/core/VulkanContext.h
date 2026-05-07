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
    uint32_t           frame = 0;
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

    // Validation log — populated when validation layers are active
    void setCurrentFrame(uint32_t frame) { m_currentFrame = frame; }
    const std::vector<ValidationMessage>& validationLog() const { return m_validationLog; }

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();

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

    uint32_t                      m_currentFrame = 0;
    mutable std::mutex            m_logMutex;
    std::vector<ValidationMessage> m_validationLog;

    static const std::vector<const char*> kValidationLayers;
    static const std::vector<const char*> kDeviceExtensions;
};

} // namespace tgt
