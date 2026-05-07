#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <cstdint>

namespace tgt {

class VulkanContext;
class Swapchain;

struct ImageDiffResult {
    std::string testName;
    bool        passed       = false;
    double      psnrDb       = 0.0;
    double      maxPixelDiff = 0.0;
    double      rmse         = 0.0;
    uint32_t    failedPixels = 0;
    uint32_t    totalPixels  = 0;
};

struct RegressionReport {
    uint32_t                     totalTests  = 0;
    uint32_t                     passedTests = 0;
    std::vector<ImageDiffResult> results;

    double passRate() const {
        return totalTests ? 100.0 * passedTests / totalTests : 0.0;
    }
};

class RegressionTester {
public:
    RegressionTester(VulkanContext& ctx, Swapchain& swapchain,
                     const std::string& referenceDir,
                     const std::string& outputDir,
                     double psnrThresholdDb = 40.0);
    ~RegressionTester();

    // Capture current swapchain image, save to disk, and compare against reference.
    // Call this after vkQueueWaitIdle so the image is settled.
    ImageDiffResult captureAndTest(const std::string& testName,
                                   uint32_t swapchainImageIndex,
                                   VkCommandPool cmdPool);

    // Save current swapchain image as the new reference for testName
    void saveReference(const std::string& testName,
                       uint32_t swapchainImageIndex,
                       VkCommandPool cmdPool);

    const RegressionReport& report() const { return m_report; }
    void exportReport(const std::string& path) const;

private:
    std::vector<uint8_t> readbackImage(uint32_t swapchainImageIndex,
                                       VkCommandPool cmdPool,
                                       uint32_t& outWidth,
                                       uint32_t& outHeight);

    bool savePNG(const std::string& path, const std::vector<uint8_t>& pixels,
                 uint32_t width, uint32_t height) const;

    ImageDiffResult compareImages(const std::string& testName,
                                  const std::vector<uint8_t>& captured,
                                  uint32_t w, uint32_t h,
                                  const std::string& refPath) const;

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    std::string    m_referenceDir;
    std::string    m_outputDir;
    double         m_psnrThreshold;
    RegressionReport m_report;

    // Staging buffer for readback
    VkBuffer       m_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_stagingMemory = VK_NULL_HANDLE;
    VkDeviceSize   m_stagingSize   = 0;
};

} // namespace tgt
