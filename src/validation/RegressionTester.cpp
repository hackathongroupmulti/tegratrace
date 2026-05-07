#include "RegressionTester.h"
#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize2.h>

namespace tgt {

using json = nlohmann::json;

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

RegressionTester::RegressionTester(VulkanContext& ctx, Swapchain& swapchain,
                                    const std::string& referenceDir,
                                    const std::string& outputDir,
                                    double psnrThreshold)
    : m_ctx(ctx), m_swapchain(swapchain),
      m_referenceDir(referenceDir), m_outputDir(outputDir),
      m_psnrThreshold(psnrThreshold)
{}

RegressionTester::~RegressionTester() {
    if (m_stagingBuffer) vkDestroyBuffer(m_ctx.device(), m_stagingBuffer, nullptr);
    if (m_stagingMemory) vkFreeMemory(m_ctx.device(), m_stagingMemory, nullptr);
}

std::vector<uint8_t> RegressionTester::readbackImage(uint32_t swapchainImageIndex,
                                                       VkCommandPool cmdPool,
                                                       uint32_t& outWidth,
                                                       uint32_t& outHeight) {
    auto ext   = m_swapchain.extent();
    outWidth   = ext.width;
    outHeight  = ext.height;
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(ext.width) * ext.height * 4;

    // Allocate or grow staging buffer
    if (imageSize > m_stagingSize) {
        if (m_stagingBuffer) {
            vkDestroyBuffer(m_ctx.device(), m_stagingBuffer, nullptr);
            vkFreeMemory(m_ctx.device(), m_stagingMemory, nullptr);
        }
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = imageSize;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(m_ctx.device(), &bci, nullptr, &m_stagingBuffer));

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(m_ctx.device(), m_stagingBuffer, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = m_ctx.findMemoryType(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(m_ctx.device(), &ai, nullptr, &m_stagingMemory));
        vkBindBufferMemory(m_ctx.device(), m_stagingBuffer, m_stagingMemory, 0);
        m_stagingSize = imageSize;
    }

    auto cmd = m_ctx.beginSingleTimeCommands(cmdPool);
    VkImage srcImage = m_swapchain.image(swapchainImageIndex);

    // Transition color image: post-render layout → TRANSFER_SRC
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = m_swapchain.colorFinalLayout();
    barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = srcImage;
    barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
    barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset       = { 0, 0, 0 };
    region.imageExtent       = { ext.width, ext.height, 1 };
    vkCmdCopyImageToBuffer(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_stagingBuffer, 1, &region);

    // Transition back: TRANSFER_SRC → post-render layout
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout     = m_swapchain.colorFinalLayout();
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    m_ctx.endSingleTimeCommands(cmdPool, cmd);

    void* mapped;
    vkMapMemory(m_ctx.device(), m_stagingMemory, 0, imageSize, 0, &mapped);
    std::vector<uint8_t> pixels(imageSize);
    std::memcpy(pixels.data(), mapped, imageSize);
    vkUnmapMemory(m_ctx.device(), m_stagingMemory);

    // Convert BGRA → RGBA for stb_image_write
    for (size_t i = 0; i < pixels.size(); i += 4)
        std::swap(pixels[i], pixels[i + 2]);

    return pixels;
}

bool RegressionTester::savePNG(const std::string& path, const std::vector<uint8_t>& pixels,
                                uint32_t width, uint32_t height) const {
    int stride = static_cast<int>(width) * 4;
    return stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height),
                          4, pixels.data(), stride) != 0;
}

ImageDiffResult RegressionTester::compareImages(const std::string& testName,
                                                 const std::vector<uint8_t>& captured,
                                                 uint32_t w, uint32_t h,
                                                 const std::string& refPath,
                                                 const std::string& diffOutputPath) const {
    ImageDiffResult result;
    result.testName    = testName;
    result.totalPixels = w * h;

    int rw, rh, rc;
    uint8_t* ref = stbi_load(refPath.c_str(), &rw, &rh, &rc, 4);
    if (!ref) {
        std::cerr << "[Regression] Reference not found: " << refPath << "\n";
        result.passed = false;
        return result;
    }

    if (static_cast<uint32_t>(rw) != w || static_cast<uint32_t>(rh) != h) {
        std::cerr << "[Regression] Size mismatch for " << testName << "\n";
        stbi_image_free(ref);
        result.passed = false;
        return result;
    }

    double sumSqErr  = 0.0;
    double maxDiff   = 0.0;
    uint32_t failed  = 0;

    std::vector<uint8_t> diffPixels;
    if (!diffOutputPath.empty()) diffPixels.resize(captured.size());

    for (size_t i = 0; i < captured.size(); ++i) {
        double diff = static_cast<double>(captured[i]) - static_cast<double>(ref[i]);
        sumSqErr += diff * diff;
        maxDiff = std::max(maxDiff, std::abs(diff));
        if (std::abs(diff) > 3) ++failed;
        if (!diffOutputPath.empty()) {
            // Amplify diff ×4, clamp to [0,255], preserve alpha channel as 255
            uint8_t v = (i % 4 == 3) ? 255u
                : static_cast<uint8_t>(std::min(255.0, std::abs(diff) * 4.0));
            diffPixels[i] = v;
        }
    }

    if (!diffOutputPath.empty() && !diffPixels.empty())
        savePNG(diffOutputPath, diffPixels, w, h);

    stbi_image_free(ref);

    double mse  = sumSqErr / static_cast<double>(captured.size());
    double rmse = std::sqrt(mse);
    double psnr = (mse < 1e-10) ? 100.0 : 20.0 * std::log10(255.0 / rmse);

    result.rmse         = rmse;
    result.psnrDb       = psnr;
    result.maxPixelDiff = maxDiff;
    result.failedPixels = failed / 4;  // divide by 4 channels
    result.passed       = psnr >= m_psnrThreshold;
    return result;
}

ImageDiffResult RegressionTester::captureAndTest(const std::string& testName,
                                                  uint32_t swapchainImageIndex,
                                                  VkCommandPool cmdPool) {
    uint32_t w, h;
    auto pixels = readbackImage(swapchainImageIndex, cmdPool, w, h);

    std::string capturePath = m_outputDir + "/" + testName + "_captured.png";
    savePNG(capturePath, pixels, w, h);

    std::string refPath  = m_referenceDir + "/" + testName + ".png";
    std::string diffPath = m_outputDir    + "/" + testName + "_diff.png";
    auto result = compareImages(testName, pixels, w, h, refPath, diffPath);

    m_report.totalTests++;
    if (result.passed) m_report.passedTests++;
    m_report.results.push_back(result);

    std::cout << "[Regression] " << testName
              << " | PSNR=" << std::fixed << std::setprecision(1) << result.psnrDb << " dB"
              << " | " << (result.passed ? "PASS" : "FAIL") << "\n";
    return result;
}

void RegressionTester::saveReference(const std::string& testName,
                                      uint32_t swapchainImageIndex,
                                      VkCommandPool cmdPool) {
    uint32_t w, h;
    auto pixels = readbackImage(swapchainImageIndex, cmdPool, w, h);
    std::string path = m_referenceDir + "/" + testName + ".png";
    savePNG(path, pixels, w, h);
    std::cout << "[Regression] Saved reference: " << path << "\n";
}

void RegressionTester::captureAndTestMultiRes(const std::string& testName,
                                               uint32_t swapchainImageIndex,
                                               VkCommandPool cmdPool) {
    uint32_t w, h;
    auto pixels = readbackImage(swapchainImageIndex, cmdPool, w, h);

    // Save native-resolution capture
    savePNG(m_outputDir + "/" + testName + "_captured.png", pixels, w, h);

    struct Scale { uint32_t sw, sh; const char* suffix; };
    Scale scales[] = {
        { w,   h,   ""      },
        { w/2, h/2, "_half" },
        { w/4, h/4, "_qtr"  },
    };

    for (auto& sc : scales) {
        if (sc.sw < 16 || sc.sh < 16) continue;

        std::vector<uint8_t> scaledPx(sc.sw * sc.sh * 4);
        stbir_resize_uint8_linear(pixels.data(), static_cast<int>(w), static_cast<int>(h), 0,
                                   scaledPx.data(), static_cast<int>(sc.sw), static_cast<int>(sc.sh), 0,
                                   STBIR_RGBA);

        std::string scaledName = testName + sc.suffix;
        std::string refPath    = m_referenceDir + "/" + scaledName + ".png";
        std::string diffPath   = m_outputDir    + "/" + scaledName + "_diff.png";

        auto result = compareImages(scaledName, scaledPx, sc.sw, sc.sh, refPath, diffPath);
        m_report.totalTests++;
        if (result.passed) m_report.passedTests++;
        m_report.results.push_back(result);

        std::cout << "[Regression] " << scaledName
                  << " (" << sc.sw << "x" << sc.sh << ")"
                  << " | PSNR=" << std::fixed << std::setprecision(1) << result.psnrDb << " dB"
                  << " | " << (result.passed ? "PASS" : "FAIL") << "\n";
    }
}

void RegressionTester::saveReferenceMultiRes(const std::string& testName,
                                              uint32_t swapchainImageIndex,
                                              VkCommandPool cmdPool) {
    uint32_t w, h;
    auto pixels = readbackImage(swapchainImageIndex, cmdPool, w, h);

    struct Scale { uint32_t sw, sh; const char* suffix; };
    Scale scales[] = {
        { w,   h,   ""      },
        { w/2, h/2, "_half" },
        { w/4, h/4, "_qtr"  },
    };

    for (auto& sc : scales) {
        if (sc.sw < 16 || sc.sh < 16) continue;

        if (sc.sw == w && sc.sh == h) {
            std::string path = m_referenceDir + "/" + testName + ".png";
            savePNG(path, pixels, w, h);
            std::cout << "[Regression] Saved reference: " << path << "\n";
        } else {
            std::vector<uint8_t> scaledPx(sc.sw * sc.sh * 4);
            stbir_resize_uint8_linear(pixels.data(), static_cast<int>(w), static_cast<int>(h), 0,
                                      scaledPx.data(), static_cast<int>(sc.sw), static_cast<int>(sc.sh), 0,
                                      STBIR_RGBA);
            std::string path = m_referenceDir + "/" + testName + sc.suffix + ".png";
            savePNG(path, scaledPx, sc.sw, sc.sh);
            std::cout << "[Regression] Saved reference: " << path << "\n";
        }
    }
}

void RegressionTester::exportReport(const std::string& path) const {
    json doc;
    doc["total_tests"]  = m_report.totalTests;
    doc["passed_tests"] = m_report.passedTests;
    doc["pass_rate_pct"]= m_report.passRate();
    doc["psnr_threshold_db"] = m_psnrThreshold;

    json results = json::array();
    for (auto& r : m_report.results) {
        results.push_back({
            {"test",          r.testName},
            {"passed",        r.passed},
            {"psnr_db",       r.psnrDb},
            {"rmse",          r.rmse},
            {"max_pixel_diff",r.maxPixelDiff},
            {"failed_pixels", r.failedPixels},
            {"total_pixels",  r.totalPixels},
        });
    }
    doc["results"] = results;

    std::ofstream f(path);
    f << doc.dump(2);
    std::cout << "[Regression] Report written: " << path << "\n";
}

} // namespace tgt
