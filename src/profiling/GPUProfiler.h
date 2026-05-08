#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace tgt {

class VulkanContext;

struct PassTiming {
    std::string name;
    double      gpuTimeMs = 0.0;   // from timestamp queries
    uint64_t    samples   = 0;
};

struct PipelineStats {
    uint64_t inputAssemblyVertices    = 0;
    uint64_t inputAssemblyPrimitives  = 0;
    uint64_t vertexShaderInvocations  = 0;
    uint64_t clippingInvocations      = 0;
    uint64_t clippingPrimitives       = 0;
    uint64_t fragmentShaderInvocations= 0;
};

struct FrameGPUReport {
    double        totalGpuMs    = 0.0;
    double        jitterMs      = 0.0;   // deviation from rolling mean (sync stall indicator)
    bool          syncSuspected = false; // true when jitter > 50% of mean GPU time
    std::vector<PassTiming> passes;
    PipelineStats pipelineStats;
};

struct FrameSpike {
    uint32_t frameSlot;
    double   gpuMs;
    double   p99Ms;
};

class GPUProfiler {
public:
    static constexpr uint32_t kMaxPasses = 64;

    GPUProfiler(VulkanContext& ctx, uint32_t framesInFlight);
    ~GPUProfiler();

    GPUProfiler(const GPUProfiler&) = delete;
    GPUProfiler& operator=(const GPUProfiler&) = delete;

    // Call before rendering a named pass
    void beginPass(VkCommandBuffer cmd, uint32_t frameIdx, const std::string& passName);
    // Call after rendering a named pass
    void endPass(VkCommandBuffer cmd, uint32_t frameIdx);

    // Begin pipeline statistics collection for a frame
    void beginPipelineStats(VkCommandBuffer cmd, uint32_t frameIdx);
    void endPipelineStats(VkCommandBuffer cmd, uint32_t frameIdx);

    // Read results from last frame (call after submit + fence wait)
    FrameGPUReport readResults(uint32_t frameIdx);

    // Most recent readback result (valid after first kMaxFramesInFlight frames)
    const FrameGPUReport& lastReport() const {
        static const FrameGPUReport kEmpty{};
        return m_frameReports.empty() ? kEmpty : m_frameReports.back();
    }

    const std::vector<FrameSpike>& spikes() const { return m_spikes; }

    // Aggregate stats across all collected frames
    void printSummary() const;
    void exportJSON(const std::string& path) const;

private:
    VulkanContext& m_ctx;
    float          m_timestampPeriodNs;
    uint32_t       m_framesInFlight;

    // Two timestamp queries per pass (begin/end) x kMaxPasses x framesInFlight
    VkQueryPool m_timestampPool = VK_NULL_HANDLE;
    VkQueryPool m_statsPool     = VK_NULL_HANDLE;

    // Track which pass names were recorded per frame
    struct FramePassInfo {
        std::vector<std::string> names;
        uint32_t                 count = 0;
    };
    std::vector<FramePassInfo> m_framePasses;

    // Accumulated per-pass history for summary
    std::unordered_map<std::string, std::vector<double>> m_passHistory;
    std::vector<FrameGPUReport>                          m_frameReports;
    std::vector<FrameSpike>                              m_spikes;

    static constexpr uint32_t kStatsCount = 7;
};

} // namespace tgt
