#include "GPUProfiler.h"
#include "core/VulkanContext.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

namespace tgt {

using json = nlohmann::json;

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

GPUProfiler::GPUProfiler(VulkanContext& ctx, uint32_t framesInFlight)
    : m_ctx(ctx),
      m_timestampPeriodNs(ctx.timestampPeriod()),
      m_framesInFlight(framesInFlight)
{
    m_framePasses.resize(framesInFlight);

    // Timestamp pool: 2 queries (begin/end) x kMaxPasses x framesInFlight
    {
        VkQueryPoolCreateInfo ci{};
        ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = 2 * kMaxPasses * framesInFlight;
        VK_CHECK(vkCreateQueryPool(ctx.device(), &ci, nullptr, &m_timestampPool));
    }

    // Pipeline statistics pool: one set per frame in flight
    {
        VkQueryPoolCreateInfo ci{};
        ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        ci.queryType  = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        ci.queryCount = framesInFlight;
        ci.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT     |
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT   |
            VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT   |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT        |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT         |
            VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        VK_CHECK(vkCreateQueryPool(ctx.device(), &ci, nullptr, &m_statsPool));
    }

    // Reset all queries on first use
    // (will be reset per-frame in beginPass)
}

GPUProfiler::~GPUProfiler() {
    vkDestroyQueryPool(m_ctx.device(), m_timestampPool, nullptr);
    vkDestroyQueryPool(m_ctx.device(), m_statsPool,     nullptr);
}

void GPUProfiler::beginPass(VkCommandBuffer cmd, uint32_t frameIdx, const std::string& passName) {
    auto& info = m_framePasses[frameIdx];
    if (info.count == 0) {
        // Reset all timestamp queries for this frame slot at the start of the first pass
        uint32_t firstQuery = frameIdx * 2 * kMaxPasses;
        vkCmdResetQueryPool(cmd, m_timestampPool, firstQuery, 2 * kMaxPasses);
    }

    if (info.count >= kMaxPasses) return;

    uint32_t queryIdx = frameIdx * 2 * kMaxPasses + info.count * 2;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_timestampPool, queryIdx);

    if (info.count < info.names.size())
        info.names[info.count] = passName;
    else
        info.names.push_back(passName);
}

void GPUProfiler::endPass(VkCommandBuffer cmd, uint32_t frameIdx) {
    auto& info = m_framePasses[frameIdx];
    if (info.count >= kMaxPasses) return;

    uint32_t queryIdx = frameIdx * 2 * kMaxPasses + info.count * 2 + 1;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_timestampPool, queryIdx);
    info.count++;
}

void GPUProfiler::beginPipelineStats(VkCommandBuffer cmd, uint32_t frameIdx) {
    vkCmdResetQueryPool(cmd, m_statsPool, frameIdx, 1);
    vkCmdBeginQuery(cmd, m_statsPool, frameIdx, 0);
}

void GPUProfiler::endPipelineStats(VkCommandBuffer cmd, uint32_t frameIdx) {
    vkCmdEndQuery(cmd, m_statsPool, frameIdx);
}

FrameGPUReport GPUProfiler::readResults(uint32_t frameIdx) {
    FrameGPUReport report{};
    auto& info = m_framePasses[frameIdx];

    if (info.count > 0) {
        uint32_t firstQuery  = frameIdx * 2 * kMaxPasses;
        uint32_t queryCount  = info.count * 2;
        std::vector<uint64_t> timestamps(queryCount, 0);

        vkGetQueryPoolResults(
            m_ctx.device(), m_timestampPool,
            firstQuery, queryCount,
            queryCount * sizeof(uint64_t), timestamps.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        for (uint32_t i = 0; i < info.count; ++i) {
            uint64_t begin = timestamps[i * 2];
            uint64_t end   = timestamps[i * 2 + 1];
            double ms = static_cast<double>(end - begin) * m_timestampPeriodNs * 1e-6;

            PassTiming pt;
            pt.name      = (i < info.names.size()) ? info.names[i] : ("pass_" + std::to_string(i));
            pt.gpuTimeMs = ms;
            pt.samples   = 1;
            report.passes.push_back(pt);
            report.totalGpuMs += ms;

            m_passHistory[pt.name].push_back(ms);
        }
    }

    // Pipeline statistics
    {
        std::array<uint64_t, 6> statsData{};
        VkResult res = vkGetQueryPoolResults(
            m_ctx.device(), m_statsPool,
            frameIdx, 1,
            statsData.size() * sizeof(uint64_t), statsData.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        if (res == VK_SUCCESS) {
            report.pipelineStats.inputAssemblyVertices    = statsData[0];
            report.pipelineStats.inputAssemblyPrimitives  = statsData[1];
            report.pipelineStats.vertexShaderInvocations  = statsData[2];
            report.pipelineStats.clippingInvocations      = statsData[3];
            report.pipelineStats.clippingPrimitives        = statsData[4];
            report.pipelineStats.fragmentShaderInvocations= statsData[5];
        }
    }

    info.count = 0;
    m_frameReports.push_back(report);
    return report;
}

void GPUProfiler::printSummary() const {
    if (m_frameReports.empty()) return;

    std::vector<double> totals;
    for (auto& r : m_frameReports) totals.push_back(r.totalGpuMs);
    std::sort(totals.begin(), totals.end());

    double avg = std::accumulate(totals.begin(), totals.end(), 0.0) / totals.size();
    double p50 = totals[totals.size() * 50 / 100];
    double p95 = totals[totals.size() * 95 / 100];
    double p99 = totals[totals.size() * 99 / 100];

    double spikeThreshold = p99 * 1.5;
    std::vector<size_t> spikeFrames;
    for (size_t i = 0; i < m_frameReports.size(); ++i)
        if (m_frameReports[i].totalGpuMs > spikeThreshold) spikeFrames.push_back(i);

    std::cout << "\n=== GPU Profiling Summary ===\n";
    std::cout << "  Frames sampled : " << m_frameReports.size() << "\n";
    std::cout << "  Total GPU avg  : " << avg  << " ms\n";
    std::cout << "  p50 GPU frame  : " << p50  << " ms\n";
    std::cout << "  p95 GPU frame  : " << p95  << " ms\n";
    std::cout << "  p99 GPU frame  : " << p99  << " ms\n";
    std::cout << "  Spikes (>p99x1.5=" << spikeThreshold << " ms): " << spikeFrames.size() << "\n";
    if (!spikeFrames.empty()) {
        std::cout << "    Frame indices:";
        for (auto idx : spikeFrames) std::cout << " " << idx;
        std::cout << "\n";
    }

    for (auto& [name, history] : m_passHistory) {
        double passAvg = std::accumulate(history.begin(), history.end(), 0.0) / history.size();
        std::cout << "  pass '" << name << "' avg: " << passAvg << " ms\n";
    }
    std::cout << "=============================\n\n";
}

void GPUProfiler::exportJSON(const std::string& path) const {
    json doc;

    std::vector<double> totals;
    for (auto& r : m_frameReports) totals.push_back(r.totalGpuMs);
    if (!totals.empty()) {
        std::sort(totals.begin(), totals.end());
        double avg = std::accumulate(totals.begin(), totals.end(), 0.0) / totals.size();
        double p99 = totals[totals.size() * 99 / 100];
        doc["gpu_summary"]["frames"]         = m_frameReports.size();
        doc["gpu_summary"]["avg_ms"]         = avg;
        doc["gpu_summary"]["p50_ms"]         = totals[totals.size() * 50 / 100];
        doc["gpu_summary"]["p95_ms"]         = totals[totals.size() * 95 / 100];
        doc["gpu_summary"]["p99_ms"]         = p99;
        doc["gpu_summary"]["min_ms"]         = totals.front();
        doc["gpu_summary"]["max_ms"]         = totals.back();

        double spikeThreshold = p99 * 1.5;
        json spikeList = json::array();
        for (size_t i = 0; i < m_frameReports.size(); ++i)
            if (m_frameReports[i].totalGpuMs > spikeThreshold) spikeList.push_back(i);
        doc["gpu_summary"]["spike_threshold_ms"] = spikeThreshold;
        doc["gpu_summary"]["spike_frames"]       = spikeList;
    }

    json passes = json::array();
    for (auto& [name, history] : m_passHistory) {
        double avg = std::accumulate(history.begin(), history.end(), 0.0) / history.size();
        passes.push_back({ {"name", name}, {"avg_ms", avg}, {"samples", history.size()} });
    }
    doc["pass_timings"] = passes;

    // Last frame pipeline stats
    if (!m_frameReports.empty()) {
        auto& s = m_frameReports.back().pipelineStats;
        doc["pipeline_stats"]["ia_vertices"]         = s.inputAssemblyVertices;
        doc["pipeline_stats"]["ia_primitives"]        = s.inputAssemblyPrimitives;
        doc["pipeline_stats"]["vs_invocations"]       = s.vertexShaderInvocations;
        doc["pipeline_stats"]["clip_invocations"]     = s.clippingInvocations;
        doc["pipeline_stats"]["clip_primitives"]      = s.clippingPrimitives;
        doc["pipeline_stats"]["fs_invocations"]       = s.fragmentShaderInvocations;
    }

    std::ofstream f(path);
    f << doc.dump(2);
    std::cout << "[GPUProfiler] Exported: " << path << "\n";
}

} // namespace tgt
