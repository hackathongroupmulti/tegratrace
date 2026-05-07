#include "MetricsCollector.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace tgt {

using json = nlohmann::json;
using clock = std::chrono::high_resolution_clock;

MetricsCollector::MetricsCollector()
    : m_sessionStart(clock::now())
{}

void MetricsCollector::beginFrame() {
    m_frameStart = clock::now();
}

void MetricsCollector::endFrame(uint32_t frameIndex, double gpuTimeMs,
                                 uint32_t drawCalls, uint32_t indexCount) {
    auto now     = clock::now();
    double wallMs = std::chrono::duration<double, std::milli>(now - m_frameStart).count();
    m_frames.push_back({ frameIndex, wallMs, gpuTimeMs, drawCalls, indexCount });
}

double MetricsCollector::currentFPS() const {
    if (m_frames.size() < 2) return 0.0;
    // Average over last 60 frames
    size_t window = std::min<size_t>(m_frames.size(), 60);
    double sum    = 0.0;
    for (size_t i = m_frames.size() - window; i < m_frames.size(); ++i)
        sum += m_frames[i].wallTimeMs;
    return 1000.0 / (sum / static_cast<double>(window));
}

static double percentile(std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p / 100.0 * (sorted.size() - 1));
    return sorted[std::min(idx, sorted.size() - 1)];
}

void MetricsCollector::exportJSON(const std::string& path) const {
    if (m_frames.empty()) return;

    std::vector<double> wallTimes, gpuTimes;
    uint64_t totalDrawCalls  = 0;
    uint64_t totalIndexCount = 0;

    for (auto& f : m_frames) {
        wallTimes.push_back(f.wallTimeMs);
        gpuTimes.push_back(f.gpuTimeMs);
        totalDrawCalls  += f.drawCalls;
        totalIndexCount += f.indexCount;
    }

    std::sort(wallTimes.begin(), wallTimes.end());
    std::sort(gpuTimes.begin(), gpuTimes.end());

    double avgWall = std::accumulate(wallTimes.begin(), wallTimes.end(), 0.0) / wallTimes.size();
    double avgGpu  = std::accumulate(gpuTimes.begin(), gpuTimes.end(), 0.0) / gpuTimes.size();

    auto sessionMs = std::chrono::duration<double, std::milli>(
        clock::now() - m_sessionStart).count();

    json doc;
    doc["session_ms"]          = sessionMs;
    doc["total_frames"]        = m_frames.size();
    doc["avg_fps"]             = 1000.0 / avgWall;
    doc["total_draw_calls"]    = totalDrawCalls;
    doc["avg_draw_calls_frame"]= static_cast<double>(totalDrawCalls) / m_frames.size();
    doc["total_index_count"]   = totalIndexCount;

    doc["cpu_frame_time_ms"]["avg"]  = avgWall;
    doc["cpu_frame_time_ms"]["p50"]  = percentile(wallTimes, 50);
    doc["cpu_frame_time_ms"]["p95"]  = percentile(wallTimes, 95);
    doc["cpu_frame_time_ms"]["p99"]  = percentile(wallTimes, 99);
    doc["cpu_frame_time_ms"]["min"]  = wallTimes.front();
    doc["cpu_frame_time_ms"]["max"]  = wallTimes.back();

    doc["gpu_frame_time_ms"]["avg"]  = avgGpu;
    doc["gpu_frame_time_ms"]["p50"]  = percentile(gpuTimes, 50);
    doc["gpu_frame_time_ms"]["p95"]  = percentile(gpuTimes, 95);
    doc["gpu_frame_time_ms"]["p99"]  = percentile(gpuTimes, 99);
    doc["gpu_frame_time_ms"]["min"]  = gpuTimes.front();
    doc["gpu_frame_time_ms"]["max"]  = gpuTimes.back();

    std::ofstream f(path);
    f << doc.dump(2);
    std::cout << "[Metrics] Exported: " << path << "\n";
}

void MetricsCollector::printSummary() const {
    if (m_frames.empty()) return;

    std::vector<double> wallTimes;
    for (auto& f : m_frames) wallTimes.push_back(f.wallTimeMs);
    std::sort(wallTimes.begin(), wallTimes.end());

    double avg = std::accumulate(wallTimes.begin(), wallTimes.end(), 0.0) / wallTimes.size();
    double p50 = percentile(wallTimes, 50);
    double p95 = percentile(wallTimes, 95);

    uint64_t totalDC = 0;
    for (auto& f : m_frames) totalDC += f.drawCalls;

    std::cout << "\n=== TegraTrace Metrics ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Rendered frames       : " << m_frames.size() << "\n";
    std::cout << "  Average FPS           : " << 1000.0 / avg << "\n";
    std::cout << "  CPU frame avg         : " << avg  << " ms\n";
    std::cout << "  CPU p50 frame         : " << p50  << " ms\n";
    std::cout << "  CPU p95 frame         : " << p95  << " ms\n";
    std::cout << "  Total draw calls      : " << totalDC << "\n";
    std::cout << "  Draw calls/frame avg  : "
              << static_cast<double>(totalDC) / m_frames.size() << "\n";
    std::cout << "==========================\n\n";
}

} // namespace tgt
