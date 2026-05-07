#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace tgt {

struct FrameMetric {
    uint32_t frameIndex;
    double   wallTimeMs;
    double   gpuTimeMs;
    uint32_t drawCalls;
    uint32_t indexCount;
};

class MetricsCollector {
public:
    MetricsCollector();

    void beginFrame();
    void endFrame(uint32_t frameIndex, double gpuTimeMs,
                  uint32_t drawCalls, uint32_t indexCount);

    double   currentFPS()      const;
    uint64_t totalFrames()     const { return m_frames.size(); }

    // Write metrics.json with FPS, latency percentiles, draw call stats
    void exportJSON(const std::string& path) const;
    void printSummary()                      const;

private:
    std::vector<FrameMetric>                      m_frames;
    std::chrono::high_resolution_clock::time_point m_frameStart;
    std::chrono::high_resolution_clock::time_point m_sessionStart;
};

} // namespace tgt
