#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace tgt {

struct CapturedDrawCall {
    uint32_t    vertexCount;
    uint32_t    instanceCount;
    uint32_t    indexCount;
    uint32_t    firstVertex;
    std::string pipeline;
    std::string vertShader;
    std::string fragShader;
    float       viewportW, viewportH;
    double      cpuTimeUs;
    float       model[16];
    float       view[16];
    float       proj[16];
};

struct CapturedFrame {
    uint32_t                      frameIndex;
    double                        wallTimestampMs;
    std::vector<CapturedDrawCall> drawCalls;
    double                        gpuTotalMs;
    uint64_t                      vsInvocations;
    uint64_t                      fsInvocations;
    uint64_t                      iaPrimitives;
};

class FrameCapture {
public:
    explicit FrameCapture(const std::string& outputDir);

    void recordFrame(const CapturedFrame& frame);
    void flush();

    uint32_t capturedFrameCount() const { return static_cast<uint32_t>(m_frames.size()); }

private:
    void writeFrameJSON(const CapturedFrame& frame) const;

    std::string              m_outputDir;
    std::vector<CapturedFrame> m_frames;
};

} // namespace tgt
