#include "FrameCapture.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace tgt {

using json = nlohmann::json;

FrameCapture::FrameCapture(const std::string& outputDir)
    : m_outputDir(outputDir)
{}

void FrameCapture::recordFrame(const CapturedFrame& frame) {
    m_frames.push_back(frame);
    writeFrameJSON(frame);
}

void FrameCapture::flush() {
    // Write aggregate capture index
    json index;
    index["total_frames"]    = m_frames.size();
    index["captured_frames"] = json::array();

    for (auto& f : m_frames) {
        json entry;
        entry["frame"]         = f.frameIndex;
        entry["timestamp_ms"]  = f.wallTimestampMs;
        entry["draw_calls"]    = f.drawCalls.size();
        entry["gpu_total_ms"]  = f.gpuTotalMs;
        index["captured_frames"].push_back(entry);
    }

    std::ofstream out(m_outputDir + "/capture_index.json");
    out << index.dump(2);
    std::cout << "[FrameCapture] Wrote " << m_frames.size()
              << " captured frames to " << m_outputDir << "\n";
}

void FrameCapture::writeFrameJSON(const CapturedFrame& frame) const {
    json doc;
    doc["frame"]            = frame.frameIndex;
    doc["wall_timestamp_ms"]= frame.wallTimestampMs;
    doc["gpu_total_ms"]     = frame.gpuTotalMs;
    doc["pipeline_stats"]["vs_invocations"] = frame.vsInvocations;
    doc["pipeline_stats"]["fs_invocations"] = frame.fsInvocations;
    doc["pipeline_stats"]["ia_primitives"]  = frame.iaPrimitives;

    json draws = json::array();
    for (auto& dc : frame.drawCalls) {
        json d;
        d["vertex_count"]   = dc.vertexCount;
        d["instance_count"] = dc.instanceCount;
        d["index_count"]    = dc.indexCount;
        d["first_vertex"]   = dc.firstVertex;
        d["pipeline"]       = dc.pipeline;
        d["vert_shader"]    = dc.vertShader;
        d["frag_shader"]    = dc.fragShader;
        d["viewport"]       = { {"w", dc.viewportW}, {"h", dc.viewportH} };
        d["cpu_time_us"]    = dc.cpuTimeUs;
        auto matToArray = [](const float* m, int n) {
            json a = json::array();
            for (int i = 0; i < n; ++i) a.push_back(m[i]);
            return a;
        };
        d["ubo"]["model"] = matToArray(dc.model, 16);
        d["ubo"]["view"]  = matToArray(dc.view,  16);
        d["ubo"]["proj"]  = matToArray(dc.proj,  16);
        draws.push_back(d);
    }
    doc["draw_calls"] = draws;

    std::ostringstream filename;
    filename << m_outputDir << "/frame_"
             << std::setw(6) << std::setfill('0') << frame.frameIndex << ".json";
    std::ofstream out(filename.str());
    out << doc.dump(2);
}

} // namespace tgt
