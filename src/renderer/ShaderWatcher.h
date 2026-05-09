#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iostream>
#include "Pipeline.h"

namespace tgt {

// Polls .spv file modification times each frame and hot-reloads the associated
// Pipeline when a change is detected.  Cost per poll: one stat() call per entry.
class ShaderWatcher {
public:
    struct Entry {
        std::string                                    spvPath;
        Pipeline*                                      pipeline;
        std::filesystem::file_time_type                lastWriteTime;
    };

    void watch(const std::string& spvPath, Pipeline* pipeline) {
        Entry e;
        e.spvPath  = spvPath;
        e.pipeline = pipeline;
        try { e.lastWriteTime = std::filesystem::last_write_time(spvPath); }
        catch (...) { e.lastWriteTime = {}; }
        m_entries.push_back(e);
    }

    // Returns true if at least one pipeline was reloaded this call.
    bool poll() {
        bool anyReloaded = false;
        for (auto& e : m_entries) {
            std::filesystem::file_time_type t{};
            try { t = std::filesystem::last_write_time(e.spvPath); } catch (...) { continue; }
            if (t == e.lastWriteTime) continue;
            e.lastWriteTime = t;
            if (e.pipeline && e.pipeline->tryReload()) {
                std::cout << "[ShaderWatcher] Reloaded: " << e.spvPath << "\n";
                anyReloaded = true;
            } else {
                std::cerr << "[ShaderWatcher] Reload failed: " << e.spvPath << "\n";
            }
        }
        return anyReloaded;
    }

private:
    std::vector<Entry> m_entries;
};

} // namespace tgt
