#include "DebugUI.h"
#include "core/VulkanContext.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace tgt {

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS) throw std::runtime_error(#x " failed"); } while(0)

DebugUI::DebugUI(VulkanContext& ctx, GLFWwindow* window,
                 VkRenderPass renderPass, uint32_t imageCount,
                 VkCommandPool /*cmdPool*/)
    : m_ctx(ctx)
{
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pi{};
    pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets       = 1;
    pi.poolSizeCount = 1;
    pi.pPoolSizes    = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx.device(), &pi, nullptr, &m_imguiPool));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 4.0f;
    ImGui::GetStyle().Alpha          = 0.92f;

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance       = ctx.instance();
    ii.PhysicalDevice = ctx.physicalDevice();
    ii.Device         = ctx.device();
    ii.QueueFamily    = ctx.queueFamilies().graphics.value();
    ii.Queue          = ctx.graphicsQueue();
    ii.DescriptorPool = m_imguiPool;
    ii.RenderPass     = renderPass;
    ii.MinImageCount  = imageCount;
    ii.ImageCount     = imageCount;
    ii.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&ii);
    ImGui_ImplVulkan_CreateFontsTexture();
}

DebugUI::~DebugUI() {
    vkDeviceWaitIdle(m_ctx.device());
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(m_ctx.device(), m_imguiPool, nullptr);
}

void DebugUI::newFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void DebugUI::render(VkCommandBuffer cmd, const UIFrameData& data) {
    buildPanels(data);
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void DebugUI::buildPanels(const UIFrameData& data) {
    m_gpuHistory.push_back(data.gpuFrameMs);
    m_cpuHistory.push_back(data.cpuFrameMs);
    if ((int)m_gpuHistory.size() > kHistorySize) m_gpuHistory.pop_front();
    if ((int)m_cpuHistory.size() > kHistorySize) m_cpuHistory.pop_front();

    // Left column (x=10)
    panelFrameTiming(data);
    panelPipelineStats(data);
    panelPipelineInspector(data);
    panelValidationLog();
    panelSceneControl();
    panelVRAMBudget(data);

    // Right column (x=330)
    panelCommandBufferInspector(data);
    panelSubmeshTimings(data);
    panelReplayControls();
    panelPerfCounters();
}

void DebugUI::panelFrameTiming(const UIFrameData& data) {
    std::vector<float> gpuVec(m_gpuHistory.begin(), m_gpuHistory.end());
    std::vector<float> cpuVec(m_cpuHistory.begin(), m_cpuHistory.end());
    float gpuMax = *std::max_element(gpuVec.begin(), gpuVec.end());
    float cpuMax = *std::max_element(cpuVec.begin(), cpuVec.end());

    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 255}, ImGuiCond_Once);
    ImGui::Begin("Frame Timing");
    ImGui::Text("FPS        %.1f", data.fps);
    ImGui::Text("CPU frame  %.3f ms", data.cpuFrameMs);
    ImGui::Text("GPU frame  %.3f ms", data.gpuFrameMs);
    ImGui::Separator();
    char gpuLbl[40]; std::snprintf(gpuLbl, sizeof(gpuLbl), "GPU %.4f ms", data.gpuFrameMs);
    char cpuLbl[40]; std::snprintf(cpuLbl, sizeof(cpuLbl), "CPU %.4f ms", data.cpuFrameMs);
    ImGui::PlotLines("##gpu", gpuVec.data(), (int)gpuVec.size(), 0,
                     gpuLbl, 0.0f, gpuMax * 1.5f + 0.0001f, {295, 45});
    ImGui::PlotLines("##cpu", cpuVec.data(), (int)cpuVec.size(), 0,
                     cpuLbl, 0.0f, cpuMax * 1.5f + 0.001f,  {295, 45});
    ImGui::Separator();
    ImGui::Text("Barrier probe  %.4f ms", data.barrierMs);
    ImGui::Text("Frame jitter   %.4f ms", data.jitterMs);
    if (data.syncSuspected)
        ImGui::TextColored({1.0f, 0.5f, 0.2f, 1.0f}, "!! Sync stall suspected");
    ImGui::Text("GPU spikes     %u", data.spikeCount);
    ImGui::End();
}

void DebugUI::panelPipelineStats(const UIFrameData& data) {
    ImGui::SetNextWindowPos({10, 275}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 145}, ImGuiCond_Once);
    ImGui::Begin("Pipeline Statistics");
    ImGui::Text("Draw calls       %u",   data.drawCalls);
    ImGui::Text("VS invocations   %llu", (unsigned long long)data.vsInvocations);
    ImGui::Text("FS invocations   %llu", (unsigned long long)data.fsInvocations);
    ImGui::Text("IA primitives    %llu", (unsigned long long)data.iaPrimitives);
    ImGui::Text("Clip primitives  %llu", (unsigned long long)data.clippingPrims);
    ImGui::Separator();
    if (data.overdrawRatio >= 0.0f)
        ImGui::Text("Overdraw ratio   %.2fx", data.overdrawRatio);
    ImGui::End();
}

void DebugUI::panelVRAMBudget(const UIFrameData& data) {
    if (data.vramHeaps.empty()) return;
    ImGui::SetNextWindowPos({10, 770}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 75}, ImGuiCond_Once);
    ImGui::Begin("VRAM Budget");
    for (int i = 0; i < static_cast<int>(data.vramHeaps.size()); ++i) {
        auto& h = data.vramHeaps[i];
        if (h.budgetMiB < 1.0f) continue;
        float frac = (h.budgetMiB > 0.0f) ? h.usedMiB / h.budgetMiB : 0.0f;
        char overlay[48];
        std::snprintf(overlay, sizeof(overlay), "Heap%d  %.0f / %.0f MiB",
                      i, h.usedMiB, h.budgetMiB);
        ImVec4 barCol = (frac > 0.85f)
            ? ImVec4{1.0f, 0.3f, 0.3f, 1.0f}
            : ImVec4{0.3f, 0.8f, 0.3f, 1.0f};
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barCol);
        ImGui::ProgressBar(frac, {-1, 16}, overlay);
        ImGui::PopStyleColor();
    }
    ImGui::End();
}

void DebugUI::panelPipelineInspector(const UIFrameData& data) {
    ImGui::SetNextWindowPos({10, 415}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 70}, ImGuiCond_Once);
    ImGui::Begin("Pipeline Inspector");
    ImGui::Text("Active  %s", data.pipelineName.c_str());
    if (data.meshShaderActive)
        ImGui::TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "VK_EXT_mesh_shader  ACTIVE");
    else
        ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "Vertex pipeline (mesh shader N/A)");
    ImGui::End();
}

void DebugUI::panelValidationLog() {
    ImGui::SetNextWindowPos({10, 480}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 200}, ImGuiCond_Once);
    ImGui::Begin("Validation Log");
    const auto& log = m_ctx.validationLog();
    if (log.empty()) {
        ImGui::TextDisabled("No messages");
    } else {
        int start = (int)log.size() > 32 ? (int)log.size() - 32 : 0;
        for (int i = start; i < (int)log.size(); ++i) {
            const auto& msg = log[i];
            ImVec4 col;
            switch (msg.severity) {
                case tgt::ValidationSeverity::Error:   col = {1.0f, 0.3f, 0.3f, 1.0f}; break;
                case tgt::ValidationSeverity::Warning: col = {1.0f, 0.9f, 0.3f, 1.0f}; break;
                default:                               col = {0.6f, 0.6f, 0.6f, 1.0f}; break;
            }
            char buf[128];
            if (msg.text.size() > 110)
                std::snprintf(buf, sizeof(buf), "[%u] %.106s...", msg.frame, msg.text.c_str());
            else
                std::snprintf(buf, sizeof(buf), "[%u] %s", msg.frame, msg.text.c_str());
            ImGui::TextColored(col, "%s", buf);

            if (!msg.suggestion.empty())
                ImGui::TextColored({0.5f, 0.8f, 1.0f, 1.0f}, "  -> %s", msg.suggestion.c_str());
        }
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
            ImGui::SetScrollHereY(1.0f);
    }
    ImGui::End();
}

void DebugUI::panelSceneControl() {
    ImGui::SetNextWindowPos({10, 690}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 75}, ImGuiCond_Once);
    ImGui::Begin("Scene Control");
    ImGui::Text("Active: Scene %d", m_activeScene);
    if (ImGui::Button("Single Cube")) { m_activeScene = 0; if (m_sceneCallback) m_sceneCallback(0); }
    ImGui::SameLine();
    if (ImGui::Button("5x5 Grid")) { m_activeScene = 1; if (m_sceneCallback) m_sceneCallback(1); }
    ImGui::End();
}

void DebugUI::panelCommandBufferInspector(const UIFrameData& data) {
    ImGui::SetNextWindowPos({330, 10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 200}, ImGuiCond_Once);
    ImGui::Begin("Command Buffer Inspector");
    ImGui::Text("Draw calls in submission order:");
    ImGui::Separator();
    if (data.drawCallList.empty()) {
        ImGui::TextDisabled("No draw calls recorded");
    } else {
        ImGui::BeginChild("##dclist", {0, 0}, false);
        for (auto& dc : data.drawCallList) {
            ImGui::Text("#%u  %s", dc.index, dc.pipeline.c_str());
            ImGui::TextDisabled("    vtx=%u  idx=%u", dc.vertexCount, dc.indexCount);
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void DebugUI::panelSubmeshTimings(const UIFrameData& data) {
    if (data.submeshTimings.empty()) return;

    ImGui::SetNextWindowPos({330, 220}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 200}, ImGuiCond_Once);
    ImGui::Begin("Submesh GPU Timings");

    // Find max for bar scaling
    float maxMs = 0.001f;
    for (auto& t : data.submeshTimings)
        maxMs = std::max(maxMs, t.gpuMs);

    for (auto& t : data.submeshTimings) {
        const char* label = t.name.c_str();
        if (t.name.size() > 4 && t.name.substr(0, 4) == "sub:") label += 4;

        // Name and timing both in the overlay so bars can fill the full width
        char overlay[64];
        std::snprintf(overlay, sizeof(overlay), "%s  %.4f ms", label, t.gpuMs);
        ImGui::ProgressBar(t.gpuMs / maxMs, {-1, 16}, overlay);
    }
    ImGui::End();
}

void DebugUI::panelReplayControls() {
    ImGui::SetNextWindowPos({330, 430}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 200}, ImGuiCond_Once);
    ImGui::Begin("Replay Controls");

    if (ImGui::Button("Scan Captures")) scanCaptureFiles();
    ImGui::SameLine();
    ImGui::TextDisabled("%d files", (int)m_captureFiles.size());

    if (m_captureFiles.empty()) {
        ImGui::TextDisabled("No capture files found.");
        ImGui::TextDisabled("Run with --capture to generate.");
    } else {
        if (m_selectedCapture >= (int)m_captureFiles.size())
            m_selectedCapture = (int)m_captureFiles.size() - 1;

        ImGui::SliderInt("##frame", &m_selectedCapture, 0, (int)m_captureFiles.size() - 1);
        ImGui::TextDisabled("%s", m_captureFiles[m_selectedCapture].c_str());

        if (ImGui::Button("Replay Selected")) {
            if (m_replayCallback)
                m_replayCallback(m_capturesDir + "/" + m_captureFiles[m_selectedCapture]);
        }

        if (m_replayResultValid) {
            ImGui::Separator();
            ImGui::Text("PSNR: %.1f dB", m_lastPsnr);
            ImGui::TextColored(
                m_replayPassed ? ImVec4{0.2f,1.0f,0.2f,1.0f} : ImVec4{1.0f,0.3f,0.3f,1.0f},
                m_replayPassed ? "PASS" : "FAIL");
        }
    }
    ImGui::End();
}

void DebugUI::panelPerfCounters() {
    if (m_perfCounterNames.empty()) return;
    ImGui::SetNextWindowPos({330, 640}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 200}, ImGuiCond_Once);
    ImGui::Begin("GPU Performance Counters");
    ImGui::TextDisabled("%d counters available (VK_KHR_performance_query)",
                        (int)m_perfCounterNames.size());
    ImGui::Separator();
    ImGui::BeginChild("##perf_list", {0, 0}, false, ImGuiWindowFlags_None);
    for (const auto& name : m_perfCounterNames)
        ImGui::TextUnformatted(name.c_str());
    ImGui::EndChild();
    ImGui::End();
}

void DebugUI::scanCaptureFiles() {
    m_captureFiles.clear();
    if (m_capturesDir.empty()) return;
    try {
        for (auto& entry : fs::directory_iterator(m_capturesDir)) {
            if (entry.path().extension() == ".json")
                m_captureFiles.push_back(entry.path().filename().string());
        }
        std::sort(m_captureFiles.begin(), m_captureFiles.end());
    } catch (...) {}
}

} // namespace tgt
