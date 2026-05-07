#include "DebugUI.h"
#include "core/VulkanContext.h"
#include <imgui.h>
#include <algorithm>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstdio>

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

    std::vector<float> gpuVec(m_gpuHistory.begin(), m_gpuHistory.end());
    std::vector<float> cpuVec(m_cpuHistory.begin(), m_cpuHistory.end());
    float gpuMax = *std::max_element(gpuVec.begin(), gpuVec.end());
    float cpuMax = *std::max_element(cpuVec.begin(), cpuVec.end());

    // --- Frame Timing ---
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 210}, ImGuiCond_Once);
    ImGui::Begin("Frame Timing");
    ImGui::Text("FPS        %.1f", data.fps);
    ImGui::Text("CPU frame  %.3f ms", data.cpuFrameMs);
    ImGui::Text("GPU frame  %.3f ms", data.gpuFrameMs);
    ImGui::Separator();
    char gpuLbl[40]; std::snprintf(gpuLbl, sizeof(gpuLbl), "GPU %.4f ms", data.gpuFrameMs);
    char cpuLbl[40]; std::snprintf(cpuLbl, sizeof(cpuLbl), "CPU %.4f ms", data.cpuFrameMs);
    ImGui::PlotLines("##gpu", gpuVec.data(), (int)gpuVec.size(), 0,
                     gpuLbl, 0.0f, gpuMax * 1.5f + 0.0001f, {295, 55});
    ImGui::PlotLines("##cpu", cpuVec.data(), (int)cpuVec.size(), 0,
                     cpuLbl, 0.0f, cpuMax * 1.5f + 0.001f,  {295, 55});
    ImGui::End();

    // --- Pipeline Statistics ---
    ImGui::SetNextWindowPos({10, 230}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 130}, ImGuiCond_Once);
    ImGui::Begin("Pipeline Statistics");
    ImGui::Text("Draw calls       %u",   data.drawCalls);
    ImGui::Text("VS invocations   %llu", (unsigned long long)data.vsInvocations);
    ImGui::Text("FS invocations   %llu", (unsigned long long)data.fsInvocations);
    ImGui::Text("IA primitives    %llu", (unsigned long long)data.iaPrimitives);
    ImGui::Text("Clip primitives  %llu", (unsigned long long)data.clippingPrims);
    ImGui::End();

    // --- Pipeline Inspector ---
    ImGui::SetNextWindowPos({10, 370}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 55}, ImGuiCond_Once);
    ImGui::Begin("Pipeline Inspector");
    ImGui::Text("Active  %s", data.pipelineName.c_str());
    ImGui::End();

    // --- Validation Log ---
    ImGui::SetNextWindowPos({10, 435}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 200}, ImGuiCond_Once);
    ImGui::Begin("Validation Log");
    const auto& log = m_ctx.validationLog();
    if (log.empty()) {
        ImGui::TextDisabled("No messages");
    } else {
        // Show last 32 messages, newest at bottom
        int start = (int)log.size() > 32 ? (int)log.size() - 32 : 0;
        for (int i = start; i < (int)log.size(); ++i) {
            const auto& msg = log[i];
            ImVec4 col;
            switch (msg.severity) {
                case tgt::ValidationSeverity::Error:   col = {1.0f, 0.3f, 0.3f, 1.0f}; break;
                case tgt::ValidationSeverity::Warning: col = {1.0f, 0.9f, 0.3f, 1.0f}; break;
                default:                               col = {0.6f, 0.6f, 0.6f, 1.0f}; break;
            }
            // Truncate long messages to fit the panel
            const char* text = msg.text.c_str();
            size_t len = msg.text.size();
            char buf[128];
            if (len > 120) {
                std::snprintf(buf, sizeof(buf), "[%u] %.116s...", msg.frame, text);
                ImGui::TextColored(col, "%s", buf);
            } else {
                ImGui::TextColored(col, "[%u] %s", msg.frame, text);
            }
        }
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
            ImGui::SetScrollHereY(1.0f);
    }
    ImGui::End();
}

} // namespace tgt
