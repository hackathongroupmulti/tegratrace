#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

namespace tgt {

class Shader {
public:
    Shader(VkDevice device, const std::string& spirvPath);
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    VkShaderModule module() const { return m_module; }

    static std::vector<char> readSPIRV(const std::string& path);

private:
    VkDevice       m_device;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

} // namespace tgt
