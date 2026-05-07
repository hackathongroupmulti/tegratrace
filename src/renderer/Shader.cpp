#include "Shader.h"
#include <fstream>
#include <stdexcept>

namespace tgt {

Shader::Shader(VkDevice device, const std::string& spirvPath)
    : m_device(device)
{
    auto code = readSPIRV(spirvPath);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    if (vkCreateShaderModule(device, &ci, nullptr, &m_module) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module: " + spirvPath);
}

Shader::~Shader() {
    if (m_module) vkDestroyShaderModule(m_device, m_module, nullptr);
}

std::vector<char> Shader::readSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open SPIR-V: " + path);
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), static_cast<std::streamsize>(size));
    return buf;
}

} // namespace tgt
