#include "Window.h"
#include <stdexcept>

namespace tgt {

Window::Window(int width, int height, const std::string& title)
    : m_width(width), m_height(height)
{
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window) throw std::runtime_error("glfwCreateWindow failed");
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
}

Window::~Window() {
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}

VkSurfaceKHR Window::createSurface(VkInstance instance) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, m_window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");
    return surface;
}

void Window::destroySurface(VkInstance instance, VkSurfaceKHR surface) {
    vkDestroySurfaceKHR(instance, surface, nullptr);
}

void Window::framebufferResizeCallback(GLFWwindow* win, int w, int h) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(win));
    self->m_resized = true;
    self->m_width   = w;
    self->m_height  = h;
}

} // namespace tgt
