#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <functional>

namespace tgt {

class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const { return glfwWindowShouldClose(m_window); }
    void pollEvents()  const { glfwPollEvents(); }

    VkSurfaceKHR createSurface(VkInstance instance);
    void         destroySurface(VkInstance instance, VkSurfaceKHR surface);

    int width()  const { return m_width; }
    int height() const { return m_height; }
    bool wasResized() const { return m_resized; }
    void clearResized()     { m_resized = false; }

    GLFWwindow* handle() const { return m_window; }

private:
    static void framebufferResizeCallback(GLFWwindow* win, int w, int h);

    GLFWwindow* m_window  = nullptr;
    int         m_width;
    int         m_height;
    bool        m_resized = false;
};

} // namespace tgt
