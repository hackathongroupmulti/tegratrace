cmake_minimum_required(VERSION 3.20)
project(TegraTrace VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

include(FetchContent)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

# ---- Shader compilation ----
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)

function(compile_shader TARGET SHADER)
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV_FILE "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT  ${SPV_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        VERBATIM
    )
    target_sources(${TARGET} PRIVATE ${SPV_FILE})
endfunction()

# ---- Source files ----
set(SOURCES
    src/main.cpp
    src/core/VulkanContext.cpp
    src/core/Window.cpp
    src/core/Swapchain.cpp
    src/renderer/Shader.cpp
    src/renderer/Buffer.cpp
    src/renderer/RenderPass.cpp
    src/renderer/Pipeline.cpp
    src/renderer/Renderer.cpp
    src/profiling/GPUProfiler.cpp
    src/capture/FrameCapture.cpp
    src/validation/RegressionTester.cpp
    src/metrics/MetricsCollector.cpp
    src/external/stb_impl.cpp
)

add_executable(tegratrace ${SOURCES})

target_include_directories(tegratrace PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${stb_SOURCE_DIR}
)

target_link_libraries(tegratrace PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    nlohmann_json::nlohmann_json
)

if(GLSLC)
    compile_shader(tegratrace shaders/triangle.vert)
    compile_shader(tegratrace shaders/triangle.frag)
    compile_shader(tegratrace shaders/mesh.vert)
    compile_shader(tegratrace shaders/mesh.frag)
    add_custom_command(TARGET tegratrace POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_BINARY_DIR}/shaders"
            "$<TARGET_FILE_DIR:tegratrace>/shaders"
    )
else()
    message(WARNING "glslc not found — pre-compiled SPIR-V will be used from shaders/precompiled/")
    add_custom_command(TARGET tegratrace POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/precompiled"
            "$<TARGET_FILE_DIR:tegratrace>/shaders"
    )
endif()

# Output directories for captures and reports
add_custom_command(TARGET tegratrace POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:tegratrace>/captures"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:tegratrace>/reports"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:tegratrace>/references"
)

if(MSVC)
    target_compile_options(tegratrace PRIVATE /W4 /WX-)
else()
    target_compile_options(tegratrace PRIVATE -Wall -Wextra)
endif()

target_compile_definitions(tegratrace PRIVATE
    $<$<CONFIG:Debug>:TEGRATRACE_VALIDATION>
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)
