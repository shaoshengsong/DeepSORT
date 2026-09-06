cmake_minimum_required(VERSION 3.16)
project(DeepTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Dependency paths (third-party libs live under external/, adjust per machine) ----
set(OPENCV_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/external/opencv/build" CACHE PATH "OpenCV build dir (contains OpenCVConfig.cmake)")
set(ORT_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime" CACHE PATH "ONNX Runtime directory")
set(CANKAO_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/reference/cankao" CACHE PATH "Reference (cankao) source directory")

# ---- OpenCV ----
set(OpenCV_DIR "${OPENCV_DIR}")
find_package(OpenCV REQUIRED)

# ---- ONNX Runtime ----
if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found, check ORT_DIR: ${ORT_DIR}")
endif()
include_directories("${ORT_DIR}/include")
link_directories("${ORT_DIR}/lib")

# ---- Project include directories ----
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/detector)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/app)

# ---- Sources ----
file(GLOB_RECURSE CORE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detector/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reid/*.cpp)
# The linear-assignment solver (LAPJV) is provided in-tree at
# src/core/lap_solver.cpp and picked up by the GLOB_RECURSE above;
# no external cankao implementation is required.

# ---- Static library ----
add_library(deeptracker STATIC ${CORE_SRC})
target_include_directories(deeptracker
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
        ${ORT_DIR}/include)
target_link_libraries(deeptracker PUBLIC ${OpenCV_LIBS} onnxruntime)

# ---- Demo executable ----
add_executable(track_demo ${CMAKE_CURRENT_SOURCE_DIR}/src/app/cli/track_demo.cpp)
target_link_libraries(track_demo PRIVATE deeptracker)

# ---- GUI executable (Win32, runs by double-click) ----
add_executable(track_gui WIN32
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app/gui/track_gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app/gui/track_gui.rc)
target_link_libraries(track_gui PRIVATE deeptracker)

# ---- Runtime resource copy (models / configs) next to the executable,
#      so the output is self-contained and runs by double-click ----
set(_BIN_DIR "$<TARGET_FILE_DIR:track_gui>")
set(_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
# Tracker parameters are compiled in (TrackerConfig::defaults_for); only the
# ONNX models need to be copied next to the executable.
add_custom_command(TARGET track_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_BIN_DIR}/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_SRC}/models/yolo26s.onnx" "${_BIN_DIR}/models/yolo26s.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_SRC}/models/yolo26s-reid.onnx" "${_BIN_DIR}/yolo26s-reid.onnx"
    COMMENT "Copying models to the output directory")

# ---- DLL deployment note ----
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "ORT lib dir : ${ORT_DIR}/lib")
message(STATUS "Before running, copy onnxruntime.dll and opencv_world*.dll next to the executable")
