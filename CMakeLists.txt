cmake_minimum_required(VERSION 3.5)

project(DeepSORT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)


set(ONNXRUNTIME_DIR "/home/a/lib/onnxruntime-linux-x64-1.12.1")

include_directories("${ONNXRUNTIME_DIR}/include"
    )
find_package(OpenCV 4 REQUIRED )


include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/tracker/deepsort/include
    ${CMAKE_SOURCE_DIR}/tracker/bytetrack/include
    ${CMAKE_SOURCE_DIR}/detector/YOLOv5/include

    )


add_executable(DeepSORT
    detector/YOLOv5/src/YOLOv5Detector.cpp

    tracker/deepsort/src/FeatureTensor.cpp
    tracker/deepsort/src/model.cpp
    tracker/deepsort/src/kalmanfilter.cpp
    tracker/deepsort/src/linear_assignment.cpp
    tracker/deepsort/src/nn_matching.cpp
    tracker/deepsort/src/track.cpp
    tracker/deepsort/src/tracker.cpp
    tracker/deepsort/src/munkres.cpp
    tracker/deepsort/src/hungarianoper.cpp

    tracker/bytetrack/src/BytekalmanFilter.cpp
    tracker/bytetrack/src/BYTETracker.cpp
    tracker/bytetrack/src/lapjv.cpp
    tracker/bytetrack/src/STrack.cpp
    tracker/bytetrack/src/utils.cpp

    main.cpp)
target_link_libraries(DeepSORT PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} Eigen3::Eigen)




add_executable(test_detector
    detector/YOLOv5/src/YOLOv5Detector.cpp

    test_detector.cpp)
target_link_libraries(test_detector PRIVATE  ${OpenCV_LIBS} Eigen3::Eigen)
