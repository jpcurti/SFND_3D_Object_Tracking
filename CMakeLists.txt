cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(SFND_3d_Object_Tracking_conan)
add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)

find_package(OpenCV 4.1 REQUIRED)


if(OpenCV_FOUND)
    #include_directories(${OpenCV_INCLUDE_DIRS})
    #link_directories(${OpenCV_LIBRARY_DIRS})
    #add_definitions(${OpenCV_DEFINITIONS})
    # Executable for create matrix exercise
    add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
    target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()



