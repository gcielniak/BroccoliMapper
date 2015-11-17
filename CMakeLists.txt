cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(PCLMapper)

find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${CMAKE_PROJECT_NAME} src/pcl_mapper.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} ${PCL_LIBRARIES})
