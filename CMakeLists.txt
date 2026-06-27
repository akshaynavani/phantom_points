cmake_minimum_required(VERSION 3.16)
project(phantom_points VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PHANTOM_BUILD_TESTS "Build unit tests" ON)

find_package(Threads REQUIRED)

add_library(phantom
  src/geometry.cpp
  src/scene.cpp
  src/config.cpp
  src/noise.cpp
  src/lidar.cpp
  src/params.cpp
)
target_include_directories(phantom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(phantom PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(phantom PRIVATE /W4)
else()
  target_compile_options(phantom PRIVATE -Wall -Wextra -Wpedantic)
endif()
