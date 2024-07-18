cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "ThreadPoolTest")

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}
    ./threadPool.h
    ./threadPoolTest.cpp
)

target_link_libraries(${PROJECT_NAME}
    dl
    pthread
)