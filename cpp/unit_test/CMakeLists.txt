cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "test_lib") 
project(${PROJECT_NAME})

find_package(OpenCV REQUIRED)


set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCE_FILES ./*.cpp)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    ${SOURCE_FILES}
    )
    
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    )
