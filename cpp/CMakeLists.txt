cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_demo)

# Check C++17 support
set(CMAKE_CXX_STANDARD 17)
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
# if(COMPILER_SUPPORTS_CXX17)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#   add_definitions(-DCOMPILEDWITHC17)
#   message(STATUS "Using flag -std=c++17.")
# else()
#   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
# endif()

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(rknpu_yolov8_file rknpu2/yolov8.cc)

if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
    add_definitions(-DRV1106_1103)
    set(rknpu_yolov8_file rknpu2/yolov8_rv1106_1103.cc)
    #dma
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)
else()
#dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)
endif()

if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    add_definitions(-DRKNPU1)
    set(rknpu_yolov8_file rknpu1/yolov8.cc)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# add_executable(${PROJECT_NAME} 
#     # main.cc
#     ./thirdparties/stopthread.h
#     mainThread.cpp
#     postprocess.cc
#     ${rknpu_yolov8_file}
# )

add_library(${PROJECT_NAME} SHARED #STATIC
    ./thirdparties/stopthread.h
    Detector.cpp
    Detector.h
    libExport.cpp
    libExport.h

    postprocess.cc
    ${rknpu_yolov8_file}
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
    log
)
endif()

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads
    stdc++fs)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)