cmake_minimum_required(VERSION 3.10)

project(HGMatmul)

# Check C++17 support
set(CMAKE_CXX_STANDARD 17)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_definitions(-DTEMPLATE_DLL_TEST_EXPORTS)
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../../3rdparty/rknpu2/${CMAKE_SYSTEM_NAME})
if(CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
else()
    set(LIB_ARCH armhf)
endif()
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB COMMONS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# add_executable(${PROJECT_NAME} 
#     main.cc
#     ${HEADERS}
#     ${FILES}
# )

add_library(${PROJECT_NAME} SHARED #STATIC
    ${HEADERS}
    ${COMMONS}
    ${FILES}
    )

target_link_libraries(${PROJECT_NAME}
    ${RKNN_RT_LIB}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
)

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