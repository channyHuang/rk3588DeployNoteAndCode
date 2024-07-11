#include <iostream>
#include <dlfcn.h>
#include <unistd.h>
#include <thread>
#include <chrono>

#include "libExport.h"

typedef bool (* Fun_Init)(const char* pModelString);
typedef bool (* Fun_Deinit)();
typedef stDetectResult* (*Fun_Detect)(char* pChar, int nWidth, int nHeight);
typedef void (* Fun_DetectAsync)(char* pChar, int nWidth, int nHeight);
typedef void (* Fun_SetCallback)(CBFun_Callback pFunc, void *pUser);

void DetectCallback(stDetectResult* stResult, void* pUser) {
    printf("DetectCallback\n");
}

int main() {
    void *handle = dlopen(".so", RTLD_LAZY);
    Fun_Init fun_Init = (Fun_Init)dlsym(handle, "Init");
    Fun_Deinit fun_Deinit = (Fun_Deinit)dlsym(handle, "Deinit");
    Fun_Detect fun_Detect = (Fun_Detect)dlsym(handle, "Detect");
    Fun_DetectAsync fun_DetectAsync = (Fun_DetectAsync)dlsym(handle, "DetectAsync");
    Fun_SetCallback fun_SetCallback = (Fun_SetCallback)dlsym(handle, "SetCallback");

    fun_Init("./model/yolov8n.rknn");
    fun_Detect();

    fun_SetCallback(DetectCallback, nullptr);

    int count = 0;
    while (count < 10) {
        if (count & 1) {
            fun_DetectAsync();
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        count++;
    }

    fun_SetCallback(nullptr, nullptr);
    fun_Deinit();
    dlclose(handle);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}