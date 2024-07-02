#ifndef DLLEXPORT_H
#define DLLEXPORT_H

#include <string.h>
#include <string>

#ifdef __cplusplus
#define D_EXTERN_C extern "C"
#else
#define D_EXTERN_C
#endif

#define __SHARE_EXPORT

#ifdef __SHARE_EXPORT
#define D_SHARE_EXPORT D_DECL_EXPORT
#else
#define D_SHARE_EXPORT D_DECL_IMPORT
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) || defined(__WIN32__)
#define D_CALLTYPE __stdcall
#define D_DECL_EXPORT __declspec(dllexport)
#define D_DECL_IMPORT
#else
#define __stdcall
#define D_CALLTYPE
#define D_DECL_EXPORT __attribute__((visibility("default")))
#define D_DECL_IMPORT __attribute__((visibility("default")))
#endif

typedef struct stDetectResult {
    unsigned char* pFrame = nullptr;
    int nDetectNum = 0;
    int* classes = nullptr;
    int* boxes = nullptr;
    float* prob = nullptr;
};

D_EXTERN_C D_SHARE_EXPORT void Log(const char* pString);
D_EXTERN_C D_SHARE_EXPORT bool Init(const char* pModelString);
D_EXTERN_C D_SHARE_EXPORT bool Deinit();
D_EXTERN_C D_SHARE_EXPORT stDetectResult* Detect(char* pChar, int nWidth, int nHeight);
D_EXTERN_C D_SHARE_EXPORT void DetectAsync(char* pChar, int nWidth, int nHeight);

typedef void (__stdcall *CBFun_Callback)(std::string sMsg, int nSeq, void* pUser);
D_EXTERN_C D_SHARE_EXPORT void SetCallback(CBFun_Callback pFunc, void *pUser);



#endif // DLLEXPORT_H