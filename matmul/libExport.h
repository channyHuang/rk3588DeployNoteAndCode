
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

template<typename Ti, typename To>
bool matrixMultiply(int M, int K, int N);

D_EXTERN_C D_SHARE_EXPORT void setMatMulType(int nType);

// 异步检测的回调函数定义
typedef void (__stdcall *CBFun_Callback)(void* pData, void* pUser);
// 设置异步检测的回调函数 
D_EXTERN_C D_SHARE_EXPORT void setCallback(CBFun_Callback pFunc, void *pUser);

#endif // DLLEXPORT_H