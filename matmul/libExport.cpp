#include "libExport.h"

#include <fstream>
#include "Detector.h"

extern "C"

void log(const char* pString)
{
    std::ofstream off("liblog.txt", std::ios::app);
    off << pString << std::endl;
    off.close();
}

bool init()
{
    return Detector::getInstance()->init();
}

template<typename Ti, typename To>
bool matrixMultiply(int M, int K, int N)
{
    return Detector::getInstance()->matrixMultiplyCheck<Ti, To>(M, K, N);
}

extern "C"
void setMatMulType(int nType) {
    return Detector::getInstance()->setMatMulType(nType);
}

void deinit() {
    Detector::getInstance()->deinit();
}

// float-8bit64
template 
D_SHARE_EXPORT bool matrixMultiply<float, float>(int M, int K, int N);
template
D_SHARE_EXPORT bool matrixMultiply<float16, float16>(int M, int K, int N);
template
D_SHARE_EXPORT bool matrixMultiply<float16, float>(int M, int K, int N);

// int-4bit32
template 
D_SHARE_EXPORT bool matrixMultiply<int, int>(int M, int K, int N);
template 
D_SHARE_EXPORT bool matrixMultiply<int8_t, int8_t>(int M, int K, int N);
template
D_SHARE_EXPORT bool matrixMultiply<int8_t, int32_t>(int M, int K, int N);
