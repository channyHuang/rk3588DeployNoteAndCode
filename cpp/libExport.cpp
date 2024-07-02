#include "libExport.h"

#include <fstream>
#include "Detector.h"

extern "C"

void Log(const char* pString)
{
    std::ofstream off("liblog.txt", std::ios::app);
    off << pString << std::endl;
    off.close();
}

bool Init(const char* pModelString)
{
    return Detector::getInstance()->init(pModelString);
}

stDetectResult* Detect(char* pChar, int nWidth, int nHeight)
{
    return Detector::getInstance()->detect(pChar, nWidth, nHeight);
}

void DetectAsync(char* pChar, int nWidth, int nHeight) {
    Detector::getInstance()->detectAsync(pChar, nWidth, nHeight);
}

void SetCallback(CBFun_Callback pFunc, void *pUser)
{
    Detector::getInstance()->setCallback(pFunc, pUser);
}

bool Deinit()
{
    return Detector::getInstance()->deinit();
}
