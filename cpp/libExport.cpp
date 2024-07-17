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

bool init(const char* pModelString)
{
    return Detector::getInstance()->init(pModelString);
}

stDetectResult* detect(char* pChar, int nWidth, int nHeight)
{
    return Detector::getInstance()->detect(pChar, nWidth, nHeight);
}

void detectAsync(char* pChar, int nWidth, int nHeight) {
    Detector::getInstance()->detectAsync(pChar, nWidth, nHeight);
}

void setCallback(CBFun_Callback pFunc, void *pUser)
{
    Detector::getInstance()->setCallback(pFunc, pUser);
}

bool deinit()
{
    return Detector::getInstance()->deinit();
}

void setThreshold(float fThreshold) {
    Detector::getInstance()->setThreshold(fThreshold);
}

void setClassNum(int nClassNum) {
    Detector::getInstance()->setClassNum(nClassNum);
}

void printProfile() {
    Detector::getInstance()->printProfile();
}