#include <iostream>
#include <experimental/filesystem>
#include <cstring>
#include <chrono>

namespace fs = std::experimental::filesystem;

#include "thirdparties/stopthread.h"
#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include "libExport.h"

// typedef struct stDetectResult {
//     char* pFrame;
//     int nDetectNum;
//     int* classes;
//     int* boxes;
//     float* prob;
// };

class Detector : public StopThread {
public:
    static Detector *getInstance() {
        if (m_pInstance == nullptr) {
            m_pInstance = new Detector();
        }
        return m_pInstance;
    }
    ~Detector();

    bool init(const char*model_path);
    bool deinit();
    stDetectResult* detect(char* pChar, int nWidth = 1920, int nHeight = 1080);
    void detectAsync(char* pChar, int nWidth = 1920, int nHeight = 1080);
    void setCallback(CBFun_Callback pFunc = nullptr, void* pUser = nullptr);

private:
    static Detector *m_pInstance;

    int ret = 0;
    image_buffer_t src_image;
    rknn_app_context_t rknn_app_ctx;
    object_detect_result_list od_results;

    stDetectResult stResult;
    CBFun_Callback m_pCallback = nullptr;
    void* m_pUser;

private:
    Detector();
    void release();
    virtual void threadLoop(std::future<void> exitListener);
};
