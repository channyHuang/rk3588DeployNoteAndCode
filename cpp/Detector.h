#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <cstring>
#include <chrono>
#include <queue>
#include <mutex>

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

// 用于导出的检测类Detector
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
    void setThreshold(float fThreshold = 0.8f);
    void setClassNum(int nClassNum = 13);
    void printProfile();

    void detectAsync(char* pChar, int nWidth = 1920, int nHeight = 1080);
    void setCallback(CBFun_Callback pFunc = nullptr, void* pUser = nullptr);

private:
    static Detector *m_pInstance;

    int ret = 0;
    // 输入输出图像
    image_buffer_t src_image;
    image_buffer_t dst_img;
    // 检测上下文
    rknn_app_context_t rknn_app_ctx;
    // 检测结果
    object_detect_result_list od_results;
    // 传输到外部的检测结果 
    stDetectResult stResult;
    // 回调函数
    CBFun_Callback m_pCallback = nullptr;
    // 回调函数传入的用户数据
    void* m_pUser = nullptr;
    // 线程锁
    std::mutex mutex;
    
    bool m_bRunning = false;
    std::queue<image_buffer_t> input_images;
    std::queue<image_buffer_t> output_images;

private:
    Detector();
    void release();
    // 复制图像数据
    bool copyImageData(char* pChar, int nWidth, int nHeight, image_buffer_t& image);
    void writeLog(const std::string& sMsg);
    // 线程循环主体
    virtual void threadLoop(std::future<void> exitListener);
};
