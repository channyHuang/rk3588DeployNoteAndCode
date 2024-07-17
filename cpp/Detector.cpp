#include "Detector.h"

Detector* Detector::m_pInstance = nullptr;

Detector::Detector() {}
// 释放资源
Detector::~Detector() {
    release();
    if (m_pInstance != nullptr) {
        delete m_pInstance;
    }
    m_pInstance = nullptr;
}
// 初始化，输入权重模型路径名称，加载模型
bool Detector::init(const char* model_path) {
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    // 读取Label文件
    ret = init_post_process();
    if (ret != 0) {
        printf("init_post_process fail! ret=%d model_path=%s\n", ret, model_path);
        return false;
    }
    // 初始化SDK，查询模型输入输出信息
    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        release();
        return false;
    }
    // 初始化输入图像
    memset(&src_image, 0, sizeof(image_buffer_t));
    return true;
}

bool Detector::deinit() {
    release();
}
// 释放传输到外部的检测结果资源
void resetStDetectResult(stDetectResult &result) {
    if (result.nDetectNum == 0) return;
    result.nDetectNum = 0;
    delete []result.pBoxes;
    delete []result.pProb;
    delete []result.pClasses;
}
// 复制图像数据到输入图像结构
bool Detector::copyImageData(char* pChar, int nWidth, int nHeight, image_buffer_t& image) {
    int sw_out_size = nWidth * nHeight * 3;
    unsigned char* sw_out_buf = image.virt_addr;
    if (sw_out_buf == NULL) {
        printf("copyImageData malloc %d\n", sw_out_size);
        sw_out_buf = (unsigned char*)malloc(sw_out_size * sizeof(unsigned char));
        // set_image_dma_buf_alloc(&image);
    }
    if (sw_out_buf == NULL) {
        printf("sw_out_buf is NULL\n");
        return false;
    }

    memcpy(sw_out_buf, pChar, sw_out_size);

    image.width = nWidth;
    image.height = nHeight;
    image.size = sw_out_size;
    image.format = IMAGE_FORMAT_RGB888;
    image.virt_addr = sw_out_buf;

    return true;
}
// 检测，输入图像数据及图像宽高，输出检测信息
stDetectResult* Detector::detect(char* pChar, int nWidth, int nHeight) {
    bool suc = copyImageData(pChar, nWidth, nHeight, src_image);
    if (!suc) {
        return &stResult;
    }
    // 检测及后处理
    ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results, &dst_img);
    if (ret != 0)
    {
        printf("inference_yolov8_model fail! ret=%d\n", ret);
        return &stResult;
    }
    // 填充传输到外部的检测结果
    resetStDetectResult(stResult);
    if (od_results.count > 0) {
        stResult.nDetectNum = od_results.count;
        stResult.pClasses = new int[od_results.count];
        stResult.pBoxes = new int[od_results.count << 2];
        stResult.pProb = new float[od_results.count];
    }
    // 画框和概率
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        // printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
        //     det_result->box.left, det_result->box.top,
        //     det_result->box.right, det_result->box.bottom,
        //     det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);

        stResult.pClasses[i] = det_result->cls_id;
        stResult.pBoxes[(i << 2)] = det_result->box.left;
        stResult.pBoxes[(i << 2) + 1] = det_result->box.top;
        stResult.pBoxes[(i << 2) + 2] = det_result->box.right;
        stResult.pBoxes[(i << 2) + 3] = det_result->box.bottom;
        stResult.pProb[i] = det_result->prop;
    }
    // write_image("out.png", &src_image);
    if (stResult.pFrame == nullptr) {
        stResult.pFrame = new unsigned char[src_image.size];
    }
    memcpy(stResult.pFrame, src_image.virt_addr, src_image.size * sizeof(unsigned char));
    stResult.nWidth = src_image.width;
    stResult.nHeight = src_image.height;

    return &stResult;
}
// 设置异步检测的回调函数 
void Detector::setCallback(CBFun_Callback pFunc, void* pUser) {
    m_pCallback = pFunc;
    m_pUser = pUser;
    if (pFunc == nullptr && m_bRunning) {
        stop();
        m_bRunning = false;
    } else if (pFunc != nullptr && !m_bRunning) {
        run();
        m_bRunning = true;
    }
    writeLog(pFunc == nullptr ? "setCallback pFunc is nullptr" : "setCallback");
}
// 异步检测
void Detector::detectAsync(char* pChar, int nWidth, int nHeight) {
    if (output_images.empty()) {
        image_buffer_t image;
        memset(&image, 0, sizeof(image_buffer_t));
        copyImageData(pChar, nWidth, nHeight, image);
        std::lock_guard<std::mutex> locker(mutex);
        while (!input_images.empty()) {
            image_buffer_t &old_image = input_images.front();
            input_images.pop();
            output_images.push(old_image);
        }
        input_images.push(image);
    } else {
        image_buffer_t& image = output_images.front();
        output_images.pop();
        copyImageData(pChar, nWidth, nHeight, image);
        std::lock_guard<std::mutex> locker(mutex);
        while (!input_images.empty()) {
            image_buffer_t &old_image = input_images.front();
            input_images.pop();
            output_images.push(old_image);
        }
        input_images.push(image);
    }
}

void Detector::release() {
    deinit_post_process();
    ret = release_yolov8_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
}

void Detector::threadLoop(std::future<void> exitListener) {
    do {
        image_buffer_t image;
        {
            std::lock_guard<std::mutex> locker(mutex);
            if (input_images.empty()) {
                continue;
            }
            image = input_images.front();
            input_images.pop();
        }

        ret = inference_yolov8_model(&rknn_app_ctx, &image, &od_results, &dst_img);
        if (ret != 0)
        {
            printf("inference_yolov8_model fail! ret=%d\n", ret);
            continue;
        }
        
        resetStDetectResult(stResult);
        if (od_results.count > 0) {
            stResult.nDetectNum = od_results.count;
            stResult.pClasses = new int[od_results.count];
            stResult.pBoxes = new int[od_results.count * 4];
            stResult.pProb = new float[od_results.count];
        }
        
        // 画框和概率
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            draw_rectangle(&image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&image, text, x1, y1 - 20, COLOR_RED, 10);

            stResult.pClasses[i] = det_result->cls_id;
            stResult.pBoxes[(i << 2)] = det_result->box.left;
            stResult.pBoxes[(i << 2) + 1] = det_result->box.top;
            stResult.pBoxes[(i << 2) + 2] = det_result->box.right;
            stResult.pBoxes[(i << 2) + 3] = det_result->box.bottom;
            stResult.pProb[i] = det_result->prop;
        }
        
        if (stResult.pFrame == nullptr) {
            stResult.pFrame = new unsigned char[image.size];
        }
        memcpy(stResult.pFrame, image.virt_addr, image.size * sizeof(unsigned char));
        stResult.nWidth = image.width;
        stResult.nHeight = image.height;

        if (m_pCallback != nullptr) {
            m_pCallback(&stResult, nullptr);
        }
    } while (exitListener.wait_for(std::chrono::microseconds(1)) == std::future_status::timeout);
}
// 设置目标检测置信度阈值，即时生效 
void Detector::setThreshold(float threshold) {
    printf("setThreshold %f\n", threshold);
    BOX_THRESH = threshold;
}
// 设置模型对应的目标总类型数
void Detector::setClassNum(int nClassNum) {
    printf("setClassNum %d\n", nClassNum);
    OBJ_CLASS_NUM = nClassNum;
}
// 打印性能统计信息到控制台
void Detector::printProfile() {
    // performance
    rknn_perf_detail perf_detail;
    ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
    printf("---> %s\n", perf_detail.perf_data);  
}
// 写日志到文件
void Detector::writeLog(const std::string& sMsg) {
    std::ofstream off("liblog.txt", std::ios::app);
    off << sMsg << std::endl;
    off.close();
}
