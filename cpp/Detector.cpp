#include "Detector.h"

Detector* Detector::m_pInstance = nullptr;

Detector::Detector() {}

Detector::~Detector() {
    rknn_perf_detail perf_detail;
    ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
    printf("---> %s\n", perf_detail.perf_data);

    release();
    if (m_pInstance != nullptr) {
        delete m_pInstance;
    }
    m_pInstance = nullptr;
}

bool Detector::init(const char* model_path) {
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_post_process();
    if (ret != 0) {
        printf("init_post_process fail! ret=%d model_path=%s\n", ret, model_path);
        return false;
    }

    ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        release();
        return false;
    }

    memset(&src_image, 0, sizeof(image_buffer_t));
    return true;
}

bool Detector::deinit() {
    release();
}

void resetStDetectResult(stDetectResult &result) {
    if (result.nDetectNum == 0) return;
    result.nDetectNum = 0;
    delete []result.boxes;
    delete []result.prob;
    delete []result.classes;
}

stDetectResult* Detector::detect(char* pChar, int nWidth, int nHeight) {
    int sw_out_size = nWidth * nHeight * 3;
    unsigned char* sw_out_buf = src_image.virt_addr;
    if (sw_out_buf == NULL) {
        sw_out_buf = (unsigned char*)malloc(sw_out_size * sizeof(unsigned char));
    }
    if (sw_out_buf == NULL) {
        printf("sw_out_buf is NULL\n");
        return &stResult;
    }

    memcpy(sw_out_buf, pChar, sw_out_size);

    src_image.width = nWidth;
    src_image.height = nHeight;
    src_image.size = sw_out_size;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.virt_addr = sw_out_buf;

    ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d\n", ret);
        return &stResult;
    }
    printf("resetStDetectResult \n");
    resetStDetectResult(stResult);
    if (od_results.count > 0) {
        stResult.nDetectNum = od_results.count;
        stResult.classes = new int[od_results.count];
        stResult.boxes = new int[od_results.count * 4];
        stResult.prob = new float[od_results.count];
    }
    printf("draw \n");
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

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);

        stResult.classes[i] = det_result->cls_id;
        stResult.boxes[(i << 2)] = det_result->box.left;
        stResult.boxes[(i << 2) + 1] = det_result->box.top;
        stResult.boxes[(i << 2) + 2] = det_result->box.right;
        stResult.boxes[(i << 2) + 3] = det_result->box.bottom;
        stResult.prob[i] = det_result->prop;
    }

    if (stResult.pFrame == nullptr) {
        stResult.pFrame = new char[src_image.size];
    }
    memcpy(stResult.pFrame, src_image.virt_addr, src_image.size);

    // performance
    rknn_perf_detail perf_detail;
    ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
    printf("---> %s\n", perf_detail.perf_data);

    return &stResult;
}

void Detector::setCallback(CBFun_Callback pFunc, void* pUser) {
    m_pCallback = pFunc;
    m_pUser = pUser;
}

void Detector::detectAsync(char* pChar, int nWidth, int nHeight) {

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
    } while (exitListener.wait_for(std::chrono::microseconds(1)) == std::future_status::timeout);
}
