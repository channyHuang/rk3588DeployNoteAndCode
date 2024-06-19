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

class DetectThread : public StopThread {
public:
    DetectThread() {}
    ~DetectThread() {}

    void release() {
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

    bool init(const char*model_path, const char*image_path) {
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

        init_post_process();

        ret = init_yolov8_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            release();
            return false;
        }

        memset(&src_image, 0, sizeof(image_buffer_t));
        pImagePath = image_path;
        return true;
    }

private:
    image_buffer_t src_image;
    const char* pImagePath = "./model/images/";
    rknn_app_context_t rknn_app_ctx;
    object_detect_result_list od_results;
    int ret;

private:
    virtual void threadLoop(std::future<void> exitListener) {
        fs::path path(pImagePath);
        auto itr = fs::recursive_directory_iterator(path);
        auto end = fs::recursive_directory_iterator();
        do {
            if (itr == end) {
                printf("end\n");
                break;
            }
            auto &entry = itr;
            ret = read_image(entry->path().c_str(), &src_image);

            if (ret != 0)
            {
                printf("read image fail! ret=%d image_path=%s\n", ret, entry);
                itr++;
                continue;
            }

            ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
            if (ret != 0)
            {
                printf("init_yolov8_model fail! ret=%d\n", ret);
                itr++;
                continue;
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

                draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

                sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
                draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
            }
            write_image(("./result/" + std::string(entry->path().filename())).c_str(), &src_image);

            itr++;
        } while (exitListener.wait_for(std::chrono::microseconds(1)) == std::future_status::timeout);

        rknn_perf_detail perf_detail;
        ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
        printf("---> %s\n", perf_detail.perf_data);
    }
};

int main(int argc, char** argv) {
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    DetectThread thread;
    thread.init(model_path, image_path);
    thread.join();
    // thread.run();
    // std::this_thread::sleep_for(std::chrono::minutes(10));
        
    return 0;
}