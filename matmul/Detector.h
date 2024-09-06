#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <cstring>
#include <chrono>
#include <queue>
#include <mutex>

namespace fs = std::experimental::filesystem;

#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "Float16.h"
using namespace rknpu2;

#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include "libExport.h"

// 用于导出的检测类Detector
class Detector {
public:
    static Detector *getInstance() {
        if (m_pInstance == nullptr) {
            m_pInstance = new Detector();
        }
        return m_pInstance;
    }
    ~Detector();

    bool init();
    void deinit();

    // A_{MxK} * B_{KxN} = C_{MxN}
    template <typename Ti, typename To>
    bool matrixMultiplyNpu(Ti* dataA, Ti* dataB, int M, int K, int N, To* dataC);

    void setMatMulType(int nType);
    
    template <typename Ti, typename To>
    bool matrixMultiplyCheck(int M, int K, int N);

private:
    static Detector *m_pInstance;

    int ret = 0;
   
    // 回调函数
    CBFun_Callback m_pCallback = nullptr;
    // 回调函数传入的用户数据
    void* m_pUser = nullptr;
    // 线程锁
    std::mutex mutex;
    
    bool m_bRunning = false;
private:
    Detector();

    rknn_matmul_ctx ctx;
    rknn_matmul_type matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    int B_layout = 0;
    int AC_layout = 0;
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    int iommu_domain_id = 0;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    
    void writeLog(const std::string& sMsg);
};
