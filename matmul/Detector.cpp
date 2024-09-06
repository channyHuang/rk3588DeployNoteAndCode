#include "Detector.h"

#include "Common.hpp"

Detector* Detector::m_pInstance = nullptr;

Detector::Detector() {}
// 释放资源
Detector::~Detector() {
    if (m_pInstance != nullptr) {
        delete m_pInstance;
    }
    m_pInstance = nullptr;
}

bool Detector::init() {
    return true;
}

void Detector::deinit() {
    rknn_matmul_destroy(ctx);
}

// K // 32, N // 16
void Detector::setMatMulType(int nType) {
    matmul_type = (rknn_matmul_type)nType;
    printf("setMatMulType %d\n", nType);
}

template <typename Ti, typename To>
bool Detector::matrixMultiplyNpu(Ti* dataA, Ti* dataB, int M, int K, int N, To* dataC) {
    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = matmul_type;
    info.B_layout = B_layout;
    info.AC_layout = AC_layout;
    info.iommu_domain_id = iommu_domain_id;

    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_create fail! ret=%d\n", ret);
        return false;
    }

    ret = rknn_matmul_set_core_mask(ctx, (rknn_core_mask)core_mask);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_core_mask fail (only support rk3588/rk3576), ret=%d\n", ret);
    }

    // Create A
    rknn_tensor_mem *A = rknn_create_mem(ctx, io_attr.A.size);
    if (A == NULL) {
        fprintf(stderr, "rknn_create_mem fail!\n");
        return false;
    }
    // Create B
    rknn_tensor_mem *B = rknn_create_mem(ctx, io_attr.B.size);
    if (B == NULL) {
        fprintf(stderr, "rknn_create_mem fail!\n");
        return false;
    }
    // Create C
    rknn_tensor_mem *C = rknn_create_mem(ctx, io_attr.C.size);
    if (C == NULL) {
        fprintf(stderr, "rknn_create_mem fail!\n");
        return false;
    }

    memcpy(A->virt_addr, (Ti*)dataA, M * K * sizeof(Ti));
    memcpy(B->virt_addr, (Ti*)dataB, K * N * sizeof(Ti));

    // Set A
    ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
        return false;
    }
    // Set B
    ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
        return false;
    }
    // Set C
    ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
        return false;
    }
    int64_t start_us = getCurrentTimeUs();
    for (int i = 0; i < 1000; ++i) {
        ret = rknn_matmul_run(ctx);
    }
    int64_t elapse_us = getCurrentTimeUs() - start_us;
    printf("NPU Elapse Time = %.2fms, FPS = %.2f\n", elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);

    memcpy((To*)dataC, C->virt_addr, M * N * sizeof(To));

    // destroy
    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);

    return true;
}

template <typename Ti, typename To>
bool Detector::matrixMultiplyCheck(int M, int K, int N) {
    void *A_Matrix = nullptr;
    void *B_Matrix = nullptr;
    void *C_Matrix = nullptr;


    A_Matrix = malloc(M * K * sizeof(Ti));
    B_Matrix = malloc(K * N * sizeof(Ti));
    C_Matrix = malloc(M * N * sizeof(To));

    Ti *A_float16_Matrix = (Ti *)A_Matrix;
    Ti *B_float16_Matrix = (Ti *)B_Matrix;
    generate_random_buffer(A_float16_Matrix, M * K, {-10.f, 10.f});
    generate_random_buffer(B_float16_Matrix, K * N, {-10.f, 10.f});

    int64_t start_us = getCurrentTimeUs();
    bool res = matrixMultiplyNpu((Ti*)A_Matrix, (Ti*)B_Matrix, M, K, N, (To*)C_Matrix);
    
    int64_t elapse_us = getCurrentTimeUs() - start_us;
    printf("Elapse Time = %.2fms, FPS = %.2f\n", elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);

    std::vector<To> npu_res((To*)C_Matrix, (To*)C_Matrix + M * N);

    start_us = getCurrentTimeUs();
    std::vector<To> cpu_res;
    cpu_res.reserve(M * N);
    for (int i = 0; i < 1000; ++i) {
        cpu_res = matrixMultiply<Ti, To>((const Ti *)A_Matrix, (const Ti *)B_Matrix, M, K, N);
    }
    elapse_us = getCurrentTimeUs() - start_us;
    printf("CPU Elapse Time = %.2fms, FPS = %.2f\n", elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
    if (arraysCosineSimilarity<To>(cpu_res, npu_res))
    {
      printf(
          "%d matmul result is correct M x K x N is %d %d %d AC_layout is %d B_layout is %d\n", matmul_type,
          M, K, N, AC_layout, B_layout);
      ret = 0;
    }
    else
    {
      printf(
          "%d matmul result is wrong M x K x N is %d %d %d AC_layout is %d B_layout is %d\n", matmul_type,
          M, K, N, AC_layout, B_layout);
      ret = -1;
    }


    // clean data
    if (A_Matrix)
    {
        free(A_Matrix);
    }
    if (B_Matrix)
    {
        free(B_Matrix);
    }
    if (C_Matrix)
    {
        free(C_Matrix);
    }

    return true;
}

template 
D_SHARE_EXPORT bool Detector::matrixMultiplyCheck<float, float>(int M, int K, int N);
template
D_SHARE_EXPORT bool Detector::matrixMultiplyCheck<float16, float16>(int M, int K, int N);
template
D_SHARE_EXPORT bool Detector::matrixMultiplyCheck<float16, float>(int M, int K, int N);

// int-4bit32
template 
D_SHARE_EXPORT bool Detector::matrixMultiplyCheck<int, int>(int M, int K, int N);
template 
D_SHARE_EXPORT bool Detector::matrixMultiplyCheck<int8_t, int8_t>(int M, int K, int N);
template
D_SHARE_EXPORT bool Detector::matrixMultiplyCheck<int8_t, int32_t>(int M, int K, int N);