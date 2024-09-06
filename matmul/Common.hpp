#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <vector>
#include <sys/time.h>

static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

template <typename T>
void generate_random_buffer(T *buffer, size_t size, std::vector<float> range)
{
    if (buffer == nullptr || size == 0)
    {
        return;
    }
    // 设置随机种子
    srand((unsigned)time(NULL));

    float min = range[0], max = range[1];
    for (size_t i = 0; i < size; ++i)
    {
        buffer[i] = static_cast<T>(min + (max - min) * (static_cast<double>(rand()) / RAND_MAX));
    }
}

// 一维矩阵乘法函数
template <typename Ti, typename To>
std::vector<To> matrixMultiply(const Ti *A, const Ti *B, int M, int K, int N)
{
  std::vector<To> result(M * N, (To)0);

  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float sum = 0;
      for (int k = 0; k < K; ++k)
      {
#if DEBUG_PRINT
        printf("A[%d][%d] = %d, B[%d][%d] = %d, A*B = %6d\n", i, k, A[i * K + k], k, j, B[k * N + j],
               A[i * K + k] * B[k * N + j]);
#endif
        sum += (float)A[i * K + k] * (float)B[k * N + j];
      }
      result[i * N + j] = sum;
    }
  }

  return result;
}

template <typename T>
bool arraysCosineSimilarity(const std::vector<T> &arr1, const std::vector<T> &arr2, float eps = 0.9999f)
{
    if (arr1.size() != arr2.size())
    {
        return false;
    }

    // 计算点积
#pragma omp parallel for reduction(+ : dotProduct)
    double dotProduct = 0.0;
    for (size_t i = 0; i < arr1.size(); ++i)
    {
        dotProduct += arr1[i] * arr2[i];
    }

// 计算向量范数
#pragma omp parallel for reduction(+ : normA, normB)
    double normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < arr1.size(); ++i)
    {
        normA += std::pow(arr1[i], 2);
        normB += std::pow(arr2[i], 2);
    }

    // 避免除以零
    if (normA == 0.0 || normB == 0.0)
    {
        return false;
    }

    if ((dotProduct / (std::sqrt(normA) * std::sqrt(normB))) < eps)
    {
        return false;
    }

    return true;
}

#endif