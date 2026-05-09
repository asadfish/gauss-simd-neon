#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <arm_neon.h>

// 矩阵规模：可以根据运行时间调整
// 如果运行太慢，可以改成 512；如果想让实验更明显，可以改成 1024
const int N = 768;
const int REPEAT = 3;

// 生成测试矩阵，避免主元为 0 或出现 NaN
void init_matrix(float *a, int n) {
    srand(0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = 0.0f;
        }
    }

    // 先生成一个上三角矩阵
    for (int i = 0; i < n; i++) {
        a[i * n + i] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            a[i * n + j] = (float)(rand() % 10 + 1);
        }
    }

    // 通过行叠加生成普通矩阵，保证高斯消去过程中不容易出问题
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] += a[k * n + j];
            }
        }
    }
}

// 串行高斯消去，用于实验报告中对比，也可以检查正确性
void gauss_serial(float *a, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = a[k * n + k];

        for (int j = k + 1; j < n; j++) {
            a[k * n + j] = a[k * n + j] / pivot;
        }
        a[k * n + k] = 1.0f;

        for (int i = k + 1; i < n; i++) {
            float factor = a[i * n + k];

            for (int j = k + 1; j < n; j++) {
                a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
            }

            a[i * n + k] = 0.0f;
        }
    }
}

// ARM NEON SIMD 高斯消去
void gauss_neon(float *a, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = a[k * n + k];
        float32x4_t pivot_vec = vdupq_n_f32(pivot);

        int j = k + 1;

        // 第 k 行除法向量化：一次处理 4 个 float
        for (; j + 4 <= n; j += 4) {
            float32x4_t row = vld1q_f32(&a[k * n + j]);
            row = vdivq_f32(row, pivot_vec);
            vst1q_f32(&a[k * n + j], row);
        }

        // 处理剩余不足 4 个的部分
        for (; j < n; j++) {
            a[k * n + j] = a[k * n + j] / pivot;
        }

        a[k * n + k] = 1.0f;

        // 用第 k 行消去下面的行
        for (int i = k + 1; i < n; i++) {
            float factor = a[i * n + k];
            float32x4_t factor_vec = vdupq_n_f32(factor);

            int j = k + 1;

            // 消去过程向量化：A[i][j] = A[i][j] - factor * A[k][j]
            for (; j + 4 <= n; j += 4) {
                float32x4_t row_k = vld1q_f32(&a[k * n + j]);
                float32x4_t row_i = vld1q_f32(&a[i * n + j]);

                row_i = vmlsq_f32(row_i, factor_vec, row_k);

                vst1q_f32(&a[i * n + j], row_i);
            }

            // 处理剩余不足 4 个的部分
            for (; j < n; j++) {
                a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
            }

            a[i * n + k] = 0.0f;
        }
    }
}

// 计算两个矩阵的最大误差，调试时可用
float max_error(float *a, float *b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float now = std::fabs(a[i] - b[i]);
        if (now > err) err = now;
    }
    return err;
}

int main(int argc, char *argv[]) {
    std::vector<float> origin(N * N);
    std::vector<float> matrix(N * N);

    init_matrix(origin.data(), N);

    double total_time = 0.0;

    for (int r = 0; r < REPEAT; r++) {
        std::memcpy(matrix.data(), origin.data(), sizeof(float) * N * N);

        auto Start = std::chrono::high_resolution_clock::now();

        gauss_neon(matrix.data(), N);

        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000000> > elapsed = End - Start;

        total_time += elapsed.count();
    }

    double avg_time = total_time / REPEAT;

    std::cout << "average latency  : " << avg_time << " (us) " << std::endl;

    return 0;
}