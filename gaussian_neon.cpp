#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <arm_neon.h>

using namespace std;

#define IDX(i, j, n) ((i) * (n) + (j))

// 生成测试矩阵，尽量避免主元为 0 或出现 NaN
void init_matrix(vector<float>& A, int n) {
    srand(0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[IDX(i, j, n)] = 0.0f;
        }
    }

    // 先构造一个上三角矩阵，对角线为 1
    for (int i = 0; i < n; i++) {
        A[IDX(i, i, n)] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            A[IDX(i, j, n)] = (float)(rand() % 100 + 1) / 10.0f;
        }
    }

    // 让下面的行叠加上面的行，生成普通稠密矩阵
    // 这种方式可以减少主元为 0 的风险
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[IDX(i, j, n)] += A[IDX(k, j, n)];
            }
        }
    }
}

// 普通 for 循环串行版本
void gaussian_serial(vector<float>& A, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = A[IDX(k, k, n)];

        for (int j = k + 1; j < n; j++) {
            A[IDX(k, j, n)] = A[IDX(k, j, n)] / pivot;
        }

        A[IDX(k, k, n)] = 1.0f;

        for (int i = k + 1; i < n; i++) {
            float factor = A[IDX(i, k, n)];

            for (int j = k + 1; j < n; j++) {
                A[IDX(i, j, n)] = A[IDX(i, j, n)] - factor * A[IDX(k, j, n)];
            }

            A[IDX(i, k, n)] = 0.0f;
        }
    }
}

// ARM NEON 版本：第 k 行归一化 + 行消去都向量化
void gaussian_neon(vector<float>& A, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = A[IDX(k, k, n)];

        // =========================
        // 1. 第 k 行归一化 NEON
        // A[k][j] = A[k][j] / pivot
        // =========================
        float32x4_t pivot_vec = vdupq_n_f32(pivot);

        int j = k + 1;

        for (; j + 4 <= n; j += 4) {
            float32x4_t row_k = vld1q_f32(&A[IDX(k, j, n)]);
            row_k = vdivq_f32(row_k, pivot_vec);
            vst1q_f32(&A[IDX(k, j, n)], row_k);
        }

        // 处理剩余不足 4 个的元素
        for (; j < n; j++) {
            A[IDX(k, j, n)] = A[IDX(k, j, n)] / pivot;
        }

        A[IDX(k, k, n)] = 1.0f;

        // =========================
        // 2. 行消去 NEON
        // A[i][j] = A[i][j] - A[i][k] * A[k][j]
        // =========================
        for (int i = k + 1; i < n; i++) {
            float factor = A[IDX(i, k, n)];
            float32x4_t factor_vec = vdupq_n_f32(factor);

            j = k + 1;

            for (; j + 4 <= n; j += 4) {
                float32x4_t row_k = vld1q_f32(&A[IDX(k, j, n)]);
                float32x4_t row_i = vld1q_f32(&A[IDX(i, j, n)]);

                // row_i = row_i - factor_vec * row_k
                // 注意这里是 vmlsq_f32，不是 vmlaq_f32
                row_i = vmlsq_f32(row_i, factor_vec, row_k);

                vst1q_f32(&A[IDX(i, j, n)], row_i);
            }

            // 处理剩余不足 4 个的元素
            for (; j < n; j++) {
                A[IDX(i, j, n)] = A[IDX(i, j, n)] - factor * A[IDX(k, j, n)];
            }

            A[IDX(i, k, n)] = 0.0f;
        }
    }
}

// 计算两个矩阵之间的最大绝对误差
float max_abs_error(const vector<float>& A, const vector<float>& B, int n) {
    float max_error = 0.0f;

    for (int i = 0; i < n * n; i++) {
        float error = fabs(A[i] - B[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}

int main() {
    int n;
    cout << "Input matrix size n: ";
    cin >> n;

    vector<float> A_init(n * n);
    vector<float> A_serial(n * n);
    vector<float> A_neon(n * n);

    init_matrix(A_init, n);

    A_serial = A_init;
    A_neon = A_init;

    clock_t start_serial = clock();
    gaussian_serial(A_serial, n);
    clock_t end_serial = clock();

    clock_t start_neon = clock();
    gaussian_neon(A_neon, n);
    clock_t end_neon = clock();

    double serial_time = (double)(end_serial - start_serial) / CLOCKS_PER_SEC;
    double neon_time = (double)(end_neon - start_neon) / CLOCKS_PER_SEC;

    float error = max_abs_error(A_serial, A_neon, n);

    cout << endl;
    cout << "========== Result ==========" << endl;
    cout << "Matrix size n       : " << n << endl;
    cout << "Serial time         : " << serial_time << " s" << endl;
    cout << "NEON time           : " << neon_time << " s" << endl;

    if (neon_time > 0) {
        cout << "Speedup             : " << serial_time / neon_time << endl;
    }

    cout << "Max absolute error  : " << error << endl;

    if (error < 1e-4) {
        cout << "Correctness check   : PASS" << endl;
    } else {
        cout << "Correctness check   : FAIL" << endl;
    }

    return 0;
}