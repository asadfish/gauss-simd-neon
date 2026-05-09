#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <arm_neon.h>

const int N = 1024;
const int REPEAT = 3;

void init_matrix(float *a, int n) {
    srand(0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = 0.0f;
        }
    }

    for (int i = 0; i < n; i++) {
        a[i * n + i] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            a[i * n + j] = (float)(rand() % 10 + 1);
        }
    }

    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] += a[k * n + j];
            }
        }
    }
}

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

void gauss_neon(float *a, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = a[k * n + k];
        float32x4_t pivot_vec = vdupq_n_f32(pivot);

        int j = k + 1;

        for (; j + 4 <= n; j += 4) {
            float32x4_t row = vld1q_f32(&a[k * n + j]);
            row = vdivq_f32(row, pivot_vec);
            vst1q_f32(&a[k * n + j], row);
        }

        for (; j < n; j++) {
            a[k * n + j] = a[k * n + j] / pivot;
        }

        a[k * n + k] = 1.0f;

        for (int i = k + 1; i < n; i++) {
            float factor = a[i * n + k];
            float32x4_t factor_vec = vdupq_n_f32(factor);

            int j = k + 1;

            for (; j + 4 <= n; j += 4) {
                float32x4_t row_k = vld1q_f32(&a[k * n + j]);
                float32x4_t row_i = vld1q_f32(&a[i * n + j]);

                row_i = vmlsq_f32(row_i, factor_vec, row_k);

                vst1q_f32(&a[i * n + j], row_i);
            }

            for (; j < n; j++) {
                a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
            }

            a[i * n + k] = 0.0f;
        }
    }
}

float max_error(float *a, float *b, int n) {
    float ans = 0.0f;

    for (int i = 0; i < n * n; i++) {
        float now = std::fabs(a[i] - b[i]);
        if (now > ans) {
            ans = now;
        }
    }

    return ans;
}

double run_serial(const std::vector<float> &origin, std::vector<float> &a) {
    double total = 0.0;

    for (int r = 0; r < REPEAT; r++) {
        std::memcpy(a.data(), origin.data(), sizeof(float) * N * N);

        auto Start = std::chrono::high_resolution_clock::now();
        gauss_serial(a.data(), N);
        auto End = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::ratio<1, 1000000> > elapsed = End - Start;
        total += elapsed.count();
    }

    return total / REPEAT;
}

double run_neon(const std::vector<float> &origin, std::vector<float> &a) {
    double total = 0.0;

    for (int r = 0; r < REPEAT; r++) {
        std::memcpy(a.data(), origin.data(), sizeof(float) * N * N);

        auto Start = std::chrono::high_resolution_clock::now();
        gauss_neon(a.data(), N);
        auto End = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::ratio<1, 1000000> > elapsed = End - Start;
        total += elapsed.count();
    }

    return total / REPEAT;
}

int main(int argc, char *argv[]) {
    std::vector<float> origin(N * N);
    std::vector<float> serial_matrix(N * N);
    std::vector<float> neon_matrix(N * N);

    init_matrix(origin.data(), N);

    double serial_time = run_serial(origin, serial_matrix);
    double neon_time = run_neon(origin, neon_matrix);

    float err = max_error(serial_matrix.data(), neon_matrix.data(), N);
    double speedup = serial_time / neon_time;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "matrix size      : " << N << " x " << N << std::endl;
    std::cout << "serial latency   : " << serial_time << " (us)" << std::endl;
    std::cout << "neon latency     : " << neon_time << " (us)" << std::endl;
    std::cout << "speedup          : " << speedup << std::endl;
    std::cout << "max error        : " << err << std::endl;

    return 0;
}
