#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

using Matrix = std::vector<float>;

static inline double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

static inline float& at(Matrix& A, int n, int i, int j) {
    return A[(size_t)i * n + j];
}

void init_matrix(Matrix& A, int n) {
    A.assign((size_t)n * n, 0.0f);

    for (int i = 0; i < n; ++i) {
        at(A, n, i, i) = 1.0f;
        for (int j = i + 1; j < n; ++j) {
            at(A, n, i, j) = (float)((i * 13 + j * 7) % 10 + 1);
        }
    }

    for (int i = 1; i < n; ++i) {
        float c = 0.01f * (float)((i % 5) + 1);
        for (int j = 0; j < n; ++j) {
            at(A, n, i, j) += c * at(A, n, i - 1, j);
        }
    }
}

double max_error(const Matrix& A, const Matrix& B) {
    double err = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        err = std::max(err, std::fabs((double)A[i] - (double)B[i]));
    }
    return err;
}

static inline void div_row_neon(float* row, int start, int n, float pivot) {
#if HAS_NEON
    int j = start;
    float32x4_t pivot_v = vdupq_n_f32(pivot);
    for (; j + 4 <= n; j += 4) {
        float32x4_t x = vld1q_f32(row + j);
        x = vdivq_f32(x, pivot_v);
        vst1q_f32(row + j, x);
    }
    for (; j < n; ++j) {
        row[j] /= pivot;
    }
#else
    for (int j = start; j < n; ++j) {
        row[j] /= pivot;
    }
#endif
}

static inline void elim_row_neon(float* row_i, const float* row_k, int start, int n, float factor) {
#if HAS_NEON
    int j = start;
    float32x4_t factor_v = vdupq_n_f32(factor);
    for (; j + 4 <= n; j += 4) {
        float32x4_t vi = vld1q_f32(row_i + j);
        float32x4_t vk = vld1q_f32(row_k + j);
        vi = vmlsq_f32(vi, factor_v, vk);
        vst1q_f32(row_i + j, vi);
    }
    for (; j < n; ++j) {
        row_i[j] -= factor * row_k[j];
    }
#else
    for (int j = start; j < n; ++j) {
        row_i[j] -= factor * row_k[j];
    }
#endif
}

void gauss_serial(Matrix& A, int n) {
    for (int k = 0; k < n; ++k) {
        float* row_k = &A[(size_t)k * n];
        float pivot = row_k[k];

        for (int j = k + 1; j < n; ++j) {
            row_k[j] /= pivot;
        }
        row_k[k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float* row_i = &A[(size_t)i * n];
            float factor = row_i[k];
            for (int j = k + 1; j < n; ++j) {
                row_i[j] -= factor * row_k[j];
            }
            row_i[k] = 0.0f;
        }
    }
}

void gauss_openmp_neon_schedule(Matrix& A, int n, int thread_count, omp_sched_t sched) {
    omp_set_num_threads(thread_count);
    omp_set_schedule(sched, 1);

#pragma omp parallel
    {
        for (int k = 0; k < n; ++k) {
            float* row_k = &A[(size_t)k * n];

#pragma omp single
            {
                float pivot = row_k[k];
                div_row_neon(row_k, k + 1, n, pivot);
                row_k[k] = 1.0f;
            }

#pragma omp for schedule(runtime)
            for (int i = k + 1; i < n; ++i) {
                float* row_i = &A[(size_t)i * n];
                float factor = row_i[k];
                elim_row_neon(row_i, row_k, k + 1, n, factor);
                row_i[k] = 0.0f;
            }
        }
    }
}

template <typename Func>
double run_time(Func f, Matrix& A, int n) {
    double t0 = now_us();
    f(A, n);
    double t1 = now_us();
    return t1 - t0;
}

void run_case(int n, int threads) {
    Matrix base;
    init_matrix(base, n);

    Matrix serial = base;
    double serial_us = run_time([](Matrix& A, int n) {
        gauss_serial(A, n);
    }, serial, n);

    struct Item {
        const char* name;
        omp_sched_t sched;
    };

    Item items[] = {
        {"static", omp_sched_static},
        {"dynamic", omp_sched_dynamic},
        {"guided", omp_sched_guided}
    };

    for (auto item : items) {
        Matrix test = base;
        double t_us = run_time([&](Matrix& A, int n) {
            gauss_openmp_neon_schedule(A, n, threads, item.sched);
        }, test, n);

        double err = max_error(serial, test);

        std::cout << n << ","
                  << threads << ","
                  << item.name << ","
                  << std::fixed << std::setprecision(3)
                  << serial_us << ","
                  << t_us << ","
                  << serial_us / t_us << ","
                  << err << "\n";
    }
}

int main() {
    std::cerr << "HAS_NEON = " << HAS_NEON << "\n";

    std::cout << "N,threads,schedule,serial_us,openmp_neon_us,speedup,max_error\n";

    int sizes[] = {512, 1024, 1500};
    int thread_list[] = {4, 8};

    for (int n : sizes) {
        for (int t : thread_list) {
            run_case(n, t);
        }
    }

    return 0;
}