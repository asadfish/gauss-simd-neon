#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

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

static inline void div_row_scalar(float* row, int start, int n, float pivot) {
    for (int j = start; j < n; ++j) {
        row[j] /= pivot;
    }
}

static inline void elim_row_scalar(float* row_i, const float* row_k, int start, int n, float factor) {
    for (int j = start; j < n; ++j) {
        row_i[j] -= factor * row_k[j];
    }
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
    div_row_scalar(row, start, n, pivot);
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
    elim_row_scalar(row_i, row_k, start, n, factor);
#endif
}

void gauss_serial(Matrix& A, int n) {
    for (int k = 0; k < n; ++k) {
        float* row_k = &A[(size_t)k * n];
        float pivot = row_k[k];

        div_row_scalar(row_k, k + 1, n, pivot);
        row_k[k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float* row_i = &A[(size_t)i * n];
            float factor = row_i[k];
            elim_row_scalar(row_i, row_k, k + 1, n, factor);
            row_i[k] = 0.0f;
        }
    }
}

class SimpleBarrier {
private:
    std::mutex mtx;
    std::condition_variable cv;
    int total;
    int arrived;
    int generation;

public:
    explicit SimpleBarrier(int count) : total(count), arrived(0), generation(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;

        arrived++;
        if (arrived == total) {
            arrived = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&]() {
                return gen != generation;
            });
        }
    }
};

void std_worker(Matrix& A, int n, int tid, int thread_count, bool use_neon,
                SimpleBarrier& barrier_division, SimpleBarrier& barrier_elimination) {
    for (int k = 0; k < n; ++k) {
        float* row_k = &A[(size_t)k * n];

        if (tid == 0) {
            float pivot = row_k[k];
            if (use_neon) {
                div_row_neon(row_k, k + 1, n, pivot);
            } else {
                div_row_scalar(row_k, k + 1, n, pivot);
            }
            row_k[k] = 1.0f;
        }

        barrier_division.wait();

        for (int i = k + 1 + tid; i < n; i += thread_count) {
            float* row_i = &A[(size_t)i * n];
            float factor = row_i[k];

            if (use_neon) {
                elim_row_neon(row_i, row_k, k + 1, n, factor);
            } else {
                elim_row_scalar(row_i, row_k, k + 1, n, factor);
            }

            row_i[k] = 0.0f;
        }

        barrier_elimination.wait();
    }
}

void gauss_std_thread(Matrix& A, int n, int thread_count, bool use_neon) {
    SimpleBarrier barrier_division(thread_count);
    SimpleBarrier barrier_elimination(thread_count);

    std::vector<std::thread> workers;

    for (int t = 0; t < thread_count; ++t) {
        workers.emplace_back(std_worker,
                             std::ref(A),
                             n,
                             t,
                             thread_count,
                             use_neon,
                             std::ref(barrier_division),
                             std::ref(barrier_elimination));
    }

    for (auto& th : workers) {
        th.join();
    }
}

template <typename Func>
double run_time(Func func, Matrix& A, int n) {
    double t0 = now_us();
    func(A, n);
    double t1 = now_us();
    return t1 - t0;
}

void run_case(int n, const std::vector<int>& thread_list) {
    Matrix base;
    init_matrix(base, n);

    Matrix serial = base;
    double serial_us = run_time([](Matrix& A, int n) {
        gauss_serial(A, n);
    }, serial, n);

    for (int T : thread_list) {
        Matrix std_scalar = base;
        double std_scalar_us = run_time([&](Matrix& A, int n) {
            gauss_std_thread(A, n, T, false);
        }, std_scalar, n);
        double std_scalar_err = max_error(serial, std_scalar);

        Matrix std_neon = base;
        double std_neon_us = run_time([&](Matrix& A, int n) {
            gauss_std_thread(A, n, T, true);
        }, std_neon, n);
        double std_neon_err = max_error(serial, std_neon);

        std::cout << n << ","
                  << T << ","
                  << std::fixed << std::setprecision(3)
                  << serial_us << ","
                  << std_scalar_us << ","
                  << serial_us / std_scalar_us << ","
                  << std_scalar_err << ","
                  << std_neon_us << ","
                  << serial_us / std_neon_us << ","
                  << std_neon_err
                  << "\n";
    }
}

int main() {
    std::cerr << "HAS_NEON = " << HAS_NEON << "\n";

    std::cout << "N,threads,serial_us,"
              << "std_thread_scalar_us,std_thread_scalar_speedup,std_thread_scalar_err,"
              << "std_thread_neon_us,std_thread_neon_speedup,std_thread_neon_err\n";

    std::vector<int> sizes = {512, 1024, 1500};
    std::vector<int> threads = {1, 2, 4, 8};

    for (int n : sizes) {
        run_case(n, threads);
    }

    return 0;
}