#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <pthread.h>
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
    return duration<double, std::micro>(high_resolution_clock::now().time_since_epoch()).count();
}

static inline float& at(Matrix& A, int n, int i, int j) {
    return A[(size_t)i * n + j];
}

static inline const float& at_const(const Matrix& A, int n, int i, int j) {
    return A[(size_t)i * n + j];
}

/*
 * 沿用 PA2 的测试矩阵思路：先构造非奇异上三角矩阵，再做行叠加。
 * 这样可以避免随机矩阵在无主元选取高斯消去中出现 pivot 接近 0、inf 或 nan。
 */
void init_matrix(Matrix& A, int n) {
    A.assign((size_t)n * n, 0.0f);

    // 1) 上三角矩阵，主对角线为 1
    for (int i = 0; i < n; ++i) {
        at(A, n, i, i) = 1.0f;
        for (int j = i + 1; j < n; ++j) {
            at(A, n, i, j) = (float)((i * 13 + j * 7) % 10 + 1);
        }
    }

    // 2) 通过小系数行叠加构造普通矩阵，保持数值稳定，且初始化只需 O(n^2)
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
    for (int j = start; j < n; ++j) row[j] /= pivot;
}

static inline void elim_row_scalar(float* row_i, const float* row_k, int start, int n, float factor) {
    for (int j = start; j < n; ++j) row_i[j] -= factor * row_k[j];
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
    for (; j < n; ++j) row[j] /= pivot;
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
        vi = vmlsq_f32(vi, factor_v, vk);  // vi = vi - factor * vk
        vst1q_f32(row_i + j, vi);
    }
    for (; j < n; ++j) row_i[j] -= factor * row_k[j];
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

struct PthreadContext {
    Matrix* A;
    int n;
    int thread_count;
    bool use_neon;
    pthread_barrier_t barrier_division;
    pthread_barrier_t barrier_elimination;
};

struct ThreadParam {
    int tid;
    PthreadContext* ctx;
};

void* pthread_worker(void* arg) {
    ThreadParam* p = (ThreadParam*)arg;
    PthreadContext* ctx = p->ctx;
    Matrix& A = *(ctx->A);
    int n = ctx->n;
    int tid = p->tid;
    int T = ctx->thread_count;

    for (int k = 0; k < n; ++k) {
        float* row_k = &A[(size_t)k * n];

        // 第 k 行归一化。这里让 0 号线程完成，其他线程等待。
        if (tid == 0) {
            float pivot = row_k[k];
            if (ctx->use_neon) div_row_neon(row_k, k + 1, n, pivot);
            else div_row_scalar(row_k, k + 1, n, pivot);
            row_k[k] = 1.0f;
        }

        pthread_barrier_wait(&ctx->barrier_division);

        // 行划分：第 k 轮中 k+1 到 n-1 行互相独立，按循环分配给不同线程。
        for (int i = k + 1 + tid; i < n; i += T) {
            float* row_i = &A[(size_t)i * n];
            float factor = row_i[k];
            if (ctx->use_neon) elim_row_neon(row_i, row_k, k + 1, n, factor);
            else elim_row_scalar(row_i, row_k, k + 1, n, factor);
            row_i[k] = 0.0f;
        }

        pthread_barrier_wait(&ctx->barrier_elimination);
    }

    return nullptr;
}

void gauss_pthread(Matrix& A, int n, int thread_count, bool use_neon) {
    PthreadContext ctx;
    ctx.A = &A;
    ctx.n = n;
    ctx.thread_count = thread_count;
    ctx.use_neon = use_neon;

    pthread_barrier_init(&ctx.barrier_division, nullptr, thread_count);
    pthread_barrier_init(&ctx.barrier_elimination, nullptr, thread_count);

    std::vector<pthread_t> threads(thread_count);
    std::vector<ThreadParam> params(thread_count);

    for (int t = 0; t < thread_count; ++t) {
        params[t] = ThreadParam{t, &ctx};
        pthread_create(&threads[t], nullptr, pthread_worker, &params[t]);
    }
    for (int t = 0; t < thread_count; ++t) {
        pthread_join(threads[t], nullptr);
    }

    pthread_barrier_destroy(&ctx.barrier_division);
    pthread_barrier_destroy(&ctx.barrier_elimination);
}

void gauss_openmp(Matrix& A, int n, int thread_count, bool use_neon) {
    omp_set_num_threads(thread_count);

#pragma omp parallel
    {
        for (int k = 0; k < n; ++k) {
            float* row_k = &A[(size_t)k * n];

#pragma omp single
            {
                float pivot = row_k[k];
                if (use_neon) div_row_neon(row_k, k + 1, n, pivot);
                else div_row_scalar(row_k, k + 1, n, pivot);
                row_k[k] = 1.0f;
            }

#pragma omp for schedule(static)
            for (int i = k + 1; i < n; ++i) {
                float* row_i = &A[(size_t)i * n];
                float factor = row_i[k];
                if (use_neon) elim_row_neon(row_i, row_k, k + 1, n, factor);
                else elim_row_scalar(row_i, row_k, k + 1, n, factor);
                row_i[k] = 0.0f;
            }
            // omp for 末尾默认有 barrier，保证下一轮 k 开始前所有行消去完成
        }
    }
}

using AlgoFunc = void(*)(Matrix&, int);

template <typename Func>
double run_and_time(Func func, Matrix& A, int n) {
    double t0 = now_us();
    func(A, n);
    double t1 = now_us();
    return t1 - t0;
}

void print_header() {
    std::cout << "N,threads,serial_us,"
              << "pthread_scalar_us,pthread_scalar_speedup,pthread_scalar_err,"
              << "pthread_neon_us,pthread_neon_speedup,pthread_neon_err,"
              << "openmp_scalar_us,openmp_scalar_speedup,openmp_scalar_err,"
              << "openmp_neon_us,openmp_neon_speedup,openmp_neon_err\n";
}

void run_case(int n, const std::vector<int>& thread_list) {
    Matrix base;
    init_matrix(base, n);

    Matrix serial = base;
    double serial_us = run_and_time([](Matrix& A, int n){ gauss_serial(A, n); }, serial, n);

    for (int T : thread_list) {
        Matrix pth_s = base;
        double pth_s_us = run_and_time([&](Matrix& A, int n){ gauss_pthread(A, n, T, false); }, pth_s, n);
        double pth_s_err = max_error(serial, pth_s);

        Matrix pth_n = base;
        double pth_n_us = run_and_time([&](Matrix& A, int n){ gauss_pthread(A, n, T, true); }, pth_n, n);
        double pth_n_err = max_error(serial, pth_n);

        Matrix omp_s = base;
        double omp_s_us = run_and_time([&](Matrix& A, int n){ gauss_openmp(A, n, T, false); }, omp_s, n);
        double omp_s_err = max_error(serial, omp_s);

        Matrix omp_n = base;
        double omp_n_us = run_and_time([&](Matrix& A, int n){ gauss_openmp(A, n, T, true); }, omp_n, n);
        double omp_n_err = max_error(serial, omp_n);

        std::cout << std::fixed << std::setprecision(3)
                  << n << "," << T << "," << serial_us << ","
                  << pth_s_us << "," << serial_us / pth_s_us << "," << pth_s_err << ","
                  << pth_n_us << "," << serial_us / pth_n_us << "," << pth_n_err << ","
                  << omp_s_us << "," << serial_us / omp_s_us << "," << omp_s_err << ","
                  << omp_n_us << "," << serial_us / omp_n_us << "," << omp_n_err
                  << "\n";
    }
}

int main(int argc, char** argv) {
    std::vector<int> sizes = {256, 512, 1024};
    std::vector<int> threads = {1, 2, 4, 8};

    // 用法：
    //   ./main                 跑默认规模 256/512/1024，线程数 1/2/4/8
    //   ./main 512             只跑 512
    //   ./main 1024 1 2 4      跑 1024，线程数 1/2/4
    if (argc >= 2) {
        sizes = {std::stoi(argv[1])};
    }
    if (argc >= 3) {
        threads.clear();
        for (int i = 2; i < argc; ++i) {
            threads.push_back(std::stoi(argv[i]));
        }
    }

    std::cerr << "HAS_NEON = " << HAS_NEON << "\n";
    std::cerr << "Tip: redirect stdout to csv, e.g. ./main > result.csv\n";

    print_header();
    for (int n : sizes) {
        run_case(n, threads);
    }

    return 0;
}
