/*
 * monte_carlo_par.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Monte Carlo Simulation untuk Estimasi Nilai π — Versi PARALEL (OpenMP)
 *
 * Strategi Paralelisme:
 *   - Pola   : Domain Decomposition (bagi iterasi loop ke tiap thread)
 *   - Sync   : OpenMP reduction(+:inside_count) — aman, tanpa race condition
 *   - PRNG   : rand_r(&seed) dengan seed PRIVATE per thread — thread-safe
 *   - Timer  : omp_get_wtime() — wall-clock presisi tinggi
 *
 * Compile : gcc -O2 -fopenmp -o monte_carlo_par monte_carlo_par.c -lm
 * Jalankan: ./monte_carlo_par 10000000 4
 *             ↑ N=10 juta, T=4 thread
 *
 * Tim     : Kelompok Monte Carlo — KPT 2024/2025
 * PIC     : Nailah Nur Fadila
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* ── Konstanta referensi ── */
#define PI_REF 3.14159265358979323846


// Fungsi LCG sederhana sebagai pengganti rand_r di Windows
double win_rand_r(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (double)(*seed / 65536 % 32768) / 32767.0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * monte_carlo_parallel()
 *   Hitung estimasi π secara paralel menggunakan OpenMP.
 *
 * Parameter:
 *   N          — jumlah titik acak total (long long)
 *   num_threads — jumlah thread OpenMP yang digunakan (int)
 *
 * Return: estimasi nilai π (double)
 *
 * Catatan Desain:
 *   - `inside_count` di-share tapi dilindungi oleh reduction — tidak perlu mutex
 *   - `x`, `y`, `seed` bersifat PRIVATE — setiap thread punya salinan sendiri
 *   - Seed unik per thread: (thread_id + 1) * 12345
 *     → mencegah setiap thread membangkitkan urutan angka yang sama
 * ─────────────────────────────────────────────────────────────────────────────
 */
double monte_carlo_parallel(long long N, int num_threads) {
    long long inside_count = 0;

    /* ── Blok Paralel Utama ─────────────────────────────────────────────── */
    #pragma omp parallel num_threads(num_threads) \
            reduction(+: inside_count)
    {
        /* Variabel privat: setiap thread punya salinan masing-masing */
        double x, y;
        unsigned int seed = (unsigned int)((omp_get_thread_num() + 1) * 12345);
        long long local_count = 0;

        /* Bagi iterasi secara merata ke semua thread (schedule static default) */
        #pragma omp for schedule(static)
        for (long long i = 0; i < N; i++) {
            x = win_rand_r(&seed);
            y = win_rand_r(&seed);

            if (x * x + y * y <= 1.0) {
                local_count++;
            }
        }

        /* Setiap thread menambahkan hasil lokalnya ke inside_count (via reduction) */
        inside_count += local_count;

    } /* ── implicit barrier: master menunggu semua thread selesai ── */

    return 4.0 * (double)inside_count / (double)N;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * main()
 * ─────────────────────────────────────────────────────────────────────────────
 */
int main_backup(int argc, char *argv[]) {
    /* Baca argumen */
    long long N       = 10000000LL; /* default: 10 juta */
    int num_threads   = 4;           /* default: 4 thread */

    if (argc >= 2) N           = atoll(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);

    if (N <= 0) {
        fprintf(stderr, "[ERROR] N harus > 0\n");
        return 1;
    }
    if (num_threads <= 0 || num_threads > 64) {
        fprintf(stderr, "[ERROR] Jumlah thread harus antara 1–64\n");
        return 1;
    }

    printf("========================================\n");
    printf("   Monte Carlo π — Parallel (OpenMP)\n");
    printf("========================================\n");
    printf("  N (titik)    : %lld\n", N);
    printf("  Thread       : %d\n", num_threads);
    printf("  Max thread   : %d (hardware)\n", omp_get_max_threads());
    printf("----------------------------------------\n");

    /* Ukur waktu wall-clock */
    double t_start = omp_get_wtime();

    double pi_estimate = monte_carlo_parallel(N, num_threads);

    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

    double error = fabs(pi_estimate - PI_REF) / PI_REF * 100.0;

    printf("  π estimasi   : %.10f\n", pi_estimate);
    printf("  π referensi  : %.10f\n", PI_REF);
    printf("  Error        : %.6f %%\n", error);
    printf("  Waktu        : %.4f detik\n", elapsed);
    printf("========================================\n");

    /* Output CSV untuk benchmark script */
    fprintf(stderr, "PAR,%lld,%d,%.10f,%.6f,%.4f\n",
            N, num_threads, pi_estimate, error, elapsed);

    return 0;
}
