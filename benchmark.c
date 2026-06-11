/*
 * benchmark.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Benchmark otomatis: menjalankan semua konfigurasi (N × T) dan
 * menyimpan hasil ke file CSV untuk analisis Python.
 *
 * Compile : gcc -O2 -fopenmp -o benchmark benchmark.c -lm
 * Jalankan: ./benchmark
 *   → Output: results/benchmark_results.csv
 *
 * Tim     : Kelompok Monte Carlo — KPT 2024/2025
 * PIC     : Gabriella Chantika Agnes Kristin
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/stat.h>

#define PI_REF      3.14159265358979323846
#define N_RUNS      5       /* jumlah repetisi per konfigurasi (ambil median) */
#define MAX_RUNS    10

/* ── Deklarasi fungsi dari modul lain ── */
double monte_carlo_parallel(long long N, int num_threads);

/* ─────────────────────────────────────────────────────────────────────────────
 * compare_double() — helper untuk qsort
 * ─────────────────────────────────────────────────────────────────────────────
 */
int compare_double(const void *a, const void *b) {
    double da = *(double*)a, db = *(double*)b;
    return (da > db) - (da < db);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * median() — ambil median dari array
 * ─────────────────────────────────────────────────────────────────────────────
 */
double median(double arr[], int n) {
    double temp[MAX_RUNS];
    for (int i = 0; i < n; i++) temp[i] = arr[i];
    qsort(temp, n, sizeof(double), compare_double);
    return (n % 2 == 0) ? (temp[n/2-1] + temp[n/2]) / 2.0 : temp[n/2];
}

/* ─────────────────────────────────────────────────────────────────────────────
 * run_benchmark_config()
 *   Jalankan satu konfigurasi (N, T) sebanyak N_RUNS kali, return median waktu
 * ─────────────────────────────────────────────────────────────────────────────
 */
double run_benchmark_config(long long N, int T, double *pi_out) {
    double times[MAX_RUNS], pi_vals[MAX_RUNS];

    for (int r = 0; r < N_RUNS; r++) {
        double t0 = omp_get_wtime();
        pi_vals[r] = monte_carlo_parallel(N, T);
        times[r]   = omp_get_wtime() - t0;
    }

    *pi_out = median(pi_vals, N_RUNS);
    return median(times, N_RUNS);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * main()
 * ─────────────────────────────────────────────────────────────────────────────
 */
int main(void) {
    /* ── Konfigurasi eksperimen ── */
    long long N_list[]  = { 1000000LL, 10000000LL, 100000000LL };
    int       T_list[]  = { 1, 2, 4, 8 };
    int       n_N       = 3;
    int       n_T       = 4;

    int hw_threads = omp_get_max_threads();
    printf("Hardware threads tersedia: %d\n", hw_threads);
    printf("Menjalankan %d konfigurasi × %d run = %d total eksekusi\n\n",
           n_N * n_T, N_RUNS, n_N * n_T * N_RUNS);

    /* ── Buat folder results ── */
    mkdir("results");

    /* ── Buka file CSV output ── */
    FILE *fp = fopen("results/benchmark_results.csv", "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] Gagal membuka file output CSV\n");
        return 1;
    }
    fprintf(fp, "mode,N,threads,pi_estimate,error_pct,time_sec,speedup,efficiency\n");

    /* ── Pertama: ukur waktu sequential (T=1) untuk semua N ── */
    double t_seq[3];
    printf("%-12s %-12s %-6s %-14s %-12s %-10s\n",
           "Mode", "N", "T", "π estimasi", "Error(%)", "Waktu(s)");
    printf("─────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < n_N; i++) {
        double pi_val;
        t_seq[i] = run_benchmark_config(N_list[i], 1, &pi_val);
        double err = fabs(pi_val - PI_REF) / PI_REF * 100.0;
        printf("%-12s %-12lld %-6d %-14.8f %-12.6f %-10.4f\n",
               "SEQ", N_list[i], 1, pi_val, err, t_seq[i]);
        fprintf(fp, "SEQ,%lld,1,%.10f,%.6f,%.4f,1.000000,1.000000\n",
                N_list[i], pi_val, err, t_seq[i]);
    }

    printf("\n");

    /* ── Kemudian: semua konfigurasi paralel ── */
    for (int i = 0; i < n_N; i++) {
        for (int j = 0; j < n_T; j++) {
            if (T_list[j] == 1) continue; /* T=1 sudah dihitung di atas */

            double pi_val;
            double t_par = run_benchmark_config(N_list[i], T_list[j], &pi_val);
            double err   = fabs(pi_val - PI_REF) / PI_REF * 100.0;
            double speedup    = t_seq[i] / t_par;
            double efficiency = speedup / T_list[j];

            printf("%-12s %-12lld %-6d %-14.8f %-12.6f %-10.4f  S=%.2f  E=%.2f\n",
                   "PAR", N_list[i], T_list[j], pi_val, err, t_par,
                   speedup, efficiency);

            fprintf(fp, "PAR,%lld,%d,%.10f,%.6f,%.4f,%.6f,%.6f\n",
                    N_list[i], T_list[j], pi_val, err, t_par,
                    speedup, efficiency);
        }
        printf("\n");
    }

    fclose(fp);
    printf("\n✓ Hasil disimpan ke: results/benchmark_results.csv\n");
    printf("  Jalankan: python3 analyze.py  untuk membuat grafik\n");
    return 0;
}
