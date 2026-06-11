/*
 * monte_carlo_seq.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Monte Carlo Simulation untuk Estimasi Nilai π — Versi SEKUENSIAL (Baseline)
 *
 * Cara Kerja:
 *   Bangkitkan N titik acak (x, y) dalam kotak [0,1] × [0,1].
 *   Titik dianggap "dalam lingkaran" jika x² + y² ≤ 1.
 *   π ≈ 4 × (jumlah titik dalam lingkaran / N)
 *
 * Compile : gcc -O2 -o monte_carlo_seq monte_carlo_seq.c -lm
 * Jalankan: ./monte_carlo_seq 10000000
 *
 * Tim     : Kelompok Monte Carlo — KPT 2024/2025
 * PIC     : Gabriella Chantika Agnes Kristin
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ── Konstanta referensi ── */
#define PI_REF 3.14159265358979323846

/* ─────────────────────────────────────────────────────────────────────────────
 * monte_carlo_seq()
 *   Hitung estimasi π secara sekuensial.
 *
 * Parameter:
 *   N    — jumlah titik acak yang dibangkitkan (long long)
 *   seed — seed untuk PRNG (unsigned int)
 *
 * Return: estimasi nilai π (double)
 * ─────────────────────────────────────────────────────────────────────────────
 */
double monte_carlo_seq(long long N, unsigned int seed) {
    long long inside_count = 0;
    double x, y;

    /* Loop utama: bangkitkan titik acak, cek apakah di dalam lingkaran */
    for (long long i = 0; i < N; i++) {
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;

        /* Cek: apakah titik (x,y) berada di dalam lingkaran satuan? */
        if (x * x + y * y <= 1.0) {
            inside_count++;
        }
    }

    return 4.0 * (double)inside_count / (double)N;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * main()
 * ─────────────────────────────────────────────────────────────────────────────
 */
int main(int argc, char *argv[]) {
    /* Baca argumen: jumlah titik N */
    long long N = 10000000LL; /* default: 10 juta */
    if (argc >= 2) {
        N = atoll(argv[1]);
    }
    if (N <= 0) {
        fprintf(stderr, "[ERROR] N harus > 0\n");
        return 1;
    }

    unsigned int seed = 42; /* seed tetap agar reproducible */

    printf("========================================\n");
    printf("  Monte Carlo π — Sequential Baseline\n");
    printf("========================================\n");
    printf("  N (titik)    : %lld\n", N);
    printf("  Seed         : %u\n", seed);
    printf("----------------------------------------\n");

    /* Ukur waktu dengan clock() */
    clock_t t_start = clock();

    double pi_estimate = monte_carlo_seq(N, seed);

    clock_t t_end = clock();
    double elapsed_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;

    /* Hitung error relatif */
    double error = fabs(pi_estimate - PI_REF) / PI_REF * 100.0;

    printf("  π estimasi   : %.10f\n", pi_estimate);
    printf("  π referensi  : %.10f\n", PI_REF);
    printf("  Error        : %.6f %%\n", error);
    printf("  Waktu        : %.4f detik\n", elapsed_sec);
    printf("========================================\n");

    /* Output CSV untuk benchmark (ke stderr agar bisa di-pipe terpisah) */
    fprintf(stderr, "SEQ,%lld,1,%.10f,%.6f,%.4f\n",
            N, pi_estimate, error, elapsed_sec);

    return 0;
}
