/*
 * test_correctness.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Unit Test — Verifikasi Kebenaran Hasil Paralel vs Sekuensial
 *
 * Test yang dijalankan:
 *   1. Smoke Test         — program tidak crash dengan input minimal
 *   2. Correctness Test   — |π_par - π_seq| < toleransi (0.01)
 *   3. Edge Case: N=1     — input ekstrem minimum
 *   4. Edge Case: N besar — N = 50 juta
 *   5. Determinism Test   — 2 run dengan T sama menghasilkan hasil mirip
 *   6. Multi-thread Test  — T=1,2,4,8 semua menghasilkan output yang benar
 *   7. Convergence Test   — π semakin akurat saat N naik
 *   8. Accuracy Test      — error < 0.1% untuk N = 10 juta
 *
 * Compile : gcc -O2 -fopenmp -o test_correctness test_correctness.c -lm
 * Jalankan: ./test_correctness
 *
 * Tim     : Kelompok Monte Carlo — KPT 2024/2025
 * PIC     : Nailah Nur Fadila
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI_REF      3.14159265358979323846
#define TOLERANCE   0.01    /* toleransi selisih paralel vs sequential */
#define ACC_THRESH  0.001   /* threshold error akurasi (0.1%) untuk N=10M */

/* ── Deklarasi fungsi ── */
double monte_carlo_parallel(long long N, int num_threads);

/* ── Warna terminal ── */
#define GREEN  "\033[0;32m"
#define RED    "\033[0;31m"
#define YELLOW "\033[0;33m"
#define RESET  "\033[0m"

/* ── Tracker hasil test ── */
static int pass_count = 0;
static int fail_count = 0;

/* ─────────────────────────────────────────────────────────────────────────────
 * assert_true() — helper assertion dengan pesan
 * ─────────────────────────────────────────────────────────────────────────────
 */
void assert_true(int condition, const char *test_name, const char *detail) {
    if (condition) {
        printf(GREEN "  [PASS]" RESET " %-45s %s\n", test_name, detail);
        pass_count++;
    } else {
        printf(RED   "  [FAIL]" RESET " %-45s %s\n", test_name, detail);
        fail_count++;
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 1: Smoke Test
 *   Program tidak crash dan mengembalikan nilai yang reasonable
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_smoke(void) {
    printf("\n" YELLOW "── Test 1: Smoke Test ──" RESET "\n");

    double pi = monte_carlo_parallel(1000, 1);
    assert_true(pi > 2.0 && pi < 5.0,
                "Output dalam rentang wajar (2.0–5.0)",
                "N=1000, T=1");

    pi = monte_carlo_parallel(1000, 4);
    assert_true(pi > 2.0 && pi < 5.0,
                "Output dalam rentang wajar (T=4)",
                "N=1000, T=4");
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 2: Correctness Test
 *   |π_parallel(T) - π_parallel(T=1)| < TOLERANCE
 *   Menggunakan T=1 sebagai proxy sequential
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_correctness(void) {
    printf("\n" YELLOW "── Test 2: Correctness Test ──" RESET "\n");

    long long N = 1000000LL; /* 1 juta */
    double pi_seq = monte_carlo_parallel(N, 1);

    int thread_configs[] = { 2, 4, 8 };
    char label[64];

    for (int i = 0; i < 3; i++) {
        int T = thread_configs[i];
        double pi_par = monte_carlo_parallel(N, T);
        double diff   = fabs(pi_par - pi_seq);

        snprintf(label, sizeof(label), "T=%d vs T=1: diff=%.6f < %.4f", T, diff, (double)TOLERANCE);
        assert_true(diff < TOLERANCE,
                    "Hasil paralel mendekati sequential",
                    label);
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 3: Edge Case — N = 1
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_edge_n1(void) {
    printf("\n" YELLOW "── Test 3: Edge Case N=1 ──" RESET "\n");

    double pi = monte_carlo_parallel(1, 1);
    /* Dengan N=1, hasilnya hanya 0.0 atau 4.0 */
    assert_true(pi == 0.0 || pi == 4.0,
                "N=1 menghasilkan 0.0 atau 4.0",
                "Nilai valid: titik di dalam atau di luar lingkaran");
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 4: Edge Case — N sangat besar
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_edge_large_n(void) {
    printf("\n" YELLOW "── Test 4: Edge Case N Besar (10 juta) ──" RESET "\n");

    double pi = monte_carlo_parallel(10000000LL, 4);
    double error = fabs(pi - PI_REF);

    char detail[64];
    snprintf(detail, sizeof(detail), "π=%.6f, err=%.6f", pi, error);

    assert_true(error < 0.05,
                "N=10 juta, T=4: error < 0.05",
                detail);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 5: Determinism Test
 *   Dua run berurutan harus menghasilkan hasil yang identik
 *   (karena seed deterministik berdasarkan thread ID)
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_determinism(void) {
    printf("\n" YELLOW "── Test 5: Determinism Test ──" RESET "\n");

    long long N = 500000LL;
    double pi1 = monte_carlo_parallel(N, 4);
    double pi2 = monte_carlo_parallel(N, 4);
    double diff = fabs(pi1 - pi2);

    char detail[64];
    snprintf(detail, sizeof(detail), "run1=%.8f, run2=%.8f, diff=%.10f", pi1, pi2, diff);
    assert_true(diff < 1e-9,
                "Dua run identik (seed deterministik)",
                detail);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 6: Multi-thread Correctness
 *   Semua konfigurasi thread menghasilkan π yang valid
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_multithread(void) {
    printf("\n" YELLOW "── Test 6: Multi-thread Test ──" RESET "\n");

    long long N       = 1000000LL;
    int threads[]     = { 1, 2, 4, 8 };
    char label[64];

    for (int i = 0; i < 4; i++) {
        int T = threads[i];
        double pi    = monte_carlo_parallel(N, T);
        double error = fabs(pi - PI_REF);

        snprintf(label, sizeof(label), "T=%d → π=%.6f, err=%.6f", T, pi, error);
        assert_true(error < 0.05,
                    "Error < 0.05 untuk semua thread config",
                    label);
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 7: Convergence Test
 *   Error harus menurun saat N naik
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_convergence(void) {
    printf("\n" YELLOW "── Test 7: Convergence Test ──" RESET "\n");

    long long N_vals[] = { 10000LL, 100000LL, 1000000LL, 10000000LL };
    int n_vals = 4;
    double prev_error = 999.0;
    int converging = 1;
    char detail[80];

    for (int i = 0; i < n_vals; i++) {
        double pi    = monte_carlo_parallel(N_vals[i], 4);
        double error = fabs(pi - PI_REF);
        snprintf(detail, sizeof(detail), "N=%7lld → error=%.6f", N_vals[i], error);
        printf("         %s\n", detail);

        /* Trend umum: error turun. Tidak harus strict karena Monte Carlo stochastic */
        if (error > prev_error * 3.0) { /* toleransi fluktuasi 3x */
            converging = 0;
        }
        prev_error = error;
    }

    assert_true(converging,
                "Error menurun seiring N bertambah",
                "Konvergensi Monte Carlo terkonfirmasi");
}

/* ─────────────────────────────────────────────────────────────────────────────
 * TEST 8: Accuracy Test
 *   Untuk N = 10 juta, error relatif harus < 0.1%
 * ─────────────────────────────────────────────────────────────────────────────
 */
void test_accuracy(void) {
    printf("\n" YELLOW "── Test 8: Accuracy Test ──" RESET "\n");

    long long N  = 10000000LL;
    double pi    = monte_carlo_parallel(N, 4);
    double error_rel = fabs(pi - PI_REF) / PI_REF;

    char detail[80];
    snprintf(detail, sizeof(detail), "π=%.8f, error_rel=%.6f%% (threshold=0.1%%)",
             pi, error_rel * 100.0);
    assert_true(error_rel < ACC_THRESH,
                "N=10 juta: error relatif < 0.1%",
                detail);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * main()
 * ─────────────────────────────────────────────────────────────────────────────
 */
int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   Unit Test — Monte Carlo π (OpenMP)                    ║\n");
    printf("║   KPT 2024/2025 — Kelompok Monte Carlo                  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("  Hardware threads: %d\n", omp_get_max_threads());

    test_smoke();
    test_correctness();
    test_edge_n1();
    test_edge_large_n();
    test_determinism();
    test_multithread();
    test_convergence();
    test_accuracy();

    /* ── Ringkasan hasil ── */
    int total = pass_count + fail_count;
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  Hasil: %d/%d PASS", pass_count, total);
    if (fail_count == 0) {
        printf(GREEN "  ✓ Semua test berhasil!\n" RESET);
    } else {
        printf(RED   "  ✗ %d test gagal — periksa output di atas\n" RESET, fail_count);
    }
    printf("══════════════════════════════════════════════════════════\n");

    return (fail_count == 0) ? 0 : 1;
}
