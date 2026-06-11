#!/usr/bin/env python3
"""
analyze.py
──────────────────────────────────────────────────────────────────────────────
Analisis Performa Monte Carlo π — Buat Grafik Speedup & Efisiensi
Membaca: results/benchmark_results.csv
Output : results/grafik_speedup.png
         results/grafik_efisiensi.png
         results/grafik_waktu.png

Jalankan: python3 analyze.py

Tim     : Kelompok Monte Carlo — KPT 2024/2025
PIC     : Muhammad Aziz Arrahman
──────────────────────────────────────────────────────────────────────────────
"""

import csv
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("[ERROR] matplotlib tidak tersedia. Jalankan: pip install matplotlib")
    sys.exit(1)

# ── Konstanta ──────────────────────────────────────────────────────────────
PI_REF     = 3.14159265358979323846
CSV_FILE   = "results/benchmark_results.csv"
OUT_DIR    = "results"
COLORS     = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6", "#F39C12"]
MARKERS    = ["o", "s", "^", "D", "v"]

# ── Hukum Amdahl ────────────────────────────────────────────────────────────
def amdahl_speedup(p, f):
    """S(p) = 1 / (f + (1-f)/p)"""
    return 1.0 / (f + (1.0 - f) / p)

def estimate_serial_fraction(speedup_data):
    """
    Estimasi serial fraction f dari data speedup empiris
    menggunakan S(p) = 1/(f + (1-f)/p)  → f = (1/S - 1/p) / (1 - 1/p)
    Rata-ratakan dari semua p > 1
    """
    f_values = []
    for p, s in speedup_data.items():
        if p > 1 and s > 0:
            try:
                f = (1.0/s - 1.0/p) / (1.0 - 1.0/p)
                if 0 <= f <= 1:
                    f_values.append(f)
            except ZeroDivisionError:
                pass
    return sum(f_values) / len(f_values) if f_values else 0.1

# ── Baca CSV ────────────────────────────────────────────────────────────────
def load_data(filepath):
    """
    Return dict:
      data[N][T] = {'time': float, 'pi': float, 'error': float,
                    'speedup': float, 'efficiency': float}
    """
    data = {}
    if not os.path.exists(filepath):
        print(f"[ERROR] File tidak ditemukan: {filepath}")
        print("        Jalankan 'make benchmark' terlebih dahulu.")
        sys.exit(1)

    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            N   = int(row['N'])
            T   = int(row['threads'])
            if N not in data:
                data[N] = {}
            data[N][T] = {
                'time'      : float(row['time_sec']),
                'pi'        : float(row['pi_estimate']),
                'error'     : float(row['error_pct']),
                'speedup'   : float(row['speedup']),
                'efficiency': float(row['efficiency']),
            }
    return data

# ── Format label N ─────────────────────────────────────────────────────────
def fmt_N(N):
    if N >= 1_000_000:
        return f"{N//1_000_000}M"
    elif N >= 1_000:
        return f"{N//1_000}K"
    return str(N)

# ── Grafik 1: Speedup ───────────────────────────────────────────────────────
def plot_speedup(data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    N_list = sorted(data.keys())

    all_threads = sorted({T for N in data for T in data[N]})
    p_range = list(range(1, max(all_threads) + 1))

    # Garis ideal S=p
    ax.plot(all_threads, all_threads, 'k--', linewidth=1.5,
            label='Ideal (S = p)', zorder=1)

    for idx, N in enumerate(N_list):
        threads = sorted(data[N].keys())
        speedups = [data[N][T]['speedup'] for T in threads]

        # Estimasi serial fraction dan plot kurva Amdahl
        speedup_dict = {T: data[N][T]['speedup'] for T in threads}
        f = estimate_serial_fraction(speedup_dict)
        amdahl_vals = [amdahl_speedup(p, f) for p in p_range]

        color = COLORS[idx % len(COLORS)]
        ax.plot(threads, speedups, marker=MARKERS[idx],
                color=color, linewidth=2, markersize=8,
                label=f'Empiris N={fmt_N(N)} (f≈{f:.3f})', zorder=3)
        ax.plot(p_range, amdahl_vals, linestyle=':', color=color,
                linewidth=1.2, alpha=0.7,
                label=f'Amdahl N={fmt_N(N)}', zorder=2)

    ax.set_xlabel('Jumlah Thread (p)', fontsize=13)
    ax.set_ylabel('Speedup S(p) = T_seq / T_par', fontsize=13)
    ax.set_title('Speedup vs Jumlah Thread\n(Monte Carlo π — OpenMP)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max(all_threads) + 0.5)
    ax.set_ylim(0.8, max(all_threads) + 0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    path = os.path.join(out_dir, 'grafik_speedup.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Disimpan: {path}")
    plt.close()

# ── Grafik 2: Efisiensi ─────────────────────────────────────────────────────
def plot_efficiency(data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    N_list = sorted(data.keys())
    all_threads = sorted({T for N in data for T in data[N]})

    # Garis ideal E=1
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5,
               label='Ideal (E = 1.0)')

    for idx, N in enumerate(N_list):
        threads = sorted(data[N].keys())
        effs    = [data[N][T]['efficiency'] for T in threads]

        ax.plot(threads, effs, marker=MARKERS[idx],
                color=COLORS[idx % len(COLORS)], linewidth=2, markersize=8,
                label=f'N={fmt_N(N)}')

    ax.set_xlabel('Jumlah Thread (p)', fontsize=13)
    ax.set_ylabel('Efisiensi E(p) = S(p) / p', fontsize=13)
    ax.set_title('Efisiensi vs Jumlah Thread\n(Monte Carlo π — OpenMP)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max(all_threads) + 0.5)
    ax.set_ylim(0.0, 1.15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    path = os.path.join(out_dir, 'grafik_efisiensi.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Disimpan: {path}")
    plt.close()

# ── Grafik 3: Waktu Eksekusi ────────────────────────────────────────────────
def plot_time(data, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    N_list = sorted(data.keys())

    for idx, N in enumerate(N_list):
        threads = sorted(data[N].keys())
        times   = [data[N][T]['time'] for T in threads]

        ax.plot(threads, times, marker=MARKERS[idx],
                color=COLORS[idx % len(COLORS)], linewidth=2, markersize=8,
                label=f'N={fmt_N(N)}')

    ax.set_xlabel('Jumlah Thread (p)', fontsize=13)
    ax.set_ylabel('Waktu Eksekusi (detik)', fontsize=13)
    ax.set_title('Waktu Eksekusi vs Jumlah Thread\n(Monte Carlo π — OpenMP)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'grafik_waktu.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Disimpan: {path}")
    plt.close()

# ── Cetak Tabel Ringkasan ────────────────────────────────────────────────────
def print_summary(data):
    N_list = sorted(data.keys())
    print("\n" + "═"*72)
    print("  RINGKASAN HASIL BENCHMARK")
    print("═"*72)
    print(f"  {'N':>10}  {'T':>4}  {'Waktu(s)':>10}  {'Speedup':>8}  {'Efisiensi':>10}  {'π Estimasi':>12}")
    print("  " + "─"*68)
    for N in N_list:
        for T in sorted(data[N].keys()):
            d = data[N][T]
            print(f"  {fmt_N(N):>10}  {T:>4}  {d['time']:>10.4f}  "
                  f"{d['speedup']:>8.3f}  {d['efficiency']:>9.1%}  "
                  f"{d['pi']:>12.8f}")
        print("  " + "─"*68)

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("══════════════════════════════════════════════")
    print("  Analisis Performa Monte Carlo π (OpenMP)")
    print("══════════════════════════════════════════════")

    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_data(CSV_FILE)

    print_summary(data)

    print("\nMembuat grafik...")
    plot_speedup(data, OUT_DIR)
    plot_efficiency(data, OUT_DIR)
    plot_time(data, OUT_DIR)

    print("\n✓ Selesai! Semua grafik tersimpan di folder results/")
    print("  Gunakan grafik ini untuk laporan dan slide presentasi Minggu 4–5.")

if __name__ == "__main__":
    main()
