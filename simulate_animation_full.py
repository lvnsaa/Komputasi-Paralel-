#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# ── Konfigurasi Awal ─────────────────────────────────────────────────────────
N_TOTAL = 3000  # Default 3000 titik agar lingkaran penuh terlihat padat
if len(sys.argv) > 1:
    try:
        N_TOTAL = int(sys.argv[1])
    except ValueError:
        print("[WARN] Angka tidak valid, menggunakan default 3000 titik.")

BATCH_SIZE = max(1, N_TOTAL // 100)  # Jumlah titik per frame animasi

# ── Inisialisasi Data Acak (Rentang -1 sampai 1) ──────────────────────────────
np.random.seed(42)
# Mengubah jangkauan acak menjadi [-1, 1] agar mencakup seluruh lingkaran
x_data = np.random.uniform(-1, 1, N_TOTAL)
y_data = np.random.uniform(-1, 1, N_TOTAL)

# Titik dianggap di dalam lingkaran jika x² + y² <= 1
inside_mask = (x_data**2 + y_data**2) <= 1.0

# ── Setup Figure dan Area Grafik ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 7.5))
fig.canvas.manager.set_window_title('Simulasi Monte Carlo Lingkaran Utuh')

# Menggambar batas lingkaran utuh (0 sampai 2*pi)
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), color='#2C3E50', lw=2.5, label='Batas Lingkaran ($x^2 + y^2 = 1$)')

# Siapkan plot kosong untuk titik dalam (hijau) dan luar (merah)
scat_inside = ax.scatter([], [], color='#2ECC71', s=8, alpha=0.8, label='Di Dalam Lingkaran')
scat_outside = ax.scatter([], [], color='#E74C3C', s=8, alpha=0.8, label='Di Luar Lingkaran')

# Mengatur batas grafik dari -1.1 sampai 1.1 agar kotak luar terlihat jelas
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

ax.set_title('Simulasi Real-time Monte Carlo $\pi$ (Lingkaran Utuh)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Sumbu X', fontsize=10)
ax.set_ylabel('Sumbu Y', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper right')

# Teks statistik real-time di pojok kiri bawah
text_stats = ax.text(-1.03, -1.03, '', fontsize=10, fontweight='bold', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# ── Fungsi Update Animasi ────────────────────────────────────────────────────
def update(frame):
    current_count = (frame + 1) * BATCH_SIZE
    if current_count > N_TOTAL:
        current_count = N_TOTAL
        
    x_curr = x_data[:current_count]
    y_curr = y_data[:current_count]
    inside_curr = inside_mask[:current_count]
    
    x_in, y_in = x_curr[inside_curr], y_curr[inside_curr]
    x_out, y_out = x_curr[~inside_curr], y_curr[~inside_curr]
    
    scat_inside.set_offsets(np.c_[x_in, y_in])
    scat_outside.set_offsets(np.c_[x_out, y_out])
    
    n_inside = len(x_in)
    
    # RUMUS BARU: Karena luas kotak sekarang 2x2 = 4, dan luas lingkaran adalah pi * r² (r=1 -> luas = pi)
    # Maka Rasio = Luas Lingkaran / Luas Kotak -> (n_inside / current_count) = pi / 4
    # Sehingga pi = 4 * (n_inside / current_count)
    pi_estimate = 4.0 * n_inside / current_count if current_count > 0 else 0
    error = abs(pi_estimate - np.pi) / np.pi * 100
    
    text_stats.set_text(
        f"Total Titik (N) : {current_count:,} / {N_TOTAL:,}\n"
        f"Dalam Lingkaran: {n_inside:,}\n"
        f"Estimasi $\pi$  : {pi_estimate:.6f}\n"
        f"Error           : {error:.4f}%"
    )
    
    return scat_inside, scat_outside, text_stats

# ── Eksekusi Animasi ─────────────────────────────────────────────────────────
frames = int(np.ceil(N_TOTAL / BATCH_SIZE))
ani = animation.FuncAnimation(
    fig, update, frames=frames, interval=30, blit=True, repeat=False
)

plt.tight_layout()
plt.show()