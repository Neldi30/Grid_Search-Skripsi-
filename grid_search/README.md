# Grid Search α, β, γ — T_dynamic Drowsiness Detection

Pipeline Python untuk mencari nilai optimal parameter `α, β, γ` di rumus
`T_dynamic = T_BASE − (α·|yaw| + β·|pitch| + γ·|roll|)` yang dipakai di
MainActivity.kt:495 dan TestingActivity.kt:585.

## Workflow

```
[1] python record_data.py      → recordings/<subject>.csv
[2] python grid_search.py      → results/thesis_report.txt + heatmap
[3] python validate.py         → verifikasi sebelum deploy
[4] Copy nilai best ke Kotlin (MainActivity.kt + TestingActivity.kt)
```

## Instalasi

```powershell
cd grid_search
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Butuh Python 3.9+ dan webcam berfungsi.

## 1. Perekaman Data

```powershell
python record_data.py --subject taufik_01            # simpan CSV + video MP4 (default)
python record_data.py --subject taufik_01 --no-video # CSV saja, tanpa video
```

**Output**:
- `recordings/<subject>.csv` — data EAR + head pose per frame (untuk grid search)
- `recordings/<subject>.mp4` — video raw webcam tanpa overlay (untuk backup/verifikasi)

Akan menampilkan jendela webcam dengan instruksi visual:
- **Fase SIAP (0–5s)**: countdown
- **Fase LURUS + MATA TERBUKA (5–25s)**: kalibrasi (30 frame pertama jadi offset)
- **Fase mata tertutup di Frontal, Right, Left, Down, Up**: ground truth = 1
- **Fase mata terbuka di tiap orientasi**: ground truth = 0
- 3 detik JEDA antar fase (tidak direkam)

**Total durasi: ~190 detik (~3 menit)**

Subjek hanya perlu **mengikuti instruksi di layar**. Setelah selesai, CSV
disimpan otomatis ke `recordings/<subject>.csv`.

### Tips perekaman
- Cahaya cukup, wajah tegak menghadap kamera saat kalibrasi
- Saat "TENGOK KANAN/KIRI", tahan kepala ~45° (jangan ekstrem 90°)
- Saat "TUNDUK/TENGADAH", pitch ~30°
- Saat "MATA TERTUTUP", tutup mata penuh seperti tidur (bukan kedipan)
- Multi-subjek: ulangi command dengan `--subject` berbeda

```powershell
python record_data.py --subject taufik_01
python record_data.py --subject teman_a
python record_data.py --subject teman_b
```

## 2. Grid Search

```powershell
# Default: stratified holdout 80/20, metric=accuracy, fine search 2 round
python grid_search.py

# Ganti primary metric (F1 untuk cost-sensitive)
python grid_search.py --metric f1
python grid_search.py --metric balanced_accuracy

# LOSO-CV (butuh >= 2 subjek) — laporan paling defensible
python grid_search.py --split loso

# Ubah proporsi test, seed, dst
python grid_search.py --test-size 0.3 --seed 123
python grid_search.py --no-fine          # coarse only
```

Output di folder `results/<timestamp>/` (run lama tidak ter-overwrite):
- `thesis_report.txt` — laporan lengkap untuk skripsi
- `tabel_3_5.csv` — top-4 (siap-tempel Tabel 3.5)
- `top10.csv` — top-10 detail
- `all_combinations.csv` — semua hasil
- `sensitivity_alpha.csv` / `_beta.csv` / `_gamma.csv` — variasi 1-param
- `heatmap_alpha_beta.png` / `_alpha_gamma.png` / `_beta_gamma.png`
- `metadata.json` — args, versi lib, baseline, boundary warnings (reproducibility)
- LOSO: `loso_per_fold.csv` (kalau --split loso)

**Metodologi:**
- **Train/Test split** — stratified per (subject, orientation, eye_state) ATAU LOSO-CV per subjek
- **Coarse (150) → 2 putaran fine search iteratif** (span ±25% lalu ±10%, refine top-3 tiap putaran)
- **Baseline statis** (α=β=γ=0) dilaporkan sebagai pembanding wajib
- **Multi-metric**: accuracy, balanced_accuracy, F1, precision, recall, specificity — semua dilaporkan
- **Boundary check** — warning kalau best param di tepi grid
- **Floor clipping report** — % frame yang T_dyn-nya ter-clip ke {T_DYN_FLOOR}
- **Weighted thesis avg** — by sample count per orientasi (bukan simple mean)
- **Vectorized** via numpy broadcasting (1500+ kombinasi < 1 detik)

## 3. Validasi

```powershell
# Verifikasi best (ganti dengan nilai dari thesis_report.txt)
python validate.py --alpha 0.0020 --beta 0.0010 --gamma 0.0005

# Bandingkan default lama vs best
python validate.py --compare 0.0015,0.001,0.0005  0.0020,0.001,0.0005
```

Output: confusion matrix + precision/recall/F1 per orientasi.

## 4. Update ke Mobile App

Setelah konfirmasi hasil, edit dua file ini di project Android:

**`app/src/main/java/com/example/mobileapp/MainActivity.kt`** (baris 88–91):
```kotlin
private val T_BASE  = 0.24
private val ALPHA   = 0.0020   // ← ganti dengan nilai best
private val BETA    = 0.0010   // ← ganti
private val GAMMA   = 0.0005   // ← ganti
```

**`app/src/main/java/com/example/mobileapp/TestingActivity.kt`** (baris 180–183):
```kotlin
private val T_BASE  = 0.24
private val ALPHA   = 0.0020
private val BETA    = 0.0010
private val GAMMA   = 0.0005
```

Lalu rebuild app & jalankan TestingActivity untuk validasi akhir
(masukkan ke Tabel 3.7 / 3.8 skripsi).

## Struktur Folder

```
grid_search/
├── record_data.py        # script perekaman (webcam + instruksi visual)
├── grid_search.py        # script utama: cari best α,β,γ
├── validate.py           # validasi nilai pilihan
├── requirements.txt
├── README.md
├── recordings/           # output record_data.py
│   ├── taufik_01.csv    # data per frame (untuk grid search)
│   ├── taufik_01.mp4    # video raw webcam (backup/verifikasi)
│   └── ...
└── results/              # output grid_search.py
    ├── thesis_report.txt
    ├── tabel_3_5.csv
    └── ...
```

## Catatan untuk Skripsi

- **Justifikasi metodologi**: grid search dilakukan offline di laptop untuk
  reproducibility & iterasi cepat. Hasil divalidasi di kamera HP via
  TestingActivity (Tabel 3.7 dan 3.8).
- **Mengapa coarse + fine**: coarse search menemukan area umum, fine search
  refine secara lokal — menghindari sampling terlalu jarang di area optimal.
- **Sensitivity analysis**: menjelaskan kontribusi tiap parameter — sesuai
  hipotesis bahwa α (yaw) paling berpengaruh, γ (roll) paling kecil.
- **T_dyn floor 0.10**: konstrain agar threshold tidak negatif/terlalu rendah
  saat kombinasi α,β,γ tinggi di pose ekstrem.
