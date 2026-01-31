# Project Artificial Intelligence — Klasifikasi Prediksi Customer Churn (Telekomunikasi)

**Nama:** Arhan Malik Alrasyid  
**NIM:** 20220801151  

---

## Ringkasan Proyek
Proyek ini membangun **model klasifikasi** untuk memprediksi apakah seorang pelanggan akan **berhenti berlangganan (churn)** atau **tetap berlangganan** berdasarkan data layanan, kontrak, dan pola pembayaran pelanggan.

Model yang digunakan:
1. **Logistic Regression** (baseline)
2. **Random Forest Classifier** (model yang lebih kuat)

Evaluasi dilakukan menggunakan:
- **Accuracy**
- **ROC-AUC**
- **Confusion Matrix**
- **Classification Report**
- **ROC Curve** (visualisasi)

Notebook juga menyimpan model akhir ke file: `model_churn_rf.joblib`.

---

## Dataset
Dataset yang dipakai: **Telco Customer Churn** (Kaggle)

- Sumber (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
- File CSV yang umum: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Penempatan dataset (penting)
Di notebook, dataset dibaca dari path berikut:

```text
dataaset/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Alur Pengerjaan (Workflow)
1. **Import library** yang dibutuhkan (NumPy, Pandas, scikit-learn, Matplotlib).
2. **Load dataset** dari file CSV.
3. **Data cleaning**:
   - Kolom `TotalCharges` dikonversi ke numerik (`errors="coerce"` untuk menangani data kosong/spasi).
   - Kolom ID `customerID` dihapus karena tidak membantu prediksi.
4. **Split data**:
   - Fitur: `X = df.drop("Churn")`
   - Target: `y` (Churn dikonversi menjadi 0/1)
   - Train-test split 80:20 dengan `stratify=y`
5. **Preprocessing otomatis (Pipeline)**:
   - Numerik: imputasi median + standardisasi
   - Kategori: imputasi modus + OneHotEncoder
6. **Training model**:
   - Baseline: Logistic Regression
   - Model utama: Random Forest (dengan `class_weight="balanced"`)
7. **Evaluasi** dengan metrik klasifikasi + plot ROC Curve.
8. **Simpan model** Random Forest menggunakan `joblib`.

---

## Cara Menjalankan (Jupyter Notebook)
### 1) Install dependency
Jika kamu memakai virtual environment (venv), jalankan:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib joblib ipykernel
```

### 2) Jalankan notebook
Buka dan jalankan file:
- `Klasifikasi_Prediksi_Customer_Churn_(Telekomunikasi).ipynb`

Lalu klik **Run All** atau jalankan cell satu per satu dari atas ke bawah.

---

## Output yang Dihasilkan
Setelah notebook dijalankan, kamu akan mendapatkan:
- Teks hasil evaluasi (Accuracy, ROC-AUC, laporan klasifikasi, confusion matrix)
- Grafik **ROC Curve**
- File model tersimpan:
  - `model_churn_rf.joblib`

---

## Catatan
- `TotalCharges` sering bermasalah karena ada nilai kosong/spasi; itu sebabnya dikonversi dengan `pd.to_numeric(..., errors="coerce")`.
- Karena dataset churn sering tidak seimbang, Random Forest menggunakan `class_weight="balanced"` untuk membantu performa pada kelas minoritas.

---

## Referensi
- Dataset Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- scikit-learn documentation: https://scikit-learn.org/
