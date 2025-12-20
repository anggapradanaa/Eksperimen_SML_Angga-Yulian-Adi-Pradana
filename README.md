# Eksperimen SML - Angga Yulian Adi Pradana

## 📁 Deskripsi Folder
Folder ini berisi eksperimen preprocessing data untuk proyek Sistem Machine Learning dengan dataset diabetes.

## 📂 Struktur Folder
```
Eksperimen_SML_Angga-Yulian-Adi-Pradana/
├── diabetes_raw/
│   └── diabetes.csv                           # Dataset asli
├── preprocessing/
│   ├── Eksperimen_Angga-Yulian-Adi-Pradana.ipynb  # Notebook eksperimen
│   └── automate_Angga-Yulian-Adi-Pradana.py       # Script otomasi preprocessing
└── diabetes_preprocessing/                     # Hasil preprocessing
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_test.csv
    └── diabetes_preprocessed.csv
```

## 📊 Dataset - Diabetes Prediction

Dataset ini berisi informasi kesehatan untuk prediksi diabetes dengan **768 baris** dan **9 kolom**.

**Kolom Dataset:**
- `Pregnancies`: Jumlah kehamilan
- `Glucose`: Level glukosa dalam darah
- `BloodPressure`: Tekanan darah
- `SkinThickness`: Ketebalan kulit
- `Insulin`: Level insulin dalam darah
- `BMI`: Body mass index
- `DiabetesPedigreeFunction`: Persentase diabetes
- `Age`: Usia
- `Outcome`: Hasil (1=Diabetes, 0=No Diabetes)

**Tujuan**: Preprocessing data untuk prediksi diabetes

## 🔧 File Utama

### 1. `Eksperimen_Angga-Yulian-Adi-Pradana.ipynb`
Jupyter notebook untuk eksperimen dan eksplorasi data preprocessing

### 2. `automate_Angga-Yulian-Adi-Pradana.py`
Script otomasi untuk preprocessing dataset diabetes dengan tahapan:

**Pipeline Preprocessing:**
1. **Loading Dataset** - Load data dari CSV
2. **Handle Missing Values** - Isi nilai kosong dengan median
3. **Drop Duplicates** - Hapus data duplikat
4. **Scale Features** - Normalisasi fitur menggunakan StandardScaler
5. **Train-Test Split** - Split data 80:20 dengan stratify
6. **Apply SMOTE** - Handle class imbalance dengan oversampling
7. **Save Data** - Simpan hasil preprocessing
8. **Copy to Modelling** - Copy file ke folder Membangun_model

## ▶️ Cara Menjalankan

**Untuk Notebook:**
```bash
jupyter notebook Eksperimen_Angga-Yulian-Adi-Pradana.ipynb
```

**Untuk Script Otomasi:**
```bash
python automate_Angga-Yulian-Adi-Pradana.py
```

## 📊 Output
- **X_train.csv** - Features training setelah SMOTE
- **X_test.csv** - Features testing
- **y_train.csv** - Target training setelah SMOTE
- **y_test.csv** - Target testing
- **diabetes_preprocessed.csv** - Dataset lengkap setelah preprocessing

## 📝 Teknik yang Digunakan
- **Missing Values Handling**: Median Imputation
- **Feature Scaling**: StandardScaler
- **Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Data Splitting**: Stratified Train-Test Split (80:20)

## 📦 Dependencies
```
pandas
numpy
scikit-learn
imbalanced-learn
```

---
**Author**: Angga Yulian Adi Pradana  
