# Sentiment Analysis — Review & Komentar Bahasa Indonesia

Portfolio project — klasifikasi sentimen (positive/neutral/negative) untuk teks Bahasa Indonesia, lengkap dengan analisis kata kunci per sentimen dan dashboard interaktif.

## 🎯 Tujuan Project

Membangun model yang bisa otomatis mengenali sentimen dari review/komentar Bahasa Indonesia — berguna untuk use case seperti monitoring sentimen media sosial, analisis review produk, atau riset opini publik, tanpa perlu membaca ribuan komentar secara manual.

## 🧰 Tech Stack

- **Python**: pandas, numpy, scikit-learn
- **NLP Bahasa Indonesia**: Sastrawi (stopword removal & stemming)
- **Visualisasi**: matplotlib, seaborn, wordcloud
- **Dashboard**: Streamlit

## 📁 Struktur Folder

```
sentiment-analysis-portfolio/
├── data/
│   └── indonlu_smsa_sentiment.csv   # dataset final (3.000 baris, sudah bersih)
├── notebooks/
│   └── 01_sentiment_analysis.ipynb  # analisis end-to-end
├── app/
│   └── dashboard.py                 # dashboard interaktif Streamlit
├── requirements.txt
└── README.md
```

## 📊 Dataset

**IndoNLU — SmSA (Sentence-level Sentiment Analysis)**
https://github.com/IndoNLP/indonlu

Dataset publik resmi berisi review & komentar Bahasa Indonesia dari berbagai platform online (restoran, produk, isu sosial-politik), dikumpulkan dan dianotasi oleh peneliti NLP Indonesia (Prosa.ai & tim IndoNLU) dengan lisensi MIT.

**Catatan penting soal data**: dataset asli IndoNLU punya total ±12.700 baris berlabel asli (train + valid). File `test_preprocess_masked_label.tsv` dari sumber asli **tidak dipakai** karena labelnya di-mask (placeholder untuk keperluan benchmark blind test), bukan ground truth asli. Project ini men-sampling 3.000 baris secara stratified dari data berlabel asli, lalu membuat split train/valid/test sendiri (proporsi label tetap terjaga).

## 🔬 Alur Analisis (lihat notebook)

1. **EDA** — distribusi label, panjang teks per sentimen, wordcloud per kelas.
2. **Text Preprocessing** — case folding, cleaning, stopword removal, stemming (Sastrawi).
3. **Feature Extraction** — TF-IDF (unigram + bigram).
4. **Modeling** — Naive Bayes, Logistic Regression, Linear SVM.
5. **Evaluasi & Error Analysis** — precision/recall per kelas, confusion matrix, contoh kesalahan prediksi.
6. **Business Insight** — kata kunci paling berpengaruh per sentimen, use case nyata, batasan model.

## 📈 Ringkasan Hasil

Model terbaik: **Logistic Regression** — Accuracy 83.1% (validation), 81% (test set, data belum pernah dilihat).

| Kelas | Precision | Recall | F1-Score |
|---|---|---|---|
| Positive | 0.93 | 0.87 | 0.90 |
| Negative | 0.71 | 0.79 | 0.75 |
| Neutral | 0.51 | 0.54 | 0.52 |

Kelas `neutral` paling sulit diprediksi karena proporsi datanya paling kecil (~10%) dan secara linguistik lebih ambigu.

## 🚀 Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan notebook
jupyter notebook notebooks/01_sentiment_analysis.ipynb

# 3. Jalankan dashboard
streamlit run app/dashboard.py
```

## 📌 Catatan Sebelum Dipakai Melamar Kerja

Notebook ini sudah **tervalidasi jalan tanpa error** (23 cell, 0 error) dengan data asli IndoNLU (bukan data sintetis). Sebelum dipakai sebagai portfolio final:

1. Baca ulang tiap bagian notebook, pastikan lo paham **kenapa** memilih TF-IDF (bukan word embedding), **kenapa** Macro F1 lebih relevan dibanding Accuracy untuk data imbalanced ini — ini yang sering ditanya wawancara.
2. Kalau ada waktu lebih, coba latih ulang pakai seluruh ±12.700 data (bukan cuma 3.000 sampel) untuk performa yang berpotensi lebih baik — notebook sudah didesain reusable untuk itu.
3. Screenshot dashboard atau deploy ke Streamlit Community Cloud (gratis) supaya recruiter bisa coba langsung.
4. Pertimbangkan menyebutkan potensi pengembangan ke arah IndoBERT/transformer sebagai langkah lanjutan — ini nunjukin lo paham batasan model klasik vs deep learning.

## 👤 Author

Dibuat sebagai bagian dari portfolio data science.
