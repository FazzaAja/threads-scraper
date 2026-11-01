# Threads Data Scraper & NLP Preprocessing Pipeline

Repositori ini berisi kumpulan skrip Python untuk melakukan ekstraksi data teks (_scraping_) dari platform sosial media Threads dan melakukan pra-pemrosesan teks (_text preprocessing_) menggunakan metode Natural Language Processing (NLP).

Program ini dirancang khusus untuk melewati rintangan _Single Page Application_ (SPA) pada Threads dan menyiapkan dataset mentah menjadi data bersih yang siap digunakan untuk pemodelan Machine Learning (Analisis Sentimen).

## 🚀 Fitur Utama

1. **Multi-Keyword Scraper:** Mampu mencari dan mengambil data dari beberapa kata kunci sekaligus secara otomatis.
2. **Anti-Bot & SPA Bypass:** Dilengkapi dengan jeda waktu untuk login manual guna menghindari blokir sistem Meta, serta eksekusi JavaScript untuk memuat halaman pencarian.
3. **Auto-Pagination:** Melakukan _scroll_ halaman secara otomatis untuk memuat data lama hingga mencapai target jumlah postingan.
4. **Indonesian NLP Preprocessing:** Menggunakan modul **Sastrawi** untuk menghapus kata hubung (_stopword removal_) dan mencari kata dasar (_stemming_) khusus bahasa Indonesia.
5. **Excel-Ready Export:** Mengekspor hasil akhir ke dalam format CSV dengan separator titik koma (`;`) dan encoding `utf-8-sig` agar rapi dan terbagi per kolom saat dibuka di Microsoft Excel.

## 🛠️ Prasyarat & Instalasi

Pastikan Anda telah menginstal **Python 3.8+** di perangkat Anda. Buka terminal/Command Prompt dan jalankan perintah berikut untuk menginstal seluruh pustaka yang dibutuhkan:

```bash
# Instal pustaka utama
pip install playwright pandas Sastrawi

# Instal browser bawaan Playwright (Wajib untuk scraping)
playwright install
```
