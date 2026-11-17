# Threads Data Scraper & NLP Preprocessing Pipeline

Kumpulan skrip Python untuk skripsi **"Analisis Sentimen Berbasis Aspek ASI Eksklusif, Imunisasi, dan Posyandu"** (IndoBERT, Bi-LSTM, dan Bi-GRU). Alurnya: mengambil komentar dari Threads (_scraping_), menyaring data yang tidak relevan, lalu melakukan _preprocessing_ teks Bahasa Indonesia hingga siap dilabeli dan dipakai untuk pemodelan.

## 🔄 Alur Pipeline

```
scraper.py            ->  file/raw_text.csv
filter_raw_text.py    ->  file/filtered_text.csv   (+ file/filtered_out.csv)
preprocessing_text.py ->  file/preprocessed_text.csv (+ file/kandidat_slang.csv)
```

Jalankan berurutan:

```bash
python scraper.py
python filter_raw_text.py
python preprocessing_text.py
```

## 🛠️ Prasyarat & Instalasi

- Python 3.8+
- Google Chrome terinstal (scraper memakai Chrome asli, bukan browser bawaan Playwright)

```bash
pip install playwright pandas Sastrawi
```

## 1️⃣ Scraping — `scraper.py`

- **Multi-keyword:** 30 kata kunci seputar stunting, ASI eksklusif, imunisasi, dan posyandu, masing-masing di tab _Top_ dan _Recent_.
- **Target 40.000 data** (`target_total`). Data ditulis per baris, jadi aman bila skrip terhenti; saat dijalankan ulang skrip **melanjutkan** dari data yang sudah ada.
- **Tanpa duplikat:** dicek berdasarkan URL postingan dan isi teks (huruf kecil, tanpa emoji/angka/tanda baca), termasuk terhadap data run sebelumnya.
- **Login manual sekali:** sesi tersimpan di `file/browser_profile`, jadi run berikutnya biasanya sudah login.
- **Mode ditinggal (`UNATTENDED = True`):**
  - Tempo dibuat pelan agar captcha jarang muncul: jeda scroll 4–8 detik, jeda antar pencarian 30–90 detik, dan istirahat 5–10 menit setiap 5 pencarian.
  - Bila captcha muncul, skrip **tidak menyelesaikannya otomatis**. Skrip istirahat 30 menit lalu mengecek ulang, maksimal 8 kali (±4 jam). Selama jeda, captcha boleh diselesaikan manual di Chrome.
  - Bila captcha tetap ada, skrip berhenti dengan rapi dan data tetap tersimpan.
  - Laptop dicegah masuk _sleep_ selama scraping (charger harus terpasang dan laptop jangan ditutup).
  - Set `UNATTENDED = False` untuk kembali ke mode manual (tekan Enter setelah menyelesaikan captcha).

Output: `file/raw_text.csv` (kolom `keyword`, `content`).

## 2️⃣ Filter — `filter_raw_text.py`

Data mentah dibuang bila:

1. **Tidak membahas kata kunci aspek**: tidak ada kata kunci ASI Eksklusif (mis. _asi, menyusui, sufor, pumping_), Imunisasi (mis. _imunisasi, vaksin, kipi, bcg, polio, campak_), atau Posyandu (mis. _posyandu, kms, buku kia, pmt_).
2. **Duplikat**: isi teks sama setelah dibersihkan.
3. **Bahasa Melayu**: skor penanda Melayu ≥ 3 (penanda kuat seperti _korang, kerana, takde, ubat_ bernilai 3; penanda lemah seperti _je, nak, tak_ bernilai 1).

Output:

- `file/filtered_text.csv`: data lolos filter, ditambah kolom `aspek` (satu komentar bisa punya lebih dari satu aspek).
- `file/filtered_out.csv`: data yang dibuang beserta kolom `alasan`.

## 3️⃣ Preprocessing — `preprocessing_text.py`

| Jalur | Tahapan                                                               | Dipakai untuk                    |
| ----- | --------------------------------------------------------------------- | -------------------------------- |
| **A** | Cleaning → Case Folding → Tokenisasi → Normalisasi kata (kamus slang) | IndoBERT                         |
| **B** | Jalur A → Stopword Removal → Stemming (Sastrawi)                      | Pelabelan InSet, Bi-LSTM, Bi-GRU |

- Kata negasi (_tidak, enggak, bukan, belum, jangan_, dll.) **tidak** dibuang saat stopword removal karena membalik polaritas sentimen.
- Istilah domain (_posyandu, imunisasi, vaksin, asi, menyusui_, dll.) dilindungi dari stemming.
- Komentar dengan lebih dari satu aspek **digandakan** menjadi satu baris per aspek.
- Kamus slang bisa ditambah lewat `file/kamus_slang.csv` (kolom `slang,baku`). Kandidatnya ada di `file/kandidat_slang.csv`.

Output `file/preprocessed_text.csv` (separator `;`, encoding `utf-8-sig`, rapi saat dibuka di Excel) berisi kolom:
`No, keyword, aspek, jumlah_aspek, content, clean_text, case_folding, tokenized, normalized_tokens, tokens_no_stopword, stemmed_tokens, text_jalur_a, text_jalur_b`.
