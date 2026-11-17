"""
Filter data mentah Threads (file/raw_text.csv) sebelum preprocessing.

Data dibuang bila:
  1. Tidak membahas kata kunci aspek penelitian (ASI Eksklusif, Imunisasi, Posyandu)
  2. Duplikat        : isi teks sama setelah dibersihkan (huruf kecil, tanpa emoji/tanda baca)
  3. Bahasa Melayu   : mengandung penanda khas Bahasa Melayu

Output:
  file/filtered_text.csv -> data lolos filter (kolom sama dengan raw_text.csv + kolom aspek)
  file/filtered_out.csv  -> data yang dibuang beserta alasannya
"""
import re

import pandas as pd

FILE_INPUT = 'file/raw_text.csv'
FILE_OUTPUT = 'file/filtered_text.csv'
FILE_BUANG = 'file/filtered_out.csv'

MIN_SKOR_MELAYU = 3

# Kata kunci tiap aspek, dicocokkan pada teks yang sudah di-lowercase
ASPEK_POLA = {
    'ASI Eksklusif': r'\b(asi|asinya|asiku|asix|asip|asi eksklusif|menyusui|menyusu|nyusu|disusui|busui|'
                     r'dbf|pumping|laktasi|sufor|susu formula|pompa asi|donor asi)\b',
    'Imunisasi': r'\b(imunisasi\w*|diimunisasi|vaksin\w*|divaksin|vaksinasi|kipi|antivaks\w*|'
                 r'bcg|dpt|polio|campak|rotavirus|pcv|mmr)\b',
    'Posyandu': r'\b(posyandu\w*|kms|buku kia|buku pink|pmt)\b',
}
ASPEK_REGEX = {aspek: re.compile(pola) for aspek, pola in ASPEK_POLA.items()}

# Penanda Bahasa Melayu: kuat = hampir tidak dipakai di Bahasa Indonesia (satu saja sudah cukup),
# lemah = dipakai juga di Bahasa Indonesia informal (baru dihitung bila beberapa muncul bersama)
PENANDA_MELAYU_KUAT = {
    'korang', 'sihat', 'kerana', 'sahaja', 'awak', 'budak', 'cakna', 'ambik', 'ibubapa', 'kanak',
    'ubat', 'jangkitan', 'takde', 'xde', 'kesihatan', 'doktor', 'percuma', 'kat', 'dorang', 'diorang',
    'diaorang', 'cucuk', 'takkan', 'cakap', 'ckp', 'kecik', 'takpe', 'takda', 'punca', 'risau',
    'jangkit', 'berjangkit', 'realiti', 'tgh', 'jugak', 'pulak', 'teruk', 'sgt', 'mcm', 'amik', 'pon', 'nk', 'xnak', 'kot',
}
PENANDA_MELAYU_LEMAH = {'je', 'ni', 'tu', 'nak', 'dah', 'macam', 'x', 'laa', 'tak'}
FRASA_MELAYU = re.compile(r'\b(kesan sampingan|klinik kesihatan|kanak kanak|tak boleh|macam mana|tak nak)\b')


def bersihkan(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'([a-z])[2²]', r'\1 \1', text)  # kata ulang "anak2" -> "anak anak"
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'([a-z])\1{2,}', r'\1', text)  # "vaksiiin" -> "vaksin"
    return re.sub(r'\s+', ' ', text).strip()


def skor_melayu(text):
    tokens = text.split()
    skor = sum(MIN_SKOR_MELAYU for t in tokens if t in PENANDA_MELAYU_KUAT)
    skor += sum(1 for t in tokens if t in PENANDA_MELAYU_LEMAH)
    skor += MIN_SKOR_MELAYU * len(FRASA_MELAYU.findall(text))
    return skor


def deteksi_aspek(text):
    return [aspek for aspek, regex in ASPEK_REGEX.items() if regex.search(text)]


def main():
    try:
        df = pd.read_csv(FILE_INPUT)
    except FileNotFoundError:
        print(f"File '{FILE_INPUT}' tidak ditemukan.")
        return

    df = df.dropna(subset=['content'])
    print(f"Total data mentah: {len(df)}")

    lolos, buang = [], []
    teks_terlihat = set()
    for row in df.itertuples(index=False):
        text = bersihkan(row.content)
        aspek = deteksi_aspek(text)

        if not aspek:
            alasan = 'tidak membahas kata kunci'
        elif text in teks_terlihat:
            alasan = 'duplikat'
        elif skor_melayu(text) >= MIN_SKOR_MELAYU:
            alasan = 'bahasa Melayu'
        else:
            teks_terlihat.add(text)
            lolos.append({'keyword': row.keyword, 'content': row.content, 'aspek': ', '.join(aspek)})
            continue
        teks_terlihat.add(text)
        buang.append({'keyword': row.keyword, 'content': row.content, 'alasan': alasan})

    hasil = pd.DataFrame(lolos)
    hasil.to_csv(FILE_OUTPUT, index=False, encoding='utf-8-sig')
    df_buang = pd.DataFrame(buang)
    df_buang.to_csv(FILE_BUANG, index=False, encoding='utf-8-sig')

    print(f"Lolos filter : {len(hasil)}")
    print(f"Dibuang      : {len(df_buang)}")
    print("\nDibuang per alasan:")
    print(df_buang['alasan'].value_counts().to_string())
    print("\nLolos per aspek (satu komentar bisa lebih dari satu aspek):")
    print(hasil['aspek'].str.split(', ').explode().value_counts().to_string())
    print(f"\n✅ Data bersih : '{FILE_OUTPUT}'")
    print(f"🗑️  Data dibuang: '{FILE_BUANG}'")


if __name__ == "__main__":
    main()
