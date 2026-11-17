"""
Cleaning dan preprocessing data hasil filter (file/filtered_text.csv)
untuk skripsi ABSA stunting (aspek: ASI Eksklusif, Imunisasi, Posyandu).

Alur (sesuai BAB III sub-bab 3.4.1):
  1. Cleaning -> Case Folding -> Tokenisasi -> Normalisasi Kata   => Jalur A (IndoBERT+Attention)
  2. Jalur A -> Stopword Removal -> Stemming                      => Jalur B (InSet, Bi-LSTM, Bi-GRU)
  3. Komentar dengan lebih dari satu aspek digandakan menjadi satu baris per aspek.

Output:
  file/preprocessed_text.csv -> hasil tiap tahap preprocessing, siap pelabelan sentimen (InSet)
  file/kandidat_slang.csv    -> kata di luar kamus paling sering, bahan menambah kamus slang
"""
import os
import re
from collections import Counter
from multiprocessing import Pool, cpu_count

import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

FILE_INPUT = 'file/filtered_text.csv'
FILE_OUTPUT = 'file/preprocessed_text.csv'
FILE_KANDIDAT_SLANG = 'file/kandidat_slang.csv'
# Kamus slang tambahan (opsional), format CSV: slang,baku
FILE_SLANG_TAMBAHAN = 'file/kamus_slang.csv'

# ---------------------------------------------------------
# KAMUS NORMALISASI (dictionary-based, dari karakteristik dataset)
# ---------------------------------------------------------
KAMUS_SLANG = {
    # kata umum
    'gak': 'enggak', 'ga': 'enggak', 'gk': 'enggak', 'nggak': 'enggak', 'ngga': 'enggak',
    'engga': 'enggak', 'kaga': 'enggak', 'kagak': 'enggak', 'tdk': 'tidak',
    'gausah': 'enggak usah', 'gamau': 'enggak mau', 'gabisa': 'enggak bisa', 'gapapa': 'enggak apa apa',
    'yg': 'yang', 'dgn': 'dengan', 'dg': 'dengan', 'utk': 'untuk', 'untk': 'untuk', 'tp': 'tapi',
    'tpi': 'tapi', 'krn': 'karena', 'karna': 'karena', 'krna': 'karena', 'jd': 'jadi', 'jdi': 'jadi',
    'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah', 'blm': 'belum', 'blom': 'belum',
    'lg': 'lagi', 'lgi': 'lagi', 'sm': 'sama', 'sama2': 'sama sama', 'aja': 'saja', 'aj': 'saja',
    'aku': 'aku', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya', 'sy': 'saya', 'ak': 'aku', 'aq': 'aku',
    'lu': 'kamu', 'lo': 'kamu', 'loe': 'kamu', 'km': 'kamu', 'kmu': 'kamu',
    'kalo': 'kalau', 'klo': 'kalau', 'kl': 'kalau', 'kalu': 'kalau', 'bgt': 'banget', 'bngt': 'banget',
    'banget': 'banget', 'emg': 'memang', 'emang': 'memang', 'skrg': 'sekarang', 'skrng': 'sekarang',
    'sekarang': 'sekarang', 'gmn': 'bagaimana', 'gimana': 'bagaimana', 'gmna': 'bagaimana',
    'knp': 'kenapa', 'napa': 'kenapa', 'dmn': 'di mana', 'mana2': 'mana mana', 'org': 'orang',
    'orng': 'orang', 'tau': 'tahu', 'tahu': 'tahu', 'tlg': 'tolong', 'tolongin': 'tolong',
    'bs': 'bisa', 'bsa': 'bisa', 'bkn': 'bukan', 'msh': 'masih', 'masi': 'masih', 'pdhl': 'padahal',
    'trs': 'terus', 'trus': 'terus', 'terus': 'terus', 'mksd': 'maksud', 'bikin': 'buat',
    'pengen': 'ingin', 'pgn': 'ingin', 'pingin': 'ingin', 'pengin': 'ingin', 'mau': 'mau', 'mo': 'mau',
    'mw': 'mau', 'bnyk': 'banyak', 'byk': 'banyak', 'dr': 'dari', 'dri': 'dari', 'kpd': 'kepada',
    'td': 'tadi', 'tdi': 'tadi', 'bbrp': 'beberapa', 'spt': 'seperti', 'kyk': 'kayak', 'kyak': 'kayak',
    'kek': 'kayak', 'gitu': 'begitu', 'gt': 'begitu', 'gini': 'begini',
    'doang': 'saja', 'cuman': 'cuma', 'cm': 'cuma', 'dapet': 'dapat',
    'liat': 'lihat', 'ngeliat': 'melihat', 'nyari': 'mencari', 'nanya': 'bertanya', 'bilang': 'bilang',
    'blg': 'bilang', 'ngasih': 'memberi', 'bener': 'benar',
    'bnr': 'benar', 'beneran': 'benar', 'trims': 'terima kasih', 'makasih': 'terima kasih',
    'mksh': 'terima kasih', 'thx': 'terima kasih', 'thanks': 'terima kasih', 'ok': 'oke', 'okey': 'oke',
    'jg': 'juga', 'jga': 'juga', 'juga': 'juga', 'ama': 'sama', 'bwt': 'buat', 'tuh': 'itu',
    'nih': 'ini', 'skali': 'sekali', 'sekali2': 'sekali sekali', 'tiap': 'setiap', 'tmn': 'teman',
    'temen': 'teman', 'ortu': 'orang tua', 'rmh': 'rumah', 'kmrn': 'kemarin', 'kemaren': 'kemarin',
    'besok2': 'besok', 'thn': 'tahun', 'th': 'tahun', 'bln': 'bulan', 'mgg': 'minggu', 'hr': 'hari',
    'jam2': 'jam', 'ngerti': 'mengerti', 'ngerasa': 'merasa', 'kerasa': 'terasa', 'nyesel': 'menyesal',
    'males': 'malas', 'cape': 'capek', 'capek': 'capek', 'sedih': 'sedih', 'seneng': 'senang',
    'sneng': 'senang', 'parah': 'parah', 'mantap': 'mantap', 'mantul': 'mantap', 'keren': 'keren',
    'jelek': 'jelek', 'ancur': 'hancur', 'gede': 'besar', 'gedhe': 'besar', 'dikit': 'sedikit',
    'sedikit': 'sedikit', 'duit': 'uang', 'mahal': 'mahal', 'murah': 'murah', 'ribet': 'rumit',
    'susah': 'susah', 'gampang': 'mudah', 'cepet': 'cepat', 'lama': 'lama', 'telat': 'terlambat',
    'mulu': 'melulu', 'mulu2': 'melulu', 'sampe': 'sampai', 'smpe': 'sampai', 'smp': 'sampai',
    'ampe': 'sampai', 'nunggu': 'menunggu', 'nangis': 'menangis', 'nolak': 'menolak',
    'minum2': 'minum', 'pake': 'pakai', 'make': 'memakai', 'krg': 'kurang',
    'lbh': 'lebih', 'kecil2': 'kecil', 'bagus': 'bagus', 'bgs': 'bagus', 'baik2': 'baik baik',
    'sehat2': 'sehat sehat', 'moga': 'semoga', 'smoga': 'semoga', 'mudah2an': 'mudah mudahan',
    'alhamdulilah': 'alhamdulillah', 'alhamdulillah': 'alhamdulillah', 'masyaallah': 'masyaallah',
    'mashaallah': 'masyaallah', 'insyaallah': 'insyaallah', 'inshaallah': 'insyaallah',
    'ttp': 'tetap', 'jgn': 'jangan', 'dlu': 'dulu', 'tgl': 'tanggal', 'sblm': 'sebelum',
    'hrs': 'harus', 'brp': 'berapa', 'gatau': 'enggak tahu', 'gaada': 'enggak ada',
    'gapernah': 'enggak pernah', 'gpp': 'enggak apa apa', 'lgsg': 'langsung', 'sbg': 'sebagai',
    'dlm': 'dalam', 'ttg': 'tentang', 'nnti': 'nanti', 'inget': 'ingat', 'deket': 'dekat',
    'dateng': 'datang', 'sempet': 'sempat', 'mikir': 'berpikir', 'sebenernya': 'sebenarnya',
    'plis': 'tolong', 'anget': 'hangat', 'terimakasih': 'terima kasih', 'ko': 'kok', 'uda': 'sudah',
    'abis': 'habis', 'gtu': 'begitu', 'serem': 'seram', 'ngomong': 'bicara',
    # sapaan dan partikel yang hanya diseragamkan
    'bun': 'bunda', 'bund': 'bunda', 'buibu': 'ibu ibu', 'buk': 'ibu', 'bu': 'ibu', 'mom': 'ibu',
    'moms': 'ibu', 'mommy': 'ibu', 'mama': 'ibu', 'mami': 'ibu', 'emak': 'ibu', 'mak': 'ibu',
    'kak': 'kakak', 'ka': 'kakak', 'kk': 'kakak', 'pak': 'bapak', 'pa': 'bapak', 'bapak2': 'bapak bapak',
    'paksu': 'suami', 'misua': 'suami', 'bumil': 'ibu hamil', 'busui': 'ibu menyusui',
    'anakku': 'anak aku', 'anaku': 'anak aku', 'ank': 'anak', 'adek': 'adik', 'mba': 'mbak',
    'mams': 'ibu', 'kakk': 'kakak', 'ortunya': 'orang tua', 'anakmu': 'anak kamu', 'bocil': 'anak kecil', 'bocah': 'anak',
    'baby': 'bayi', 'babyku': 'bayi aku', 'newborn': 'bayi baru lahir', 'nb': 'bayi baru lahir',
    'toddler': 'balita',
    # istilah domain kesehatan anak
    'bb': 'berat badan', 'tb': 'tinggi badan', 'lk': 'lingkar kepala',
    'dsa': 'dokter spesialis anak', 'dokter anak': 'dokter spesialis anak',
    'dok': 'dokter', 'nakes': 'tenaga kesehatan', 'pkm': 'puskesmas', 'rs': 'rumah sakit',
    'rsud': 'rumah sakit', 'bidan': 'bidan', 'faskes': 'fasilitas kesehatan',
    'imun': 'imunisasi', 'imunisasinya': 'imunisasi', 'vaksinnya': 'vaksin', 'vaksinasinya': 'vaksinasi',
    'vax': 'vaksin', 'vaksen': 'vaksin', 'divaksin': 'divaksin', 'antivaks': 'anti vaksin',
    'antivaksin': 'anti vaksin', 'antivax': 'anti vaksin', 'provaks': 'pro vaksin',
    'asinya': 'asi', 'asiku': 'asi aku', 'asix': 'asi eksklusif', 'asieks': 'asi eksklusif',
    'asip': 'asi perah', 'mpasinya': 'mpasi', 'sufornya': 'sufor', 'susu formula': 'sufor',
    'pumping': 'pumping', 'mumping': 'pumping', 'pompa': 'pompa', 'pd': 'payudara',
    'bbnya': 'berat badan', 'nyusu': 'menyusu', 'puskes': 'puskesmas', 'bapil': 'batuk pilek',
    'gtm': 'gerakan tutup mulut', 'rota': 'rotavirus',
    'posyandunya': 'posyandu', 'yandu': 'posyandu', 'buku pink': 'buku kia',
}

# Partikel/kata pengisi informal tambahan untuk stopword removal (Jalur B)
STOPWORD_TAMBAHAN = {
    'sih', 'deh', 'dong', 'kok', 'ya', 'yah', 'yaa', 'kan', 'nah', 'lah', 'loh', 'lho', 'wkwk',
    'wkwkwk', 'haha', 'hahaha', 'hehe', 'hihi', 'hmm', 'eh', 'aduh', 'duh', 'nya', 'an', 'nge',
    'kayak', 'banget', 'aja', 'saja', 'gitu', 'begitu', 'begini', 'oke', 'tuh', 'nih',
}
# Negasi tidak dibuang karena membalik polaritas sentimen (mis. "enggak", "tidak", "belum")
STOPWORD_DIPERTAHANKAN = {'tidak', 'enggak', 'bukan', 'belum', 'jangan', 'tak', 'kurang', 'lebih'}

# Istilah domain yang tidak boleh diubah Sastrawi (mis. "menyusui" -> "susu" menjadi ambigu)
KATA_DILINDUNGI = {
    'posyandu', 'imunisasi', 'vaksin', 'vaksinasi', 'asi', 'mpasi', 'sufor', 'menyusui', 'menyusu',
    'pumping', 'kader', 'stunting', 'balita', 'puskesmas', 'kipi', 'bcg', 'dpt', 'polio', 'campak',
    'rotavirus', 'pcv', 'kms', 'kia', 'pmt', 'eksklusif', 'laktasi', 'bidan',
}

FRASA_SLANG = []  # diisi di main(): entri kamus berupa frasa dua kata


def muat_kamus_slang():
    kamus = dict(KAMUS_SLANG)
    if os.path.exists(FILE_SLANG_TAMBAHAN):
        tambahan = pd.read_csv(FILE_SLANG_TAMBAHAN)
        kamus.update(dict(zip(tambahan['slang'].str.strip().str.lower(), tambahan['baku'].str.strip().str.lower())))
        print(f"Kamus slang tambahan dimuat: {len(tambahan)} entri dari '{FILE_SLANG_TAMBAHAN}'")
    return kamus


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', ' ', text)
    text = text.replace('#', ' ')
    # Kata ulang "anak2" / "hambur²in" -> "anak anak" / "hambur hamburin"
    text = re.sub(r'\b([A-Za-z]+)[2²]([A-Za-z]*)\b', r'\1 \1\2', text)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)  # angka, tanda baca, emoji, karakter khusus
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def case_folding(text):
    text = text.lower()
    # Huruf berulang ekspresif: "bangeeettt" -> "banget", "asiii" -> "asi"
    return re.sub(r'([a-z])\1{2,}', r'\1', text)


def tokenisasi(text):
    return text.split()


def normalisasi(tokens, kamus):
    hasil = []
    for token in tokens:
        hasil.extend(kamus.get(token, token).split())
    # Frasa dua kata dalam kamus (mis. "buku pink", "susu formula")
    teks = ' '.join(hasil)
    for frasa, baku in FRASA_SLANG:
        if frasa in teks:
            teks = re.sub(rf'\b{frasa}\b', baku, teks)
    return teks.split()


_STEMMER = None


def _stem_kata(kata):
    # Dipanggil di proses worker multiprocessing; tiap worker membuat stemmer sendiri
    global _STEMMER
    if _STEMMER is None:
        _STEMMER = StemmerFactory().create_stemmer()
    return kata, _STEMMER.stem(kata)


def stem_kosakata(kosakata):
    """Stemming kosakata unik sekali saja secara paralel (Sastrawi lambat per kata)."""
    with Pool(max(1, cpu_count() - 1)) as pool:
        return dict(pool.imap_unordered(_stem_kata, sorted(kosakata), chunksize=200))


def main():
    try:
        df = pd.read_csv(FILE_INPUT)
    except FileNotFoundError:
        print(f"File '{FILE_INPUT}' tidak ditemukan. Jalankan filter_raw_text.py terlebih dahulu.")
        return

    print("Menginisialisasi modul Sastrawi...")
    kata_dasar = set(StemmerFactory().get_words())
    stopwords = (set(StopWordRemoverFactory().get_stop_words()) | STOPWORD_TAMBAHAN) - STOPWORD_DIPERTAHANKAN
    kamus = muat_kamus_slang()
    FRASA_SLANG[:] = [(f, b) for f, b in kamus.items() if ' ' in f]
    kosakata_dikenal = kata_dasar | set(kamus) | KATA_DILINDUNGI | stopwords
    for baku in kamus.values():
        kosakata_dikenal.update(baku.split())

    df = df.dropna(subset=['content'])
    print(f"Total data hasil filter: {len(df)}")

    # Tahap 1: cleaning, case folding, tokenisasi, normalisasi (Jalur A)
    baris = []
    for row in df.itertuples(index=False):
        clean = clean_text(row.content)
        folded = case_folding(clean)
        tokens = tokenisasi(folded)
        normalized = normalisasi(tokens, kamus)
        baris.append((row, clean, folded, tokens, normalized))

    # Tahap 2: stemming kosakata unik yang belum berbentuk kata dasar
    kosakata = {t for *_, normalized in baris for t in normalized if t not in stopwords}
    perlu_stem = {t for t in kosakata if t not in kata_dasar and t not in KATA_DILINDUNGI}
    print(f"Stemming {len(perlu_stem)} kata unik secara paralel. Mohon tunggu...")
    hasil_stem = stem_kosakata(perlu_stem)

    def stem(token):
        return hasil_stem.get(token, token)

    # Tahap 3: stopword removal + stemming (Jalur B), digandakan per aspek
    hasil = []
    kata_luar_kamus = Counter()
    for row, clean, folded, tokens, normalized in baris:
        no_stopword = [t for t in normalized if t not in stopwords]
        stemmed = [t for t in (stem(t) for t in no_stopword) if len(t) > 1]
        kata_luar_kamus.update(t for t in normalized if t not in kosakata_dikenal and stem(t) not in kata_dasar)

        aspek_list = str(row.aspek).split(', ')
        for aspek in aspek_list:
            hasil.append({
                'keyword': row.keyword,
                'aspek': aspek,
                'jumlah_aspek': len(aspek_list),
                'content': row.content,
                'clean_text': clean,
                'case_folding': folded,
                'tokenized': str(tokens),
                'normalized_tokens': str(normalized),
                'tokens_no_stopword': str(no_stopword),
                'stemmed_tokens': str(stemmed),
                'text_jalur_a': ' '.join(normalized),
                'text_jalur_b': ' '.join(stemmed),
            })

    df_hasil = pd.DataFrame(hasil)
    # Teks yang kosong setelah stopword removal & stemming tidak bisa dipakai model
    kosong = df_hasil['text_jalur_b'].str.strip() == ''
    df_hasil = df_hasil[~kosong]
    df_hasil.insert(0, 'No', range(1, 1 + len(df_hasil)))
    df_hasil.to_csv(FILE_OUTPUT, index=False, sep=';', encoding='utf-8-sig')

    pd.DataFrame(kata_luar_kamus.most_common(500), columns=['kata', 'frekuensi']).to_csv(
        FILE_KANDIDAT_SLANG, index=False, sep=';', encoding='utf-8-sig')

    print("\n================ RINGKASAN ================")
    print(f"Komentar hasil filter : {len(df)}")
    print(f"Baris per aspek       : {len(df_hasil)} (dibuang karena kosong: {int(kosong.sum())})")
    print("\nJumlah per aspek:")
    print(df_hasil['aspek'].value_counts().to_string())
    print(f"\n✅ Hasil preprocessing: '{FILE_OUTPUT}'")
    print(f"📝 Kandidat slang untuk ditambahkan ke '{FILE_SLANG_TAMBAHAN}': '{FILE_KANDIDAT_SLANG}'")


if __name__ == "__main__":
    main()
