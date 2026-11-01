"""
Preprocessing dan filter data Threads untuk skripsi ABSA stunting
(aspek: ASI Eksklusif, Imunisasi, Posyandu).

Alur (sesuai BAB III):
  1. Cleaning -> Case Folding -> Tokenisasi -> Normalisasi Kata   => Jalur A (IndoBERT+Attention)
  2. Jalur A -> Stopword Removal -> Stemming                      => Jalur B (InSet, Bi-LSTM, Bi-GRU)
  3. Filter seleksi data (sub-bab 3.1.2):
       F1 teks terlalu pendek / tidak bermakna
       F2 promosi, iklan, spam
       F3 bukan Bahasa Indonesia (Melayu, Inggris, dll.)
       F4 tidak menyinggung satu pun dari tiga aspek
       F5 istilah aspek muncul tetapi konteksnya bukan anak (vaksin kucing, kader partai, ...)
       F6 duplikat secara substansi (teks hasil normalisasi sama)
  4. Pelabelan aspek: satu komentar dengan lebih dari satu aspek digandakan per aspek.

Output:
  file/cleaned_text.csv   -> dataset siap pelabelan sentimen (InSet)
  file/filtered_out.csv   -> data yang dibuang beserta alasannya (untuk laporan jumlah data)
  file/kandidat_slang.csv -> kata di luar kamus paling sering, bahan menambah kamus slang
"""
import os
import re
from collections import Counter
from multiprocessing import Pool, cpu_count

import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

FILE_INPUT = 'file/raw_text.csv'
FILE_OUTPUT = 'file/cleaned_text.csv'
FILE_BUANG = 'file/filtered_out.csv'
FILE_KANDIDAT_SLANG = 'file/kandidat_slang.csv'
# Kamus slang tambahan (opsional), format CSV: slang,baku
FILE_SLANG_TAMBAHAN = 'file/kamus_slang.csv'

MIN_TOKEN = 4             # F1: minimal jumlah kata setelah normalisasi
MIN_RASIO_INDONESIA = 0.5 # F3: minimal proporsi kata yang dikenali sebagai Bahasa Indonesia
MIN_SKOR_MELAYU = 3       # F3: skor penanda Bahasa Melayu untuk dibuang

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

# ---------------------------------------------------------
# LEKSIKON ASPEK (pelabelan aspek dan filter topik)
# ---------------------------------------------------------
ASPEK_POLA = {
    'ASI Eksklusif': [
        r'\basi\b', r'\bmenyusui\b', r'\bmenyusu\b', r'\bdisusui\b', r'\bdbf\b', r'\bpumping\b',
        r'\blaktasi\b', r'\bsufor\b', r'\bbusui\b', r'\bibu menyusui\b', r'\bpompa asi\b',
    ],
    'Imunisasi': [
        r'\bimunisasi\b', r'\bdiimunisasi\b', r'\bvaksin\w*\b', r'\bdivaksin\b', r'\bkipi\b',
        r'\bbcg\b', r'\bdpt\b', r'\bpolio\b', r'\bcampak\b', r'\brotavirus\b', r'\bpcv\b',
        r'\bhepatitis b\b', r'\bmmr\b',
    ],
    'Posyandu': [
        r'\bposyandu\b', r'\bkms\b', r'\bbuku kia\b', r'\bpmt\b', r'\bpenimbangan balita\b',
        r'\bkader\b',
    ],
}
ASPEK_REGEX = {aspek: re.compile('|'.join(pola)) for aspek, pola in ASPEK_POLA.items()}

# F5: aspek hanya dihitung bila konteksnya memang kesehatan anak/balita
KONTEKS_ANAK = re.compile(
    r'\b(anak|bayi|balita|batita|si kecil|newborn|bunda|ibu|dokter spesialis anak|idai|posyandu|'
    r'puskesmas|bidan|imunisasi|stunting|tumbuh kembang|berat badan|tinggi badan|bulan|usia|umur|'
    r'jadwal|kia|kms|gizi|asi|menyusui)\b'
)
KONTEKS_BUKAN_ANAK = {
    'Imunisasi': re.compile(r'\b(kucing|anjing|hewan|ternak|sapi|kambing|ayam|peliharaan|rabies|meong|anabul)\b'),
    'Posyandu': re.compile(r'\b(kader partai|kader pks|kader pdip|kader gerindra|kader golkar|kader hmi|kader pmii|kaderisasi)\b'),
}
ASI_KUAT = re.compile(r'\b(asi|menyusui anak|menyusui bayi|disusui|dbf|pumping|laktasi|sufor|pompa asi)\b')
PROGRAM_MBG = re.compile(r'\b(mbg|makan bergizi gratis|bgn|sppg)\b')
# "kader" baru dianggap aspek Posyandu bila disertai konteks posyandu/balita
KADER_KONTEKS = re.compile(r'\b(posyandu|balita|bayi|timbang|nimbang|stunting|kms|pmt|ibu hamil)\b')

# ---------------------------------------------------------
# F2: penanda promosi / spam
# ---------------------------------------------------------
POLA_PROMOSI = re.compile(
    r'(https?://|www\.|wa\.me|bit\.ly|linktr|\blink (di )?bio\b|\bklik link\b|\bcek bio\b|'
    r'\bopen (po|order|reseller)\b|\bready stock\b|\bpromo\b|\bdiskon\b|\bcheckout\b|'
    r'\bongkir\b|\bcod\b|\breseller\b|\bdropship\b|\bshopee\b|\btokopedia\b|'
    r'\btiktok ?shop\b|\bkeranjang kuning\b|\bwhatsapp\b|\b(0|\+62)8\d{8,}\b|\border\b sekarang|'
    r'\bdm (aja|ya|kak|untuk)\b|\bslot (gacor|online)\b|\bjudi online\b|\bgacor\b)',
    re.IGNORECASE,
)

# ---------------------------------------------------------
# F3: penanda Bahasa Melayu (banyak muncul pada pencarian "vaksin anak")
# ---------------------------------------------------------
PENANDA_MELAYU_KUAT = {
    'korang', 'sihat', 'pasal', 'kerana', 'sahaja', 'awak', 'budak', 'cakna', 'ambik', 'ibubapa',
    'kanak', 'hospital', 'ubat', 'jangkitan', 'mahu', 'tengok', 'betul2', 'takde', 'xde', 'tak de',
    'kesihatan', 'doktor', 'percuma', 'cucuk', 'takkan', 'cakap', 'mak cik', 'baik pulih', 'kat', 'sangat2', 'dorang', 'diorang', 'macam mana',
}
PENANDA_MELAYU_LEMAH = {'je', 'ni', 'tu', 'tak', 'nak', 'dah', 'macam', 'boleh', 'kena', 'pun', 'lah', 'x'}
FRASA_SLANG = []  # diisi di main(): entri kamus berupa frasa dua kata

BIGRAM_MELAYU = re.compile(r'\b(kesan sampingan|klinik kesihatan|kanak kanak|tak boleh|macam mana|anak anak kecik)\b')


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


def skor_melayu(teks):
    tokens = teks.split()
    skor = sum(2 for t in tokens if t in PENANDA_MELAYU_KUAT)
    skor += sum(1 for t in tokens if t in PENANDA_MELAYU_LEMAH)
    skor += 2 * len(BIGRAM_MELAYU.findall(teks))
    return skor


def deteksi_aspek(teks):
    aspek_ditemukan = []
    alasan_konteks = []
    ada_konteks_anak = bool(KONTEKS_ANAK.search(teks))
    for aspek, regex in ASPEK_REGEX.items():
        if not regex.search(teks):
            continue
        if aspek == 'Posyandu' and not re.search(r'\b(posyandu|kms|buku kia|pmt|penimbangan balita)\b', teks):
            # hanya "kader" yang cocok -> wajib ada konteks posyandu/balita
            if not KADER_KONTEKS.search(teks):
                alasan_konteks.append('kader bukan konteks posyandu')
                continue
        if aspek == 'ASI Eksklusif' and not ASI_KUAT.search(teks) and PROGRAM_MBG.search(teks):
            # "ibu menyusui" hanya sebagai sasaran program MBG, bukan opini tentang ASI
            alasan_konteks.append('ibu menyusui dalam konteks MBG')
            continue
        if aspek in KONTEKS_BUKAN_ANAK and KONTEKS_BUKAN_ANAK[aspek].search(teks):
            alasan_konteks.append(f'{aspek}: konteks non-anak')
            continue
        if aspek == 'Imunisasi' and not ada_konteks_anak:
            alasan_konteks.append('vaksin tanpa konteks anak')
            continue
        aspek_ditemukan.append(aspek)
    return aspek_ditemukan, alasan_konteks


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
        print(f"File '{FILE_INPUT}' tidak ditemukan.")
        return

    print("Menginisialisasi modul Sastrawi...")
    kata_dasar = set(StemmerFactory().get_words())
    stopwords = (set(StopWordRemoverFactory().get_stop_words()) | STOPWORD_TAMBAHAN) - STOPWORD_DIPERTAHANKAN
    kamus = muat_kamus_slang()
    FRASA_SLANG[:] = [(f, b) for f, b in kamus.items() if ' ' in f]
    kosakata_dikenal = kata_dasar | set(kamus) | KATA_DILINDUNGI | stopwords
    for baku in kamus.values():
        kosakata_dikenal.update(baku.split())

    print(f"Total data mentah: {len(df)}")
    df = df.dropna(subset=['content'])

    kandidat, buang = [], []
    teks_terlihat = set()

    # Tahap 1: cleaning, case folding, tokenisasi, normalisasi + filter F1, F2, F3 (Melayu), F4, F5, F6
    for row in df.itertuples(index=False):
        raw = str(row.content)

        def catat_buang(alasan):
            buang.append({'keyword': row.keyword, 'content': raw, 'alasan': alasan})

        if POLA_PROMOSI.search(raw):
            catat_buang('F2 promosi/spam/tautan')
            continue

        clean = clean_text(raw)
        folded = case_folding(clean)
        tokens = normalisasi(tokenisasi(folded), kamus)
        teks_a = ' '.join(tokens)

        if len(tokens) < MIN_TOKEN:
            catat_buang('F1 terlalu pendek')
            continue
        if skor_melayu(folded) >= MIN_SKOR_MELAYU:
            catat_buang('F3 bahasa Melayu')
            continue

        aspek_list, alasan_konteks = deteksi_aspek(teks_a)
        if not aspek_list:
            catat_buang('F5 ' + '; '.join(alasan_konteks) if alasan_konteks else 'F4 tidak membahas aspek')
            continue
        if teks_a in teks_terlihat:
            catat_buang('F6 duplikat')
            continue
        teks_terlihat.add(teks_a)
        kandidat.append((row.keyword, raw, tokens, teks_a, aspek_list))

    # Tahap 2: stemming kosakata unik yang belum berbentuk kata dasar
    kosakata = {t for _, _, tokens, _, _ in kandidat for t in tokens}
    perlu_stem = {t for t in kosakata if t not in kata_dasar and t not in KATA_DILINDUNGI}
    print(f"Stemming {len(perlu_stem)} kata unik secara paralel. Mohon tunggu...")
    hasil_stem = stem_kosakata(perlu_stem)

    def stem(token):
        return hasil_stem.get(token, token)

    # Tahap 3: filter F3 (rasio kata Indonesia), stopword removal, stemming, pelabelan aspek
    lolos = []
    jumlah_komentar = 0
    kata_luar_kamus = Counter()
    for keyword, raw, tokens, teks_a, aspek_list in kandidat:
        dikenal = [t in kosakata_dikenal or stem(t) in kata_dasar for t in tokens]
        if sum(dikenal) / len(tokens) < MIN_RASIO_INDONESIA:
            buang.append({'keyword': keyword, 'content': raw, 'alasan': 'F3 bukan bahasa Indonesia'})
            continue
        kata_luar_kamus.update(t for t, ok in zip(tokens, dikenal) if not ok)

        # Jalur B: stopword removal + stemming
        tokens_b = [stem(t) for t in tokens if t not in stopwords]
        tokens_b = [t for t in tokens_b if len(t) > 1]

        # Pelabelan aspek: digandakan satu baris per aspek
        jumlah_komentar += 1
        for aspek in aspek_list:
            lolos.append({
                'keyword': keyword,
                'aspek': aspek,
                'jumlah_aspek': len(aspek_list),
                'content': raw,
                'text_jalur_a': teks_a,
                'tokens_jalur_b': str(tokens_b),
                'text_jalur_b': ' '.join(tokens_b),
            })

    hasil = pd.DataFrame(lolos)
    hasil.insert(0, 'No', range(1, 1 + len(hasil)))
    hasil.to_csv(FILE_OUTPUT, index=False, sep=';', encoding='utf-8-sig')

    df_buang = pd.DataFrame(buang)
    df_buang.to_csv(FILE_BUANG, index=False, sep=';', encoding='utf-8-sig')

    pd.DataFrame(kata_luar_kamus.most_common(500), columns=['kata', 'frekuensi']).to_csv(
        FILE_KANDIDAT_SLANG, index=False, sep=';', encoding='utf-8-sig')

    print("\n================ RINGKASAN ================")
    print(f"Data mentah            : {len(df)}")
    print(f"Komentar lolos filter  : {jumlah_komentar}")
    print(f"Baris setelah digandakan per aspek: {len(hasil)}")
    print("\nJumlah per aspek:")
    print(hasil['aspek'].value_counts().to_string())
    print("\nData dibuang per alasan:")
    print(df_buang['alasan'].str.split(';').str[0].value_counts().to_string())
    print(f"\n✅ Dataset  : '{FILE_OUTPUT}'")
    print(f"🗑️  Dibuang : '{FILE_BUANG}'")
    print(f"📝 Kandidat slang untuk ditambahkan ke '{FILE_SLANG_TAMBAHAN}': '{FILE_KANDIDAT_SLANG}'")


if __name__ == "__main__":
    main()
