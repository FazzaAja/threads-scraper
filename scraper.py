import csv
import os
import random
import re
import subprocess
import time
from playwright.sync_api import sync_playwright, Error as PlaywrightError

# Selector umum yang muncul saat Threads/Meta menampilkan tantangan captcha/verifikasi
CAPTCHA_SELECTORS = [
    'iframe[src*="captcha"]',
    'iframe[title*="captcha" i]',
    'div[class*="g-recaptcha"]',
    'iframe[src*="hcaptcha"]',
    'text=/verifikasi|confirm you.?re human|suspicious activity|unusual activity/i',
]

CAPTCHA_URL_MARKERS = ['auth_platform', 'challenge', 'recaptcha', 'checkpoint']

CHROME_PATHS = [
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
]

class ChromeSession:
    """Chrome dijalankan sebagai proses biasa; Playwright hanya menyambung (CDP) saat perlu.
    reCAPTCHA Meta tidak mau termuat selama ada koneksi CDP, jadi koneksi dilepas
    setiap kali pengguna harus login atau menyelesaikan captcha."""

    def __init__(self, p, profile_dir, start_url, port=9222):
        chrome_path = next((path for path in CHROME_PATHS if os.path.exists(path)), None)
        if chrome_path is None:
            raise FileNotFoundError("Google Chrome tidak ditemukan. Install Chrome terlebih dahulu.")

        os.makedirs(profile_dir, exist_ok=True)
        self.p = p
        self.port = port
        self.browser = self.context = self.page = None
        self.process = subprocess.Popen([
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={os.path.abspath(profile_dir)}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1280,800",
            start_url,
        ])

    def connect(self):
        # Tunggu sampai port debugging Chrome siap menerima koneksi
        for _ in range(30):
            try:
                self.browser = self.p.chromium.connect_over_cdp(f"http://127.0.0.1:{self.port}")
                self.context = self.browser.contexts[0]
                # Pilih tab Threads; tab lain (mis. halaman login Instagram/Chrome) bisa membuat navigasi gagal
                threads_pages = [pg for pg in self.context.pages if 'threads.' in pg.url]
                self.page = threads_pages[-1] if threads_pages else self.context.new_page()
                self.page.bring_to_front()
                return
            except PlaywrightError:
                time.sleep(1)
        raise RuntimeError("Gagal tersambung ke Chrome. Tutup semua jendela Chrome dari profil skrip lalu coba lagi.")

    def disconnect(self):
        # Untuk koneksi CDP, close() hanya memutus Playwright; jendela Chrome tetap terbuka
        if self.browser is not None:
            self.browser.close()
            self.browser = self.context = self.page = None

    def pause_for_user(self, message):
        """Lepas koneksi agar pengguna bisa berinteraksi dengan Chrome normal, lalu sambung lagi."""
        self.disconnect()
        print(message)
        input()
        self.connect()
        self.page.wait_for_timeout(2000)

    def close(self):
        self.disconnect()
        self.process.terminate()

def captcha_visible(page):
    if any(marker in page.url for marker in CAPTCHA_URL_MARKERS):
        return True
    for selector in CAPTCHA_SELECTORS:
        try:
            if page.locator(selector).first.is_visible(timeout=1000):
                return True
        except PlaywrightError:
            continue
    return False

def wait_if_captcha(session, context_label=""):
    """Jeda skrip dan minta penyelesaian manual jika terdeteksi captcha/verifikasi."""
    if not captcha_visible(session.page):
        return False
    print(f"\n🛑 CAPTCHA/verifikasi terdeteksi{(' saat ' + context_label) if context_label else ''}.")
    session.pause_for_user(
        "Koneksi skrip dilepas sementara. Jika captcha belum muncul, tekan F5 di browser.\n"
        "Selesaikan captcha secara manual, lalu tekan Enter di sini untuk melanjutkan..."
    )
    return True

# Ambil isi postingan saja: hanya kartu yang punya link /post/ (kartu saran profil dilewati),
# lalu pilih span terpanjang yang bukan nama akun (<a>) maupun tanggal (<time>)
EXTRACT_POSTS_JS = """() => {
    const posts = [];
    for (const card of document.querySelectorAll('div[data-pressable-container]')) {
        const link = card.querySelector('a[href*="/post/"]');
        if (!link) continue;
        let body = '';
        for (const span of card.querySelectorAll('span[dir="auto"]')) {
            if (span.closest('a') || span.closest('time')) continue;
            const text = span.innerText.trim();
            if (text.length > body.length) body = text;
        }
        if (body) posts.push({url: link.getAttribute('href').split('?')[0], text: body});
    }
    return posts;
}"""

# Teks tombol/indikator Threads yang ikut terbaca di akhir isi postingan
UI_NOISE_PATTERN = re.compile(r'\b(Translate|Lihat terjemahan|Terjemahkan)\b|\b\d+\s*/\s*\d+\s*$')

def clean_post_text(text):
    text = UI_NOISE_PATTERN.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_logged_in(context):
    """Cek cookie sesi Threads/Instagram untuk memastikan benar-benar sudah login."""
    return any(c['name'] == 'sessionid' and c['value'] for c in context.cookies())

# Tab hasil pencarian Threads; isi "Top" dan "Recent" berbeda sehingga data per keyword bertambah
SEARCH_TABS = {
    'top': '',
    'recent': '&filter=recent',
}
MAX_SCROLL_PER_SEARCH = 300
MAX_STAGNANT_SCROLL = 6

def load_existing_rows(output_file):
    """Baca data hasil run sebelumnya agar scraping bisa dilanjutkan tanpa duplikat."""
    if not os.path.exists(output_file):
        return []
    with open(output_file, newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))

def scrape_multiple_keywords(keywords, target_total, output_file, profile_dir="file/browser_profile"):
    base_url = "https://www.threads.com/"

    existing_rows = load_existing_rows(output_file)
    extracted_texts = {row['content'] for row in existing_rows} # Set global agar tidak ada data duplikat antar keyword
    extracted_urls = set()
    total_collected = len(existing_rows)
    if total_collected:
        print(f"📂 Melanjutkan dari {total_collected} data yang sudah ada di '{output_file}'.")

    # Data langsung ditulis per baris, jadi aman jika skrip terhenti di tengah jalan
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    output = open(output_file, mode='a', newline='', encoding='utf-8')
    writer = csv.DictWriter(output, fieldnames=['keyword', 'content'])
    if not existing_rows:
        writer.writeheader()

    with sync_playwright() as p:
        print("Membuka Sosial Media untuk inisiasi sesi...")
        session = ChromeSession(p, profile_dir, base_url)

        # Playwright belum menyambung, jadi login dan captcha berjalan seperti di Chrome biasa
        print("\n⏳ Jika belum login, silakan LOGIN MANUAL di jendela Chrome (selesaikan captcha jika muncul).")
        print("Setelah berada di beranda Threads (atau jika sudah login), tekan Enter di sini...")
        input()
        session.connect()
        while not is_logged_in(session.context):
            session.pause_for_user("❌ Belum terdeteksi login. Selesaikan login dulu, lalu tekan Enter lagi...")
        print("✅ Login berhasil. Sesi tersimpan di profil browser untuk run berikutnya.")
        page = session.page

        try:
            for keyword in keywords:
                for tab_name, tab_param in SEARCH_TABS.items():
                    if total_collected >= target_total:
                        break

                    print(f"\n==================================================")
                    print(f"🔎 Keyword: '{keyword}' | Tab: {tab_name}")
                    print(f"   (Total saat ini: {total_collected}/{target_total})")
                    print(f"==================================================")

                    search_url = f"https://www.threads.com/search?q={keyword.replace(' ', '+')}{tab_param}"

                    # Memaksa pindah ke halaman pencarian keyword saat ini
                    try:
                        page.goto(search_url, wait_until="domcontentloaded")
                    except PlaywrightError as e:
                        print(f"⚠️ Gagal membuka halaman pencarian: {e}")
                    page.wait_for_timeout(6000)
                    if wait_if_captcha(session, f"pencarian keyword '{keyword}'"):
                        page = session.page

                    # Validasi apakah berada di halaman pencarian
                    if "search" not in page.url:
                        print(f"⚠️ Tidak berada di halaman pencarian ({page.url}), melewati pencarian ini...")
                        continue

                    scroll_attempts = 0
                    search_collected = 0
                    stagnant_count = 0
                    last_collected_count = 0

                    # Looping ekstraksi untuk satu keyword + tab: scroll sampai Threads tidak memuat postingan baru
                    while total_collected < target_total and scroll_attempts < MAX_SCROLL_PER_SEARCH:
                        try:
                            # state='attached': span pertama di Threads tersembunyi, jadi menunggu 'visible' selalu timeout
                            page.wait_for_selector('span[dir="auto"]', state='attached', timeout=5000)

                            for post in page.evaluate(EXTRACT_POSTS_JS):
                                text = clean_post_text(post['text'])
                                # Filter postingan dengan panjang wajar & cek duplikasi global (berdasarkan URL postingan)
                                if len(text) > 20 and post['url'] not in extracted_urls and text not in extracted_texts:
                                    extracted_urls.add(post['url'])
                                    extracted_texts.add(text)
                                    writer.writerow({'keyword': keyword, 'content': text})
                                    search_collected += 1
                                    total_collected += 1
                            output.flush()
                        except PlaywrightError as e:
                            print(f"   ⚠️ Gagal membaca postingan di {page.url}: {str(e).splitlines()[0]}")

                        # Pengecekan jika Threads berhenti memuat data baru (Stagnan)
                        if search_collected == last_collected_count:
                            stagnant_count += 1
                        else:
                            stagnant_count = 0
                            last_collected_count = search_collected

                        if stagnant_count >= MAX_STAGNANT_SCROLL:
                            print(f"⚠️ Tidak ada data baru untuk '{keyword}' ({tab_name}). Pindah ke pencarian selanjutnya...")
                            break

                        try:
                            page.mouse.wheel(0, random.randint(2000, 3500)) # Scroll ke bawah (jarak diacak)
                            page.wait_for_timeout(random.randint(2500, 5000)) # Tunggu loading (durasi diacak)
                        except PlaywrightError:
                            pass

                        if wait_if_captcha(session, f"scroll keyword '{keyword}'"):
                            page = session.page
                            continue

                        scroll_attempts += 1
                        print(f"   -> '{keyword}' ({tab_name}): {search_collected} data | Total Keseluruhan: {total_collected}/{target_total}")
        except KeyboardInterrupt:
            print("\n⏹️ Dihentikan oleh pengguna. Data yang sudah terkumpul tetap tersimpan.")
        finally:
            output.close()
            session.close()

    print(f"\n✅ EKSTRAKSI SELESAI!")
    print(f"Total {total_collected} baris data sentimen tersimpan di '{output_file}'.")

if __name__ == "__main__":
    # Daftar kata kunci yang diinginkan.
    # Satu pencarian Threads hanya memberi ratusan postingan, jadi untuk 20.000 data perlu banyak keyword.
    daftar_keyword = [
        "ASI eksklusif stunting",
        "imunisasi bayi stunting",
        "vaksin anak",
        "vaksin stunting",
        "posyandu stunting",
        "kader posyandu",
        "stunting",
        "cegah stunting",
        "anak stunting",
        "gizi buruk anak",
        "tumbuh kembang anak",
        "MPASI",
        "ASI eksklusif",
        "imunisasi",
        "imunisasi dasar",
        "imunisasi anak",
        "vaksin bayi",
        "KIPI vaksin",
        "anti vaksin",
        "posyandu",
        "posyandu balita",
        "timbang balita",
        "berat badan anak",
        "tinggi badan anak",
        "makan bergizi gratis",
        "MBG balita",
        "puskesmas imunisasi",
        "buku KIA",
        "PMT balita",
        "gizi ibu hamil",
    ]

    scrape_multiple_keywords(
        keywords=daftar_keyword,
        target_total=20000,
        output_file="file/raw_text.csv"
    )
