"""
EKŞİ SÖZLÜK AI - Sıfırdan yazıldı, Railway'de çalışır
- Tüm başlıklar ve tüm entry'ler eksiksiz çekilir
- SQLite ile kalıcı depolama
- TF-IDF ile akıllı arama
- HuggingFace API ile Türkçe yanıt (opsiyonel)
- Gradio web arayüzü
"""

import os
import time
import random
import sqlite3
import threading
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════
#  AYARLAR
# ════════════════════════════════════════════════════════
BASE_URL      = "https://eksisozluk.com"
DB_DOSYASI    = os.environ.get("DB_PATH", "eksi.db")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
PORT          = int(os.environ.get("PORT", 7860))
GECIKME_MIN   = 2.5   # saniye - ban yememek için
GECIKME_MAX   = 5.0
RATE_LIMIT_BK = 90    # 429 gelince bekle (saniye)
MAX_BASLIK_SAYFA = 50 # kaç başlık sayfası dolaşılacak

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.5",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ════════════════════════════════════════════════════════
#  VERİTABANI
# ════════════════════════════════════════════════════════

_db_lock = threading.Lock()

def db_baglan():
    """Her thread için bağımsız bağlantı döner."""
    conn = sqlite3.connect(DB_DOSYASI, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # eş zamanlı yazma için
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def db_olustur():
    """Tablolar yoksa oluşturur."""
    conn = db_baglan()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entryles (
            id         TEXT PRIMARY KEY,
            baslik     TEXT NOT NULL,
            baslik_url TEXT NOT NULL,
            icerik     TEXT NOT NULL,
            yazar      TEXT,
            tarih      TEXT,
            favori     TEXT DEFAULT '0',
            eklendi    TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_baslik ON entryles(baslik);

        CREATE TABLE IF NOT EXISTS basliklar (
            url        TEXT PRIMARY KEY,
            baslik     TEXT NOT NULL,
            tamamlandi INTEGER DEFAULT 0,
            son_sayfa  INTEGER DEFAULT 0,
            guncellendi TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS durum (
            anahtar TEXT PRIMARY KEY,
            deger   TEXT
        );
    """)
    conn.commit()
    conn.close()
    log.info("✅ Veritabanı hazır: %s", DB_DOSYASI)

def durum_oku(anahtar, varsayilan=""):
    conn = db_baglan()
    try:
        r = conn.execute("SELECT deger FROM durum WHERE anahtar=?", (anahtar,)).fetchone()
        return r["deger"] if r else varsayilan
    finally:
        conn.close()

def durum_yaz(anahtar, deger):
    conn = db_baglan()
    try:
        conn.execute("INSERT OR REPLACE INTO durum VALUES (?,?)", (anahtar, str(deger)))
        conn.commit()
    finally:
        conn.close()

def entry_kaydet(entryler: list):
    """Toplu entry kaydet, var olanları atla."""
    if not entryler:
        return 0
    conn = db_baglan()
    try:
        c = conn.executemany(
            """INSERT OR IGNORE INTO entryles
               (id, baslik, baslik_url, icerik, yazar, tarih, favori)
               VALUES (:id, :baslik, :baslik_url, :icerik, :yazar, :tarih, :favori)""",
            entryler,
        )
        conn.commit()
        return c.rowcount
    except Exception as e:
        log.error("entry_kaydet hata: %s", e)
        return 0
    finally:
        conn.close()

def baslik_kaydet(basliklar: list):
    """Başlık kaydeder, var olanı atlar."""
    if not basliklar:
        return
    conn = db_baglan()
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO basliklar (url, baslik) VALUES (:url, :baslik)",
            basliklar,
        )
        conn.commit()
    finally:
        conn.close()

def baslik_tamamla(url: str, son_sayfa: int):
    conn = db_baglan()
    try:
        conn.execute(
            "UPDATE basliklar SET tamamlandi=1, son_sayfa=?, guncellendi=CURRENT_TIMESTAMP WHERE url=?",
            (son_sayfa, url),
        )
        conn.commit()
    finally:
        conn.close()

def bekleyen_basliklar():
    """Henüz tamamlanmamış başlıkları döner."""
    conn = db_baglan()
    try:
        rows = conn.execute(
            "SELECT url, baslik FROM basliklar WHERE tamamlandi=0 ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def toplam_say():
    conn = db_baglan()
    try:
        r = conn.execute("SELECT COUNT(*) as n FROM entryles").fetchone()
        return r["n"]
    finally:
        conn.close()

def baslik_sayisi():
    conn = db_baglan()
    try:
        r = conn.execute("SELECT COUNT(*) as n FROM basliklar").fetchone()
        return r["n"]
    finally:
        conn.close()

def tamamlanan_baslik_sayisi():
    conn = db_baglan()
    try:
        r = conn.execute("SELECT COUNT(*) as n FROM basliklar WHERE tamamlandi=1").fetchone()
        return r["n"]
    finally:
        conn.close()

# ════════════════════════════════════════════════════════
#  GLOBAL DURUM
# ════════════════════════════════════════════════════════
durum_g = {
    "aktif":      False,
    "su_an":      "Bekleniyor...",
    "baslangic":  None,
    "hata_sayac": 0,
}

# ════════════════════════════════════════════════════════
#  HTTP YARDIMCISI
# ════════════════════════════════════════════════════════

def guvvenli_get(url: str, deneme=3) -> object:
    """Rate limit ve geçici hatalara dayanıklı GET."""
    for i in range(deneme):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                bk = RATE_LIMIT_BK * (i + 1)
                log.warning("⚠️  Rate limit! %ds bekleniyor...", bk)
                time.sleep(bk)
                continue
            if r.status_code in (404, 410):
                return None   # sayfa yok, geç
            log.warning("HTTP %d: %s", r.status_code, url)
            time.sleep(5 * (i + 1))
        except requests.RequestException as e:
            log.warning("İstek hatası (%d/%d): %s", i + 1, deneme, e)
            time.sleep(10 * (i + 1))
    return None

# ════════════════════════════════════════════════════════
#  EKŞİ TARAYICI
# ════════════════════════════════════════════════════════

BASLIK_ENDPOINTLER = [
    "/basliklar/bugun",
    "/basliklar/populer",
    "/basliklar/gundem",
    "/basliklar/kanal",
    "/son-entryleri",
]

def tum_basliklar_cek() -> list:
    """
    Birden fazla endpoint ve birden fazla sayfa dolaşarak
    olabildiğince çok başlık toplar.
    """
    toplanan = {}   # url -> baslik (duplikat önle)

    for endpoint in BASLIK_ENDPOINTLER:
        for sayfa in range(1, MAX_BASLIK_SAYFA + 1):
            url = f"{BASE_URL}{endpoint}?p={sayfa}"
            r = guvvenli_get(url)
            if not r:
                break

            soup = BeautifulSoup(r.text, "html.parser")

            # Ana liste
            items = soup.select("ul.topic-list li a, ul.partial-topic-list li a")
            if not items:
                break  # bu endpoint bitti

            sayfa_bos = True
            for item in items:
                href   = item.get("href", "").strip()
                baslik = item.get_text(strip=True)

                # Sadece başlık linkleri (entry linki değil)
                if not href or not baslik:
                    continue
                if "/entry/" in href:
                    continue
                if href.startswith("/?q="):
                    continue

                # Tam URL yap
                if href.startswith("/"):
                    tam_url = BASE_URL + href
                elif href.startswith("http"):
                    tam_url = href
                else:
                    continue

                # Sayfa numarasını temizle (başlığın ilk sayfasını al)
                if "?p=" in tam_url:
                    tam_url = tam_url.split("?p=")[0]

                if tam_url not in toplanan:
                    toplanan[tam_url] = baslik
                    sayfa_bos = False

            if sayfa_bos:
                break

            time.sleep(1.5)

    sonuc = [{"url": u, "baslik": b} for u, b in toplanan.items()]
    log.info("📚 Toplam %d başlık bulundu", len(sonuc))
    return sonuc


def entry_cek_tek_sayfa(url: str, sayfa: int) -> tuple[list, bool]:
    """
    Bir başlığın belirtilen sayfasındaki entry'leri çeker.
    Döner: (entry_listesi, sonraki_sayfa_var_mi)
    """
    r = guvvenli_get(f"{url}?p={sayfa}")
    if not r:
        return [], False

    soup = BeautifulSoup(r.text, "html.parser")

    # Başlık adını al
    baslik_el = soup.select_one("h1#title a, h1#title")
    baslik = baslik_el.get_text(strip=True) if baslik_el else ""
    if not baslik:
        # URL'den başlık çıkar
        parca = url.rstrip("/").split("/")[-1]
        baslik = parca.replace("-", " ")

    entryler = []
    for el in soup.select("ul#entry-item-list li[data-id]"):
        eid = el.get("data-id", "").strip()
        if not eid:
            continue

        yazar  = el.get("data-author", "").strip()
        ic_el  = el.select_one("div.content")
        icerik = ic_el.get_text(" ", strip=True) if ic_el else ""

        if len(icerik) < 10:
            continue   # boş/çok kısa entry'leri atla

        t_el   = el.select_one("footer .entry-date")
        tarih  = t_el.get_text(strip=True) if t_el else ""

        f_el   = el.select_one("footer .favorite-count, footer .rate")
        favori = f_el.get_text(strip=True) if f_el else "0"

        entryler.append({
            "id":        eid,
            "baslik":    baslik,
            "baslik_url": url,
            "icerik":    icerik,
            "yazar":     yazar,
            "tarih":     tarih,
            "favori":    favori,
        })

    # Sonraki sayfa var mı?
    sonraki = bool(
        soup.select_one("div.pager a[rel='next']") or
        soup.select_one(".pager .next")
    )

    return entryler, sonraki


def baslik_tum_entryleri_cek(url: str, baslik: str) -> int:
    """
    Bir başlığın TÜM sayfalarını dolaşarak tüm entry'leri çeker.
    Kaç entry kaydedildiğini döner.
    """
    toplam = 0
    sayfa  = 1

    while True:
        durum_g["su_an"] = f"'{baslik[:35]}...' — sayfa {sayfa}"
        entryler, sonraki = entry_cek_tek_sayfa(url, sayfa)

        if entryler:
            kaydedilen = entry_kaydet(entryler)
            toplam += kaydedilen

        if not sonraki:
            break

        sayfa += 1
        time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))

    return toplam


def tarama_thread():
    """
    Ana tarama döngüsü — daemon thread olarak çalışır.
    1. Tüm başlıkları topla → kaydet
    2. Bekleyen başlıkları sırayla işle
    3. Hepsini bitirince yeniden başlar
    """
    global durum_g
    durum_g["aktif"]     = True
    durum_g["baslangic"] = datetime.now()

    turno = 0
    while True:
        turno += 1
        log.info("🔄 Tarama turu %d başlıyor...", turno)
        durum_g["su_an"] = f"Tur {turno} — başlıklar toplanıyor..."

        # --- 1. Başlıkları topla ---
        yeni_basliklar = tum_basliklar_cek()
        baslik_kaydet(yeni_basliklar)

        # --- 2. Bekleyen başlıkları işle ---
        bekleyenler = bekleyen_basliklar()
        log.info("📋 %d bekleyen başlık var", len(bekleyenler))

        for b in bekleyenler:
            try:
                n = baslik_tum_entryleri_cek(b["url"], b["baslik"])
                baslik_tamamla(b["url"], 0)
                if n > 0:
                    log.info("✅ '%s' → %d yeni entry", b["baslik"][:40], n)
            except Exception as e:
                log.error("Başlık hata '%s': %s", b["baslik"][:40], e)
                durum_g["hata_sayac"] += 1
                time.sleep(15)

            time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))

        log.info("✅ Tur %d tamamlandı. Bekleniyor...", turno)
        durum_g["su_an"] = f"Tur {turno} tamamlandı, yeni tur için bekleniyor..."
        time.sleep(600)   # 10 dakika bekle, sonra tekrar başla


# ════════════════════════════════════════════════════════
#  TF-IDF ARAMA
# ════════════════════════════════════════════════════════
_vec   = None
_mat   = None
_idler = []
_vlock = threading.Lock()


def tfidf_guncelle_thread():
    """Her 3 dakikada TF-IDF matrisini yeniler."""
    global _vec, _mat, _idler
    while True:
        try:
            conn  = db_baglan()
            satirlar = conn.execute(
                """SELECT id, baslik, icerik
                   FROM entryles
                   ORDER BY rowid DESC
                   LIMIT 50000"""
            ).fetchall()
            conn.close()

            if len(satirlar) < 3:
                time.sleep(30)
                continue

            metinler = [
                f"{r['baslik']} {r['icerik'][:400]}"
                for r in satirlar
            ]
            idler = [r["id"] for r in satirlar]

            v   = TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
            mat = v.fit_transform(metinler)

            with _vlock:
                _vec   = v
                _mat   = mat
                _idler = idler

            log.info("🔍 TF-IDF güncellendi: %d entry", len(idler))
        except Exception as e:
            log.error("TF-IDF hata: %s", e)

        time.sleep(180)   # 3 dakika


def benzer_bul(soru: str, kac: int = 7) -> list:
    with _vlock:
        if _vec is None or _mat is None:
            return []
        try:
            sv     = _vec.transform([soru])
            skorlar = cosine_similarity(sv, _mat).flatten()
            en_iyi  = np.argsort(skorlar)[::-1][:kac * 2]

            sonuclar = []
            conn = db_baglan()
            for idx in en_iyi:
                if skorlar[idx] < 0.03:
                    break
                if idx >= len(_idler):
                    continue
                r = conn.execute(
                    "SELECT * FROM entryles WHERE id=?", (_idler[idx],)
                ).fetchone()
                if r:
                    sonuclar.append(dict(r))
                if len(sonuclar) >= kac:
                    break
            conn.close()
            return sonuclar
        except Exception as e:
            log.error("benzer_bul hata: %s", e)
            return []


# ════════════════════════════════════════════════════════
#  YANIT ÜRETİCİ
# ════════════════════════════════════════════════════════

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def hf_yanitla(soru: str, entryler: list, gecmis: list) -> str:
    if not HF_TOKEN:
        return lokal_yanitla(soru, entryler)

    baglam = "\n\n".join(
        f"[{i+1}] {e['baslik']}: {e['icerik'][:300]}"
        for i, e in enumerate(entryler[:5])
    ) if entryler else "İlgili entry bulunamadı."

    gecmis_str = ""
    for g in gecmis[-3:]:
        if isinstance(g, (list, tuple)) and len(g) == 2 and g[1]:
            gecmis_str += f"Kullanıcı: {g[0]}\nBot: {g[1]}\n"

    prompt = (
        "<s>[INST] Sen Ekşi Sözlük tarzında, samimi, esprili ve zaman zaman "
        "alaycı bir Türkçe asistansın. Cevapların kısa ve öz olsun.\n\n"
        f"İlgili Ekşi entry'leri:\n{baglam}\n\n"
        f"{gecmis_str}"
        f"Kullanıcı: {soru} [/INST]"
    )

    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens":   400,
                    "temperature":      0.75,
                    "top_p":            0.9,
                    "return_full_text": False,
                },
            },
            timeout=60,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                yanit = data[0].get("generated_text", "").strip()
                if yanit:
                    return yanit
        elif r.status_code == 503:
            return "⏳ Model ısınıyor, 20 saniye bekleyip tekrar dene."
        log.warning("HF API durumu %d", r.status_code)
    except Exception as e:
        log.error("HF API hata: %s", e)

    return lokal_yanitla(soru, entryler)


def lokal_yanitla(soru: str, entryler: list) -> str:
    """HF yokken veya hata olunca entry'leri doğrudan göster."""
    if not entryler:
        return (
            "Bu konuda henüz yeterli entry yok. "
            "Tarama devam ediyor, biraz bekleyip tekrar dene 🔄"
        )

    satirlar = [f"**{entryler[0]['baslik']}** başlığından benzer entry'ler:\n"]
    for e in entryler[:3]:
        icerik = e["icerik"][:400]
        if len(e["icerik"]) > 400:
            icerik += "..."
        satirlar.append(
            f"> {icerik}\n"
            f"> *— {e['yazar'] or 'anonim'}, {e['tarih'] or '?'} · {e['favori']} fav*\n"
        )

    return "\n".join(satirlar)


# ════════════════════════════════════════════════════════
#  DURUM METNİ
# ════════════════════════════════════════════════════════

def sayac_metni() -> str:
    sure_str = ""
    if durum_g["baslangic"]:
        fark = datetime.now() - durum_g["baslangic"]
        saat = int(fark.total_seconds() // 3600)
        dk   = int((fark.total_seconds() % 3600) // 60)
        sure_str = f"\n\n⏱ **Süre:** {saat}s {dk}dk"

    return (
        f"### 📊 Tarama Durumu\n"
        f"**Entry:** {toplam_say():,}\n\n"
        f"**Başlık:** {baslik_sayisi():,}\n\n"
        f"**Tamamlanan:** {tamamlanan_baslik_sayisi():,}\n\n"
        f"**Aranabilir:** {len(_idler):,}\n\n"
        f"**Şu an:** {durum_g['su_an']}\n\n"
        f"**Hata:** {durum_g['hata_sayac']}\n\n"
        f"**Durum:** {'🟢 Aktif' if durum_g['aktif'] else '🔴 Durdu'}"
        f"{sure_str}"
    )


# ════════════════════════════════════════════════════════
#  SOHBET FONKSİYONU
# ════════════════════════════════════════════════════════

def sohbet(mesaj: str, gecmis: list):
    if not mesaj or not mesaj.strip():
        return "", gecmis or []

    gecmis   = gecmis or []
    entryler = benzer_bul(mesaj)
    yanit    = hf_yanitla(mesaj, entryler, gecmis)
    gecmis   = gecmis + [[mesaj, yanit]]
    return "", gecmis


# ════════════════════════════════════════════════════════
#  GRADIO ARAYÜZÜ
# ════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 960px !important; margin: auto !important; }
footer { display: none !important; }
.chatbot .message { font-size: 0.95rem; }
"""

with gr.Blocks(
    title="🍋 Ekşi Sözlük AI",
    theme=gr.themes.Soft(primary_hue="lime"),
    css=CSS,
) as demo:

    gr.Markdown(
        "# 🍋 Ekşi Sözlük AI\n"
        "Ekşi Sözlük'teki tüm başlıkları ve entry'leri öğrenen sohbet botu. "
        "Tarama arka planda sürekli çalışır."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, show_label=False, bubble_full_width=False)
            with gr.Row():
                mesaj_box = gr.Textbox(
                    placeholder="Bir şey sor… (Enter = gönder)",
                    scale=5,
                    show_label=False,
                    container=False,
                )
                gonder_btn = gr.Button("Gönder 🚀", variant="primary", scale=1)
            temizle_btn = gr.Button("🗑️ Sohbeti Temizle", size="sm")

        with gr.Column(scale=1):
            sayac_md   = gr.Markdown(sayac_metni())
            yenile_btn = gr.Button("🔄 Yenile", size="sm")
            gr.Markdown(
                "---\n**Nasıl çalışır?**\n\n"
                "1. Ekşi'yi baştan sona tarar\n"
                "2. Tüm başlık + entry'leri saklar\n"
                "3. TF-IDF ile benzer entry bulur\n"
                "4. Ekşi ruhuyla yanıt verir\n\n"
                "---\n**HF_TOKEN** env değişkeni ile\n"
                "Mistral-7B destekli yanıt aktif olur."
            )

    # Event bağlantıları
    mesaj_box.submit(
        fn=sohbet,
        inputs=[mesaj_box, chatbot],
        outputs=[mesaj_box, chatbot],
    )
    gonder_btn.click(
        fn=sohbet,
        inputs=[mesaj_box, chatbot],
        outputs=[mesaj_box, chatbot],
    )
    temizle_btn.click(
        fn=lambda: ("", []),
        inputs=None,
        outputs=[mesaj_box, chatbot],
    )
    yenile_btn.click(
        fn=sayac_metni,
        inputs=None,
        outputs=[sayac_md],
    )


# ════════════════════════════════════════════════════════
#  BAŞLATMA
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Veritabanını hazırla
    db_olustur()

    # Arka plan thread'leri başlat
    threading.Thread(target=tarama_thread,        daemon=True, name="tarayici").start()
    threading.Thread(target=tfidf_guncelle_thread, daemon=True, name="tfidf").start()

    log.info("🌐 Sunucu başlıyor... port=%d", PORT)
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
        max_threads=40,
    )
