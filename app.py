"""
EKŞİ SÖZLÜK AI
- Tüm başlıkları ve entry'leri çeker
- SQLite ile saklar
- TF-IDF ile arama
- Gradio 5 web arayüzü
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

# ── LOGGING ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── AYARLAR ──────────────────────────────────────────────
BASE_URL         = "https://eksisozluk.com"
DB_DOSYASI       = os.environ.get("DB_PATH", "eksi.db")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
PORT             = int(os.environ.get("PORT", 7860))
GECIKME_MIN      = 2.5
GECIKME_MAX      = 5.0
RATE_LIMIT_BEKLE = 90
MAX_BASLIK_SAYFA = 50

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.5",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── VERİTABANI ───────────────────────────────────────────

def db_baglan():
    conn = sqlite3.connect(DB_DOSYASI, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_olustur():
    conn = db_baglan()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entryles (
            id          TEXT PRIMARY KEY,
            baslik      TEXT NOT NULL,
            baslik_url  TEXT NOT NULL,
            icerik      TEXT NOT NULL,
            yazar       TEXT,
            tarih       TEXT,
            favori      TEXT DEFAULT '0',
            eklendi     TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_baslik ON entryles(baslik);

        CREATE TABLE IF NOT EXISTS basliklar (
            url         TEXT PRIMARY KEY,
            baslik      TEXT NOT NULL,
            tamamlandi  INTEGER DEFAULT 0,
            guncellendi TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    log.info("Veritabani hazir: %s", DB_DOSYASI)


def _say(sql):
    try:
        conn = db_baglan()
        r = conn.execute(sql).fetchone()
        conn.close()
        return r[0] if r else 0
    except Exception:
        return 0


def toplam_say():
    return _say("SELECT COUNT(*) FROM entryles")


def baslik_sayisi():
    return _say("SELECT COUNT(*) FROM basliklar")


def tamamlanan_sayisi():
    return _say("SELECT COUNT(*) FROM basliklar WHERE tamamlandi=1")


def entry_kaydet(entryler):
    if not entryler:
        return 0
    try:
        conn = db_baglan()
        c = conn.executemany(
            """INSERT OR IGNORE INTO entryles
               (id, baslik, baslik_url, icerik, yazar, tarih, favori)
               VALUES (:id, :baslik, :baslik_url, :icerik, :yazar, :tarih, :favori)""",
            entryler,
        )
        conn.commit()
        n = c.rowcount
        conn.close()
        return n
    except Exception as e:
        log.error("entry_kaydet hata: %s", e)
        return 0


def baslik_kaydet(basliklar):
    if not basliklar:
        return
    try:
        conn = db_baglan()
        conn.executemany(
            "INSERT OR IGNORE INTO basliklar (url, baslik) VALUES (:url, :baslik)",
            basliklar,
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("baslik_kaydet hata: %s", e)


def baslik_tamamla(url):
    try:
        conn = db_baglan()
        conn.execute(
            "UPDATE basliklar SET tamamlandi=1, guncellendi=CURRENT_TIMESTAMP WHERE url=?",
            (url,),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("baslik_tamamla hata: %s", e)


def bekleyen_basliklar():
    try:
        conn = db_baglan()
        rows = conn.execute(
            "SELECT url, baslik FROM basliklar WHERE tamamlandi=0 ORDER BY rowid"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ── GLOBAL DURUM ─────────────────────────────────────────
durum_g = {
    "aktif":     False,
    "su_an":     "Baslatiliyor...",
    "baslangic": None,
    "hata":      0,
}

# ── HTTP ─────────────────────────────────────────────────

def guvenli_get(url, deneme=3):
    for i in range(deneme):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                bk = RATE_LIMIT_BEKLE * (i + 1)
                log.warning("Rate limit! %ds bekleniyor...", bk)
                time.sleep(bk)
                continue
            if r.status_code in (404, 410):
                return None
            time.sleep(5 * (i + 1))
        except requests.RequestException as e:
            log.warning("Istek hatasi (%d/%d): %s", i + 1, deneme, e)
            time.sleep(10 * (i + 1))
    return None


# ── TARAYICI ─────────────────────────────────────────────

ENDPOINTLER = [
    "/basliklar/bugun",
    "/basliklar/populer",
    "/basliklar/gundem",
    "/basliklar/kanal",
]


def tum_basliklar_cek():
    toplanan = {}
    for ep in ENDPOINTLER:
        for sayfa in range(1, MAX_BASLIK_SAYFA + 1):
            r = guvenli_get(f"{BASE_URL}{ep}?p={sayfa}")
            if not r:
                break
            soup  = BeautifulSoup(r.text, "html.parser")
            items = soup.select("ul.topic-list li a, ul.partial-topic-list li a")
            if not items:
                break
            bos = True
            for item in items:
                href   = item.get("href", "").strip()
                baslik = item.get_text(strip=True)
                if not href or not baslik:
                    continue
                if "/entry/" in href or href.startswith("/?q="):
                    continue
                if href.startswith("/"):
                    tam = BASE_URL + href
                elif href.startswith("http"):
                    tam = href
                else:
                    continue
                if "?p=" in tam:
                    tam = tam.split("?p=")[0]
                if tam not in toplanan:
                    toplanan[tam] = baslik
                    bos = False
            if bos:
                break
            time.sleep(1.5)

    sonuc = [{"url": u, "baslik": b} for u, b in toplanan.items()]
    log.info("%d baslik bulundu", len(sonuc))
    return sonuc


def entry_cek_sayfa(url, sayfa):
    r = guvenli_get(f"{url}?p={sayfa}")
    if not r:
        return [], False

    soup   = BeautifulSoup(r.text, "html.parser")
    bel    = soup.select_one("h1#title a, h1#title")
    baslik = bel.get_text(strip=True) if bel else url.rstrip("/").split("/")[-1]
    entryler = []

    for el in soup.select("ul#entry-item-list li[data-id]"):
        eid    = el.get("data-id", "").strip()
        if not eid:
            continue
        yazar  = el.get("data-author", "").strip()
        ic     = el.select_one("div.content")
        icerik = ic.get_text(" ", strip=True) if ic else ""
        if len(icerik) < 10:
            continue
        t_el   = el.select_one("footer .entry-date")
        tarih  = t_el.get_text(strip=True) if t_el else ""
        f_el   = el.select_one("footer .favorite-count, footer .rate")
        favori = f_el.get_text(strip=True) if f_el else "0"
        entryler.append({
            "id": eid, "baslik": baslik, "baslik_url": url,
            "icerik": icerik, "yazar": yazar,
            "tarih": tarih, "favori": favori,
        })

    sonraki = bool(
        soup.select_one("div.pager a[rel='next']") or
        soup.select_one(".pager .next")
    )
    return entryler, sonraki


def baslik_isle(url, baslik):
    sayfa  = 1
    toplam = 0
    while True:
        durum_g["su_an"] = f"'{baslik[:35]}' sayfa {sayfa}"
        entryler, sonraki = entry_cek_sayfa(url, sayfa)
        if entryler:
            toplam += entry_kaydet(entryler)
        if not sonraki:
            break
        sayfa += 1
        time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))
    return toplam


def tarama_thread():
    durum_g["aktif"]     = True
    durum_g["baslangic"] = datetime.now()
    tur = 0
    while True:
        tur += 1
        log.info("Tur %d basliyor...", tur)
        durum_g["su_an"] = f"Tur {tur} basliklar toplanıyor..."
        yeni = tum_basliklar_cek()
        baslik_kaydet(yeni)
        bekleyenler = bekleyen_basliklar()
        log.info("%d bekleyen baslik", len(bekleyenler))
        for b in bekleyenler:
            try:
                n = baslik_isle(b["url"], b["baslik"])
                baslik_tamamla(b["url"])
                if n > 0:
                    log.info("'%s' -> %d entry", b["baslik"][:40], n)
            except Exception as e:
                log.error("Hata '%s': %s", b["baslik"][:40], e)
                durum_g["hata"] += 1
                time.sleep(15)
            time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))
        durum_g["su_an"] = f"Tur {tur} bitti, bekleniyor..."
        time.sleep(600)


# ── TF-IDF ───────────────────────────────────────────────
_vec   = None
_mat   = None
_idler = []
_vlock = threading.Lock()


def tfidf_thread():
    global _vec, _mat, _idler
    while True:
        try:
            conn = db_baglan()
            rows = conn.execute(
                "SELECT id, baslik, icerik FROM entryles ORDER BY rowid DESC LIMIT 50000"
            ).fetchall()
            conn.close()
            if len(rows) < 3:
                time.sleep(30)
                continue
            metinler = [f"{r['baslik']} {r['icerik'][:400]}" for r in rows]
            idler    = [r["id"] for r in rows]
            v   = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=1)
            mat = v.fit_transform(metinler)
            with _vlock:
                _vec   = v
                _mat   = mat
                _idler = idler
            log.info("TF-IDF: %d entry", len(idler))
        except Exception as e:
            log.error("TF-IDF hata: %s", e)
        time.sleep(180)


def benzer_bul(soru, kac=7):
    with _vlock:
        if _vec is None or _mat is None:
            return []
        try:
            sv      = _vec.transform([soru])
            skorlar = cosine_similarity(sv, _mat).flatten()
            en_iyi  = np.argsort(skorlar)[::-1][:kac * 2]
            sonuc   = []
            conn    = db_baglan()
            for idx in en_iyi:
                if skorlar[idx] < 0.03:
                    break
                if idx >= len(_idler):
                    continue
                r = conn.execute(
                    "SELECT * FROM entryles WHERE id=?", (_idler[idx],)
                ).fetchone()
                if r:
                    sonuc.append(dict(r))
                if len(sonuc) >= kac:
                    break
            conn.close()
            return sonuc
        except Exception as e:
            log.error("benzer_bul: %s", e)
            return []


# ── YANIT ────────────────────────────────────────────────
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def hf_yanitla(soru, entryler, gecmis):
    if not HF_TOKEN:
        return lokal_yanitla(soru, entryler)

    baglam = "\n\n".join(
        f"[{i+1}] {e['baslik']}: {e['icerik'][:300]}"
        for i, e in enumerate(entryler[:5])
    ) if entryler else "Ilgili entry bulunamadi."

    gecmis_str = ""
    for g in gecmis[-3:]:
        if isinstance(g, dict):
            rol = g.get("role", "")
            icerik = g.get("content", "")
            if rol == "user":
                gecmis_str += f"Kullanici: {icerik}\n"
            elif rol == "assistant":
                gecmis_str += f"Bot: {icerik}\n"

    prompt = (
        "<s>[INST] Sen Eksi Sozluk tarzinda, samimi ve esprili Turkce asistansin.\n\n"
        f"Ilgili entry'ler:\n{baglam}\n\n"
        f"{gecmis_str}"
        f"Kullanici: {soru} [/INST]"
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
                    "max_new_tokens": 400,
                    "temperature": 0.75,
                    "top_p": 0.9,
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
            return "Model isinuyor, 20 saniye sonra tekrar dene."
    except Exception as e:
        log.error("HF API: %s", e)

    return lokal_yanitla(soru, entryler)


def lokal_yanitla(soru, entryler):
    if not entryler:
        return "Bu konuda henuz entry yok, tarama devam ediyor."
    satirlar = [f"**{entryler[0]['baslik']}** basligindan:\n"]
    for e in entryler[:3]:
        ic = e["icerik"][:400] + ("..." if len(e["icerik"]) > 400 else "")
        satirlar.append(
            f"> {ic}\n"
            f"> *- {e['yazar'] or 'anonim'}, {e['tarih'] or '?'} - {e['favori']} fav*\n"
        )
    return "\n".join(satirlar)


# ── DURUM METNİ ──────────────────────────────────────────

def sayac_metni():
    sure_str = ""
    if durum_g["baslangic"]:
        fark = datetime.now() - durum_g["baslangic"]
        s    = int(fark.total_seconds() // 3600)
        dk   = int((fark.total_seconds() % 3600) // 60)
        sure_str = f"\n\n Sure: {s}s {dk}dk"

    return (
        "### Tarama Durumu\n"
        f"**Entry:** {toplam_say():,}\n\n"
        f"**Baslik:** {baslik_sayisi():,}\n\n"
        f"**Tamamlanan:** {tamamlanan_sayisi():,}\n\n"
        f"**Aranabilir:** {len(_idler):,}\n\n"
        f"**Su an:** {durum_g['su_an']}\n\n"
        f"**Hata:** {durum_g['hata']}\n\n"
        f"**Durum:** {'Aktif' if durum_g['aktif'] else 'Durdu'}"
        f"{sure_str}"
    )


# ── SOHBET ───────────────────────────────────────────────

def sohbet(mesaj, gecmis):
    if not mesaj or not mesaj.strip():
        return "", gecmis or []
    gecmis   = gecmis or []
    entryler = benzer_bul(mesaj)
    yanit    = hf_yanitla(mesaj, entryler, gecmis)
    gecmis   = gecmis + [
        {"role": "user",      "content": mesaj},
        {"role": "assistant", "content": yanit},
    ]
    return "", gecmis


# ── VERİTABANINI BURADA OLUŞTUR (import sırasında) ───────
db_olustur()

# ── GRADIO ───────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 960px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Eksi Sozluk AI",
    theme=gr.themes.Soft(primary_hue="lime"),
    css=CSS,
) as demo:

    gr.Markdown(
        "# Eksi Sozluk AI\n"
        "Eksi Sozluk entry'lerinden ogrenen sohbet botu. "
        "Tarama arka planda sureklI devam eder."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                type="messages",
            )
            with gr.Row():
                mesaj_box = gr.Textbox(
                    placeholder="Bir sey sor... (Enter = gonder)",
                    scale=5,
                    show_label=False,
                    container=False,
                )
                gonder_btn = gr.Button("Gonder", variant="primary", scale=1)
            temizle_btn = gr.Button("Temizle", size="sm")

        with gr.Column(scale=1):
            sayac_md   = gr.Markdown(sayac_metni())
            yenile_btn = gr.Button("Yenile", size="sm")
            gr.Markdown(
                "---\n**Nasil calisir?**\n\n"
                "1. Eksi'yi tarar\n"
                "2. Entry'leri saklar\n"
                "3. TF-IDF ile arar\n"
                "4. Yanit uretir\n\n"
                "---\nHF_TOKEN env ile\nMistral-7B aktif olur."
            )

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


# ── BAŞLATMA ─────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=tarama_thread, daemon=True, name="tarayici").start()
    threading.Thread(target=tfidf_thread,  daemon=True, name="tfidf").start()

    log.info("Sunucu basliyor... port=%d", PORT)
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
        max_threads=40,
    )
