"""
TÜRKÇE AI - Kendi kendini eğiten Türkçe sohbet botu
- HuggingFace'den Türkçe veri çeker (Wikipedia, haberler, kitaplar)
- TF-IDF tabanlı akıllı arama
- HuggingFace Inference API ile Türkçe GPT-2 yanıt
- Gradio web arayüzü - Railway'de çalışır
"""

import os
import time
import sqlite3
import threading
import logging
import re
from datetime import datetime

import requests
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
DB_DOSYASI = os.environ.get("DB_PATH", "turkce_ai.db")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
PORT       = int(os.environ.get("PORT", 7860))

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "User-Agent": "datasets/2.14.0",
}

# Türkçe veri kaynakları - HuggingFace dataset server (engelsiz)
VERI_KAYNAKLARI = [
    # (dataset_id, config, split, max_satir, aciklama)
    ("wikipedia",           "20231101.tr", "train", 100000, "Türkçe Wikipedia"),
    ("mc4",                 "tr",          "train", 200000, "Türkçe web metinleri"),
    ("cc100",               "tr",          "train", 200000, "Türkçe Common Crawl"),
    ("Helsinki-NLP/opus_books", "tr-en",   "train", 50000,  "Türkçe kitaplar"),
    ("emre/turkish-text-news",  None,      "train", 100000, "Türkçe haberler"),
]

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
        CREATE TABLE IF NOT EXISTS metinler (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            kaynak   TEXT NOT NULL,
            baslik   TEXT DEFAULT '',
            icerik   TEXT NOT NULL,
            eklendi  TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_kaynak ON metinler(kaynak);

        CREATE TABLE IF NOT EXISTS egitim_log (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            kaynak   TEXT,
            satir    INTEGER DEFAULT 0,
            zaman    TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sohbet_gecmis (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            soru     TEXT,
            yanit    TEXT,
            zaman    TEXT DEFAULT CURRENT_TIMESTAMP
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
    return _say("SELECT COUNT(*) FROM metinler")


def toplu_kaydet(satirlar):
    if not satirlar:
        return 0
    conn = db_baglan()
    try:
        c = conn.executemany(
            "INSERT INTO metinler (kaynak, baslik, icerik) VALUES (:kaynak, :baslik, :icerik)",
            satirlar,
        )
        conn.commit()
        return c.rowcount
    except Exception as e:
        log.error("toplu_kaydet: %s", e)
        return 0
    finally:
        conn.close()


def egitim_kaydet(kaynak, satir):
    conn = db_baglan()
    try:
        conn.execute(
            "INSERT INTO egitim_log (kaynak, satir) VALUES (?,?)", (kaynak, satir)
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def kaynak_tamamlandi_mi(kaynak):
    conn = db_baglan()
    try:
        r = conn.execute(
            "SELECT COUNT(*) FROM egitim_log WHERE kaynak=?", (kaynak,)
        ).fetchone()
        conn.close()
        return r[0] > 0
    except Exception:
        return False


def sohbet_kaydet(soru, yanit):
    conn = db_baglan()
    try:
        conn.execute(
            "INSERT INTO sohbet_gecmis (soru, yanit) VALUES (?,?)", (soru, yanit)
        )
        conn.commit()
    finally:
        conn.close()


# ── GLOBAL DURUM ─────────────────────────────────────────
durum_g = {
    "aktif":     False,
    "su_an":     "Baslatiliyor...",
    "baslangic": None,
    "hata":      0,
    "indirilen": 0,
}

# ── HF DATASET İNDİRİCİ ──────────────────────────────────

def hf_rows(dataset_id, config, split, offset, length=100):
    """HuggingFace dataset server'dan satir cek."""
    params = {
        "dataset": dataset_id,
        "split":   split,
        "offset":  offset,
        "length":  length,
    }
    if config:
        params["config"] = config

    try:
        r = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params=params,
            headers=HF_HEADERS,
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()

        # 422: config/split hatasi - dataset info'dan dogru config bul
        if r.status_code == 422:
            info = hf_info(dataset_id)
            if info:
                for cfg_name in info.get("dataset_info", {}).keys():
                    splits = info["dataset_info"][cfg_name].get("splits", {})
                    for sp in splits.keys():
                        params["config"] = cfg_name
                        params["split"]  = sp
                        r2 = requests.get(
                            "https://datasets-server.huggingface.co/rows",
                            params=params,
                            headers=HF_HEADERS,
                            timeout=30,
                        )
                        if r2.status_code == 200:
                            return r2.json()

        log.warning("HF rows HTTP %d: %s", r.status_code, dataset_id)
        return None
    except Exception as e:
        log.error("hf_rows hata: %s", e)
        return None


def hf_info(dataset_id):
    try:
        r = requests.get(
            f"https://datasets-server.huggingface.co/info?dataset={dataset_id}",
            headers=HF_HEADERS,
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def metin_temizle(metin):
    """Ham metni temizle."""
    if not metin:
        return ""
    metin = str(metin)
    # HTML tag temizle
    metin = re.sub(r"<[^>]+>", " ", metin)
    # Cok fazla bosluk temizle
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


def satir_parse(row, kaynak):
    """HF row'undan metin ve baslik cikar."""
    # Icerik sutun isimleri
    icerik_keys = ["text", "content", "passage", "article", "body",
                   "translation", "sentence", "paragraph"]
    baslik_keys = ["title", "baslik", "headline", "subject"]

    def bul(keys):
        for k in keys:
            if k in row and row[k]:
                v = row[k]
                # Sozluk ise icinden metin al (ornegin translation)
                if isinstance(v, dict):
                    v = v.get("tr") or v.get("text") or str(v)
                return metin_temizle(str(v))
        return ""

    icerik = bul(icerik_keys)
    baslik = bul(baslik_keys)

    if not icerik or len(icerik) < 20:
        return None

    return {
        "kaynak": kaynak,
        "baslik": baslik[:200],
        "icerik": icerik[:3000],
    }


def kaynak_indir(dataset_id, config, split, max_satir, aciklama):
    """Bir veri kaynagini indir ve DB'ye kaydet."""
    if kaynak_tamamlandi_mi(dataset_id):
        log.info("Zaten indirilmis: %s", aciklama)
        return 0

    log.info("Indiriliyor: %s", aciklama)
    durum_g["su_an"] = f"Indiriliyor: {aciklama}"

    toplam   = 0
    offset   = 0
    batch    = 100
    hata_say = 0

    while toplam < max_satir:
        data = hf_rows(dataset_id, config, split, offset, batch)

        if not data:
            hata_say += 1
            if hata_say > 5:
                log.warning("Cok fazla hata, atlanıyor: %s", aciklama)
                break
            time.sleep(15)
            continue
        else:
            hata_say = 0

        rows = data.get("rows", [])
        if not rows:
            break

        satirlar = []
        for row_obj in rows:
            row    = row_obj.get("row", row_obj)
            parsed = satir_parse(row, dataset_id)
            if parsed:
                satirlar.append(parsed)

        if satirlar:
            n = toplu_kaydet(satirlar)
            toplam += n
            durum_g["indirilen"] = toplam_say()

        offset += len(rows)

        if offset % 1000 == 0:
            log.info("%s: %d metin (%d offset)", aciklama, toplam, offset)

        # Son sayfa
        num_rows = data.get("num_rows_total", 0)
        if num_rows and offset >= min(num_rows, max_satir):
            break
        if len(rows) < batch:
            break

        time.sleep(0.3)

    egitim_kaydet(dataset_id, toplam)
    log.info("Tamamlandi: %s -> %d metin", aciklama, toplam)
    return toplam


def egitim_thread():
    """Arka planda veri indir ve modeli hazirla."""
    durum_g["aktif"]     = True
    durum_g["baslangic"] = datetime.now()

    # Zaten yeterli veri varsa atla
    mevcut = toplam_say()
    if mevcut > 50000:
        log.info("Yeterli veri mevcut: %d metin", mevcut)
        durum_g["su_an"] = f"Hazir: {mevcut:,} metin yuklu"
        return

    toplam = 0
    for dataset_id, config, split, max_satir, aciklama in VERI_KAYNAKLARI:
        try:
            n = kaynak_indir(dataset_id, config, split, max_satir, aciklama)
            toplam += n
            log.info("Toplam metin: %d", toplam_say())
        except Exception as e:
            log.error("Kaynak hata %s: %s", aciklama, e)
            durum_g["hata"] += 1
            time.sleep(20)

    if toplam_say() == 0:
        log.error("Hicbir veri yuklenemedi!")
        durum_g["su_an"] = "Veri yuklenemedi - lütfen HF_TOKEN ekle"
    else:
        durum_g["su_an"] = f"Egitim tamamlandi: {toplam_say():,} metin"
        log.info("Tum veri yuklendi: %d metin", toplam_say())


# ── TF-IDF ARAMA ─────────────────────────────────────────
_vec   = None
_mat   = None
_idler = []
_vlock = threading.Lock()


def tfidf_thread():
    global _vec, _mat, _idler
    while True:
        try:
            n = toplam_say()
            if n < 10:
                time.sleep(30)
                continue

            conn = db_baglan()
            rows = conn.execute(
                """SELECT id, baslik, icerik FROM metinler
                   ORDER BY id DESC LIMIT 200000"""
            ).fetchall()
            conn.close()

            metinler = [f"{r['baslik']} {r['icerik'][:500]}" for r in rows]
            idler    = [r["id"] for r in rows]

            v   = TfidfVectorizer(
                max_features=150000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                analyzer="char_wb",  # Turkce icin karakter n-gram daha iyi
            )
            mat = v.fit_transform(metinler)

            with _vlock:
                _vec   = v
                _mat   = mat
                _idler = idler

            log.info("TF-IDF guncellendi: %d metin", len(idler))
        except Exception as e:
            log.error("TF-IDF hata: %s", e)
        time.sleep(600)  # 10 dakikada bir guncelle


def benzer_bul(soru, kac=5):
    with _vlock:
        if _vec is None or _mat is None:
            return []
        try:
            sv      = _vec.transform([soru])
            skorlar = cosine_similarity(sv, _mat).flatten()
            en_iyi  = np.argsort(skorlar)[::-1][:kac * 3]
            sonuc   = []
            conn    = db_baglan()
            for idx in en_iyi:
                if skorlar[idx] < 0.01:
                    break
                if idx >= len(_idler):
                    continue
                r = conn.execute(
                    "SELECT * FROM metinler WHERE id=?", (_idler[idx],)
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


# ── YANIT ÜRETİCİ ────────────────────────────────────────

# Türkçe modeller - sirayla dene
TURKCE_MODELLER = [
    "redrussianarmy/gpt2-turkish",
    "turkish-nlp-suite/turkish-gpt2",
    "akoksal/TurkishBERTweet",
    "dbmdz/bert-base-turkish-cased",
]


def hf_model_yanit(soru, baglam):
    """HuggingFace Inference API ile Turkce yanit uret."""
    if not HF_TOKEN:
        return None

    prompt = (
        f"Konu: {baglam[:500]}\n\n"
        f"Soru: {soru}\n\n"
        f"Cevap:"
    )

    for model in TURKCE_MODELLER:
        try:
            r = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.7,
                        "return_full_text": False,
                        "do_sample": True,
                    },
                },
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    metin = data[0].get("generated_text", "").strip()
                    if metin and len(metin) > 10:
                        return metin
            elif r.status_code == 503:
                time.sleep(5)
                continue
        except Exception:
            continue
    return None


def yanitla(soru, entryler, gecmis):
    """Soruya yanit uret."""
    if not entryler:
        return (
            "Bu konuda henuz yeterli bilgi yok. "
            f"Simdilik {toplam_say():,} metin yuklu, egitim devam ediyor."
        )

    # Baglam olustur
    baglam_parcalar = []
    for e in entryler[:3]:
        baslik = e.get("baslik", "")
        icerik = e.get("icerik", "")[:400]
        if baslik:
            baglam_parcalar.append(f"[{baslik}]: {icerik}")
        else:
            baglam_parcalar.append(icerik)
    baglam = "\n\n".join(baglam_parcalar)

    # HF model ile dene
    model_yanit = hf_model_yanit(soru, baglam)
    if model_yanit:
        return model_yanit

    # Fallback: direkt metin goster
    en_iyi = entryler[0]
    baslik = en_iyi.get("baslik", "")
    icerik = en_iyi.get("icerik", "")
    kaynak = en_iyi.get("kaynak", "")

    yanit = ""
    if baslik:
        yanit += f"**{baslik}**\n\n"
    yanit += icerik[:600]
    if len(icerik) > 600:
        yanit += "..."

    # Kaynak bilgisi
    kaynak_ad = {
        "wikipedia": "Türkçe Wikipedia",
        "mc4": "Web metinleri",
        "cc100": "Common Crawl",
    }.get(kaynak, kaynak)
    if kaynak_ad:
        yanit += f"\n\n*Kaynak: {kaynak_ad}*"

    if len(entryler) > 1:
        yanit += f"\n\n📚 +{len(entryler)-1} ilgili metin daha bulundu."

    return yanit


# ── DURUM ────────────────────────────────────────────────

def sayac_metni():
    sure_str = ""
    if durum_g["baslangic"]:
        fark = datetime.now() - durum_g["baslangic"]
        s    = int(fark.total_seconds() // 3600)
        dk   = int((fark.total_seconds() % 3600) // 60)
        sure_str = f"\n\nSure: {s}s {dk}dk"

    return (
        "### Durum\n"
        f"**Yuklu metin:** {toplam_say():,}\n\n"
        f"**Aranabilir:** {len(_idler):,}\n\n"
        f"**Su an:** {durum_g['su_an']}\n\n"
        f"**Hata:** {durum_g['hata']}\n\n"
        f"**Durum:** {'Aktif' if durum_g['aktif'] else 'Bekliyor'}"
        f"{sure_str}"
    )


# ── SOHBET ───────────────────────────────────────────────

def sohbet(mesaj, gecmis):
    if not mesaj or not mesaj.strip():
        return "", gecmis or []
    gecmis   = gecmis or []
    entryler = benzer_bul(mesaj)
    yanit    = yanitla(mesaj, entryler, gecmis)
    sohbet_kaydet(mesaj, yanit)
    gecmis = gecmis + [
        {"role": "user",      "content": mesaj},
        {"role": "assistant", "content": yanit},
    ]
    return "", gecmis


def ozet_yap(metin, gecmis):
    """Verilen metni ozetle."""
    if not metin or not metin.strip():
        return "", gecmis or []
    gecmis = gecmis or []
    prompt = f"Su metni ozetle:\n\n{metin[:1000]}"
    entryler = benzer_bul(metin[:200])
    yanit = yanitla(prompt, entryler, gecmis)
    gecmis = gecmis + [
        {"role": "user",      "content": f"Özet: {metin[:100]}..."},
        {"role": "assistant", "content": yanit},
    ]
    return "", gecmis


# ── DB BAŞLAT ────────────────────────────────────────────
db_olustur()

# ── GRADIO ───────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1000px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Turkce AI", theme=gr.themes.Soft(primary_hue="blue"), css=CSS) as demo:

    gr.Markdown(
        "# Turkce AI\n"
        "Turkce Wikipedia, haberler, kitaplar ve web metinlerinden ogrenen yapay zeka."
    )

    with gr.Tabs():

        # --- SOHBET ---
        with gr.Tab("Sohbet"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=450, show_label=False, type="messages")
                    with gr.Row():
                        mesaj_box = gr.Textbox(
                            placeholder="Bir sey sor veya konustur...",
                            scale=5, show_label=False, container=False,
                        )
                        gonder_btn = gr.Button("Gonder", variant="primary", scale=1)
                    temizle_btn = gr.Button("Temizle", size="sm")

                with gr.Column(scale=1):
                    sayac_md   = gr.Markdown(sayac_metni())
                    yenile_btn = gr.Button("Yenile", size="sm")
                    gr.Markdown(
                        "---\n**Veri Kaynaklari:**\n\n"
                        "- Turkce Wikipedia\n"
                        "- mC4 web metinleri\n"
                        "- CC-100 Turkce\n"
                        "- Turkce kitaplar\n"
                        "- Turkce haberler\n\n"
                        "---\n`HF_TOKEN` ekle:\nModel yaniti aktif olur."
                    )

        # --- ÖZET ---
        with gr.Tab("Ozet"):
            with gr.Row():
                with gr.Column():
                    ozet_giris = gr.Textbox(
                        label="Ozetlenecek metin",
                        placeholder="Buraya metni yapistir...",
                        lines=8,
                    )
                    ozet_btn = gr.Button("Ozetle", variant="primary")
                with gr.Column():
                    ozet_cikis = gr.Chatbot(
                        height=350, show_label=False, type="messages",
                        label="Ozet"
                    )

        # --- ARAMA ---
        with gr.Tab("Arama"):
            with gr.Row():
                arama_box = gr.Textbox(
                    placeholder="Ara...",
                    scale=5, show_label=False, container=False,
                )
                ara_btn = gr.Button("Ara", variant="primary", scale=1)
            arama_sonuc = gr.Dataframe(
                headers=["Baslik", "Icerik", "Kaynak"],
                datatype=["str", "str", "str"],
                label="Sonuclar",
            )

    # Event handler'lar
    mesaj_box.submit(fn=sohbet, inputs=[mesaj_box, chatbot], outputs=[mesaj_box, chatbot])
    gonder_btn.click(fn=sohbet, inputs=[mesaj_box, chatbot], outputs=[mesaj_box, chatbot])
    temizle_btn.click(fn=lambda: ("", []), inputs=None, outputs=[mesaj_box, chatbot])
    yenile_btn.click(fn=sayac_metni, inputs=None, outputs=[sayac_md])

    ozet_btn.click(
        fn=ozet_yap,
        inputs=[ozet_giris, ozet_cikis],
        outputs=[ozet_giris, ozet_cikis],
    )

    def ara(sorgu):
        if not sorgu:
            return []
        sonuclar = benzer_bul(sorgu, kac=10)
        return [[
            r.get("baslik", "")[:50],
            r.get("icerik", "")[:200],
            r.get("kaynak", ""),
        ] for r in sonuclar]

    ara_btn.click(fn=ara, inputs=[arama_box], outputs=[arama_sonuc])
    arama_box.submit(fn=ara, inputs=[arama_box], outputs=[arama_sonuc])


# ── BAŞLATMA ─────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=egitim_thread, daemon=True, name="egitim").start()
    threading.Thread(target=tfidf_thread,  daemon=True, name="tfidf").start()

    log.info("Sunucu basliyor... port=%d", PORT)
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
        max_threads=40,
    )
