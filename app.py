"""
EKŞİ SÖZLÜK AI
- HuggingFace'den Eksi dataset'i indirir (engel yok, milyonlarca entry)
- SQLite ile saklar
- TF-IDF ile arama
- Gradio 5 web arayüzü
"""

import os
import time
import sqlite3
import threading
import logging
import json
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
DB_DOSYASI = os.environ.get("DB_PATH", "eksi.db")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
PORT       = int(os.environ.get("PORT", 7860))

# HuggingFace dataset listesi - sirayla denenir
HF_DATASETS = [
    # (dataset_id, config, split)
    ("Toygar/eksisozluk-entries",          None,     "train"),
    ("hilaluyanik/eksisozluk",             None,     "train"),
    ("turkish-nlp-suite/eksisozluk",       None,     "train"),
    ("muratsilahtaroglu/ExsiSozlukDataset",None,     "train"),
    ("datasets/eksisozluk",                None,     "train"),
]

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}" if HF_TOKEN else "",
    "User-Agent": "datasets/2.0.0",
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
            baslik_url  TEXT DEFAULT '',
            icerik      TEXT NOT NULL,
            yazar       TEXT DEFAULT '',
            tarih       TEXT DEFAULT '',
            favori      TEXT DEFAULT '0'
        );
        CREATE INDEX IF NOT EXISTS idx_baslik ON entryles(baslik);

        CREATE TABLE IF NOT EXISTS meta (
            anahtar TEXT PRIMARY KEY,
            deger   TEXT
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


def meta_oku(k, v=""):
    try:
        conn = db_baglan()
        r = conn.execute("SELECT deger FROM meta WHERE anahtar=?", (k,)).fetchone()
        conn.close()
        return r[0] if r else v
    except Exception:
        return v


def meta_yaz(k, v):
    try:
        conn = db_baglan()
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(v)))
        conn.commit()
        conn.close()
    except Exception:
        pass


def toplu_kaydet(satirlar):
    """List of dict -> DB. Her dict: id, baslik, icerik zorunlu."""
    if not satirlar:
        return 0
    conn = db_baglan()
    try:
        c = conn.executemany(
            """INSERT OR IGNORE INTO entryles
               (id, baslik, baslik_url, icerik, yazar, tarih, favori)
               VALUES (:id,:baslik,:baslik_url,:icerik,:yazar,:tarih,:favori)""",
            satirlar,
        )
        conn.commit()
        return c.rowcount
    except Exception as e:
        log.error("toplu_kaydet: %s", e)
        return 0
    finally:
        conn.close()


# ── GLOBAL DURUM ─────────────────────────────────────────
durum_g = {
    "aktif":     False,
    "su_an":     "Baslatiliyor...",
    "baslangic": None,
    "hata":      0,
}

# ── HUGGINGFACE DATASET İNDİRİCİ ─────────────────────────

def hf_dataset_bilgi(dataset_id):
    """Dataset'in mevcut split ve config'lerini dondur."""
    try:
        url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
        r = requests.get(url, headers=HF_HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning("Dataset bilgi hatasi %s: %s", dataset_id, e)
    return None


def hf_satirlari_cek(dataset_id, config, split, offset=0, length=100):
    """HuggingFace datasets-server API ile satirlari cek."""
    try:
        params = f"dataset={dataset_id}&split={split}&offset={offset}&length={length}"
        if config:
            params += f"&config={config}"
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        r = requests.get(url, headers=HF_HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        log.warning("HF rows HTTP %d: %s", r.status_code, dataset_id)
    except Exception as e:
        log.error("hf_satirlari_cek: %s", e)
    return None


def satir_parse(row, dataset_id, idx):
    """
    HF row'unu entryles formatina donustur.
    Farkli datasetlerin farkli sutun isimleri olabilir.
    """
    # Olasi sutun isimleri
    icerik_keys  = ["entry", "content", "text", "body", "entry_text", "icerik"]
    baslik_keys  = ["title", "baslik", "topic", "subject", "topic_title"]
    yazar_keys   = ["author", "nick", "username", "yazar", "user"]
    tarih_keys   = ["date", "created_at", "tarih", "timestamp", "created"]
    favori_keys  = ["favorite_count", "fav_count", "favorites", "favori", "like_count"]

    def bul(keys):
        for k in keys:
            if k in row and row[k] is not None:
                return str(row[k]).strip()
        return ""

    icerik = bul(icerik_keys)
    baslik = bul(baslik_keys)

    if not icerik or len(icerik) < 5:
        return None

    return {
        "id":         str(row.get("id", f"{dataset_id}-{idx}")),
        "baslik":     baslik or "genel",
        "baslik_url": "",
        "icerik":     icerik[:2000],
        "yazar":      bul(yazar_keys),
        "tarih":      bul(tarih_keys),
        "favori":     bul(favori_keys) or "0",
    }


def dataset_indir_ve_kaydet(dataset_id, config, split):
    """Bir dataseti tamamen indirip DB'ye kaydeder."""
    log.info("Dataset deneniyor: %s / %s / %s", dataset_id, config, split)
    durum_g["su_an"] = f"Dataset: {dataset_id} kontrol ediliyor..."

    # Once kac satir var ogren
    bilgi = hf_dataset_bilgi(dataset_id)
    if not bilgi:
        # Direkt dene
        test = hf_satirlari_cek(dataset_id, config, split, 0, 5)
        if not test or not test.get("rows"):
            log.warning("Dataset erisim yok: %s", dataset_id)
            return 0

    toplam_indir = 0
    offset = 0
    batch  = 100
    hata_say = 0

    while True:
        durum_g["su_an"] = f"{dataset_id} indiriliyor... ({toplam_indir:,} entry)"
        data = hf_satirlari_cek(dataset_id, config, split, offset, batch)

        if not data:
            hata_say += 1
            if hata_say > 3:
                break
            time.sleep(10)
            continue

        rows = data.get("rows", [])
        if not rows:
            break  # Bitti

        satirlar = []
        for i, row_obj in enumerate(rows):
            row = row_obj.get("row", row_obj)
            parsed = satir_parse(row, dataset_id, offset + i)
            if parsed:
                satirlar.append(parsed)

        if satirlar:
            n = toplu_kaydet(satirlar)
            toplam_indir += n

        offset += len(rows)

        # Her 1000 entry'de bir log
        if offset % 1000 == 0:
            log.info("%s: %d entry indirildi", dataset_id, toplam_indir)

        # Toplam satir sayisini asinca dur
        num_rows = data.get("num_rows_total", data.get("numRowsTotal", 0))
        if num_rows and offset >= num_rows:
            break

        if len(rows) < batch:
            break  # Son sayfa

        time.sleep(0.5)  # Rate limit

    log.info("Dataset tamamlandi: %s -> %d entry", dataset_id, toplam_indir)
    return toplam_indir


def veri_yukle_thread():
    """
    Arka planda HF datasetlerini indirir.
    Zaten veri varsa atlar.
    """
    global durum_g
    durum_g["aktif"]     = True
    durum_g["baslangic"] = datetime.now()

    # Zaten veri var mi?
    mevcut = toplam_say()
    if mevcut > 10000:
        log.info("Zaten %d entry var, indirme atlandi.", mevcut)
        durum_g["su_an"] = f"Hazir: {mevcut:,} entry mevcut"
        return

    log.info("Veri yuklemesi basliyor...")
    toplam = 0

    for dataset_id, config, split in HF_DATASETS:
        if toplam > 500000:  # 500k yeterli
            break
        try:
            n = dataset_indir_ve_kaydet(dataset_id, config, split)
            toplam += n
            if n > 0:
                meta_yaz("son_dataset", dataset_id)
                log.info("Toplam entry: %d", toplam_say())
        except Exception as e:
            log.error("Dataset hata %s: %s", dataset_id, e)
            durum_g["hata"] += 1
            time.sleep(15)

    if toplam == 0:
        # Hicbir dataset calismadiysa manuel fallback
        log.error("Hicbir HF dataset erisilemedı! Manuel veri ekleniyor...")
        durum_g["su_an"] = "HF erisim yok - ornek veri yuklendi"
        ornek_ekle()
    else:
        durum_g["su_an"] = f"Tamamlandi: {toplam_say():,} entry"
        log.info("Veri yukleme tamamlandi: %d entry", toplam_say())


def ornek_ekle():
    """HF erisim yoksa ornek entry ekle - test icin."""
    ornekler = [
        {"id": "o1", "baslik": "python programlama", "baslik_url": "",
         "icerik": "guzel bir programlama dili. ogrenmesi kolay ama ustalasmasi zor.", "yazar": "biri", "tarih": "2024", "favori": "5"},
        {"id": "o2", "baslik": "yapay zeka", "baslik_url": "",
         "icerik": "artik her yerde yapay zeka var. iyi mi kotu mu bilemedik.", "yazar": "digeri", "tarih": "2024", "favori": "10"},
        {"id": "o3", "baslik": "turkiye", "baslik_url": "",
         "icerik": "cok guzel bir ulke. sorunlari var ama potansiyeli buyuk.", "yazar": "birkisi", "tarih": "2024", "favori": "3"},
    ]
    toplu_kaydet(ornekler)


# ── TF-IDF ───────────────────────────────────────────────
_vec   = None
_mat   = None
_idler = []
_vlock = threading.Lock()


def tfidf_thread():
    global _vec, _mat, _idler
    while True:
        try:
            n = toplam_say()
            if n < 3:
                time.sleep(30)
                continue

            conn = db_baglan()
            rows = conn.execute(
                "SELECT id, baslik, icerik FROM entryles ORDER BY rowid DESC LIMIT 100000"
            ).fetchall()
            conn.close()

            metinler = [f"{r['baslik']} {r['icerik'][:400]}" for r in rows]
            idler    = [r["id"] for r in rows]

            v   = TfidfVectorizer(max_features=100000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=1)
            mat = v.fit_transform(metinler)

            with _vlock:
                _vec   = v
                _mat   = mat
                _idler = idler
            log.info("TF-IDF guncellendi: %d entry", len(idler))
        except Exception as e:
            log.error("TF-IDF hata: %s", e)
        time.sleep(300)  # 5 dakikada bir guncelle


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
                if skorlar[idx] < 0.02:
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
            if g.get("role") == "user":
                gecmis_str += f"Kullanici: {g['content']}\n"
            elif g.get("role") == "assistant":
                gecmis_str += f"Bot: {g['content']}\n"

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
        return "Bu konuda henuz entry yok. Veri yuklemesi devam ediyor, bekle."
    satirlar = [f"**{entryler[0]['baslik']}** basligindan:\n"]
    for e in entryler[:3]:
        ic = e["icerik"][:400] + ("..." if len(e["icerik"]) > 400 else "")
        yazar  = e.get("yazar") or "anonim"
        tarih  = e.get("tarih") or "?"
        favori = e.get("favori") or "0"
        satirlar.append(f"> {ic}\n> *- {yazar}, {tarih} - {favori} fav*\n")
    return "\n".join(satirlar)


# ── DURUM METNİ ──────────────────────────────────────────

def sayac_metni():
    sure_str = ""
    if durum_g["baslangic"]:
        fark = datetime.now() - durum_g["baslangic"]
        s    = int(fark.total_seconds() // 3600)
        dk   = int((fark.total_seconds() % 3600) // 60)
        sure_str = f"\n\nSure: {s}s {dk}dk"

    return (
        "### Durum\n"
        f"**Entry:** {toplam_say():,}\n\n"
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
    yanit    = hf_yanitla(mesaj, entryler, gecmis)
    gecmis   = gecmis + [
        {"role": "user",      "content": mesaj},
        {"role": "assistant", "content": yanit},
    ]
    return "", gecmis


# ── DB BAŞLAT ────────────────────────────────────────────
db_olustur()

# ── GRADIO ───────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 960px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Eksi Sozluk AI", theme=gr.themes.Soft(primary_hue="lime"), css=CSS) as demo:

    gr.Markdown(
        "# Eksi Sozluk AI\n"
        "Milyonlarca Eksi entry'sinden ogrenen sohbet botu."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, show_label=False, type="messages")
            with gr.Row():
                mesaj_box = gr.Textbox(
                    placeholder="Bir sey sor...",
                    scale=5, show_label=False, container=False,
                )
                gonder_btn = gr.Button("Gonder", variant="primary", scale=1)
            temizle_btn = gr.Button("Temizle", size="sm")

        with gr.Column(scale=1):
            sayac_md   = gr.Markdown(sayac_metni())
            yenile_btn = gr.Button("Yenile", size="sm")
            gr.Markdown(
                "---\n**Nasil calisir?**\n\n"
                "1. HuggingFace'den\n   Eksi dataset'i indirir\n"
                "2. TF-IDF ile arar\n"
                "3. Yanit uretir\n\n"
                "---\n`HF_TOKEN` ile\nMistral-7B aktif olur."
            )

    mesaj_box.submit(fn=sohbet, inputs=[mesaj_box, chatbot], outputs=[mesaj_box, chatbot])
    gonder_btn.click(fn=sohbet, inputs=[mesaj_box, chatbot], outputs=[mesaj_box, chatbot])
    temizle_btn.click(fn=lambda: ("", []), inputs=None, outputs=[mesaj_box, chatbot])
    yenile_btn.click(fn=sayac_metni, inputs=None, outputs=[sayac_md])


# ── BAŞLATMA ─────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=veri_yukle_thread, daemon=True, name="yukleyici").start()
    threading.Thread(target=tfidf_thread,      daemon=True, name="tfidf").start()

    log.info("Sunucu basliyor... port=%d", PORT)
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
        max_threads=40,
    )
