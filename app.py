"""
EKŞİ SÖZLÜK AI
- Ekşi Sözlük'ü arka planda tarar
- TF-IDF ile arama yapar (torch/GPU gerektirmez)
- HuggingFace API ile Türkçe yanıt üretir
"""

import gradio as gr
import requests
from bs4 import BeautifulSoup
import time
import os
import random
import threading
import sqlite3
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ════════════════════════════════════════════════
#  AYARLAR
# ════════════════════════════════════════════════
GECIKME_MIN  = 2.0
GECIKME_MAX  = 5.0
DB_DOSYASI   = "eksi.db"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HEADERS      = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}

# ════════════════════════════════════════════════
#  VERİTABANI
# ════════════════════════════════════════════════

def db_baglan():
    conn = sqlite3.connect(DB_DOSYASI, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_olustur():
    conn = db_baglan()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entryles (
            id         TEXT PRIMARY KEY,
            baslik     TEXT,
            baslik_url TEXT,
            icerik     TEXT,
            yazar      TEXT,
            tarih      TEXT,
            favori     TEXT
        );
        CREATE TABLE IF NOT EXISTS durum (
            anahtar TEXT PRIMARY KEY,
            deger   TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Veritabanı hazır")

def durum_oku(anahtar, varsayilan=""):
    conn = db_baglan()
    r = conn.execute("SELECT deger FROM durum WHERE anahtar=?", (anahtar,)).fetchone()
    conn.close()
    return r["deger"] if r else varsayilan

def durum_yaz(anahtar, deger):
    conn = db_baglan()
    conn.execute("INSERT OR REPLACE INTO durum VALUES (?,?)", (anahtar, str(deger)))
    conn.commit()
    conn.close()

def entry_kaydet(entryler):
    if not entryler:
        return
    conn = db_baglan()
    conn.executemany(
        """INSERT OR IGNORE INTO entryles
           (id, baslik, baslik_url, icerik, yazar, tarih, favori)
           VALUES (:id,:baslik,:baslik_url,:icerik,:yazar,:tarih,:favori)""",
        entryler
    )
    conn.commit()
    conn.close()

def toplam_say():
    conn = db_baglan()
    r = conn.execute("SELECT COUNT(*) as n FROM entryles").fetchone()
    conn.close()
    return r["n"]

# ════════════════════════════════════════════════
#  GLOBAL DURUM
# ════════════════════════════════════════════════
tarama = {
    "aktif":    False,
    "toplam":   0,
    "su_an":    "Başlatılıyor...",
    "baslangic": None,
}

# ════════════════════════════════════════════════
#  EKŞİ TARAYICI
# ════════════════════════════════════════════════

def baslik_cek(sayfa=1):
    basliklar = []
    for endpoint in [
        f"/basliklar/bugun?p={sayfa}",
        f"/basliklar/populer?p={sayfa}",
        f"/basliklar/gundem?p={sayfa}",
    ]:
        try:
            r = requests.get(
                "https://eksisozluk.com" + endpoint,
                headers=HEADERS, timeout=15
            )
            soup = BeautifulSoup(r.text, "html.parser")
            for item in soup.select("ul.topic-list li a"):
                href   = item.get("href", "")
                baslik = item.get_text(strip=True)
                if href and baslik and "/entry/" not in href:
                    url = ("https://eksisozluk.com" + href
                           if href.startswith("/") else href)
                    if not any(b["url"] == url for b in basliklar):
                        basliklar.append({"baslik": baslik, "url": url})
        except Exception:
            pass
        time.sleep(1)
    return basliklar

def entry_cek(url, sayfa=1):
    try:
        r = requests.get(f"{url}?p={sayfa}", headers=HEADERS, timeout=15)
        if r.status_code == 429:
            print("  ⚠️  Rate limit, 60sn bekleniyor...")
            time.sleep(60)
            return entry_cek(url, sayfa)
        if r.status_code != 200:
            return [], False

        soup      = BeautifulSoup(r.text, "html.parser")
        baslik_el = soup.select_one("h1#title a")
        baslik    = baslik_el.get_text(strip=True) if baslik_el else "?"
        entryler  = []

        for el in soup.select("ul#entry-item-list li[data-id]"):
            eid      = el.get("data-id", "")
            yazar    = el.get("data-author", "")
            ic_el    = el.select_one("div.content")
            icerik   = ic_el.get_text(" ", strip=True) if ic_el else ""
            t_el     = el.select_one("footer .entry-date")
            tarih    = t_el.get_text(strip=True) if t_el else ""
            f_el     = el.select_one("footer .favorite-count")
            favori   = f_el.get_text(strip=True) if f_el else "0"
            if icerik and len(icerik) > 15:
                entryler.append({
                    "id": eid, "baslik": baslik, "baslik_url": url,
                    "icerik": icerik, "yazar": yazar,
                    "tarih": tarih, "favori": favori,
                })

        sonraki = bool(soup.select_one("div.pager a[rel='next']"))
        return entryler, sonraki
    except Exception:
        return [], False

def tarama_thread():
    global tarama
    tarama["aktif"]     = True
    tarama["baslangic"] = datetime.now()
    tarama["toplam"]    = toplam_say()

    tamamlananlar = set(durum_oku("tamamlananlar", "").split(","))
    sayfa = 1

    while True:
        basliklar = baslik_cek(sayfa)
        if not basliklar:
            sayfa = 1
            time.sleep(300)
            continue

        for b in basliklar:
            if b["url"] in tamamlananlar:
                continue
            s = 1
            while True:
                tarama["su_an"] = f"'{b['baslik'][:30]}' — sayfa {s}"
                entryler, sonraki = entry_cek(b["url"], s)
                if entryler:
                    entry_kaydet(entryler)
                    tarama["toplam"] += len(entryler)
                if not sonraki or not entryler:
                    break
                s += 1
                time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))

            tamamlananlar.add(b["url"])
            durum_yaz("tamamlananlar", ",".join(list(tamamlananlar)[-5000:]))
            time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))

        sayfa += 1

# ════════════════════════════════════════════════
#  TF-IDF ARAMA (torch gerektirmez)
# ════════════════════════════════════════════════
_vectorizer  = None
_tfidf_mat   = None
_tfidf_idler = []
_tfidf_lock  = threading.Lock()

def tfidf_guncelle_thread():
    global _vectorizer, _tfidf_mat, _tfidf_idler
    while True:
        try:
            conn   = db_baglan()
            satirlar = conn.execute(
                "SELECT id, baslik, icerik FROM entryles ORDER BY rowid DESC LIMIT 20000"
            ).fetchall()
            conn.close()

            if len(satirlar) < 5:
                time.sleep(30)
                continue

            metinler = [f"{r['baslik']} {r['icerik'][:300]}" for r in satirlar]
            idler    = [r["id"] for r in satirlar]

            v   = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
            mat = v.fit_transform(metinler)

            with _tfidf_lock:
                _vectorizer  = v
                _tfidf_mat   = mat
                _tfidf_idler = idler

            print(f"✅ TF-IDF güncellendi: {len(idler):,} entry")
        except Exception as e:
            print(f"TF-IDF hata: {e}")

        time.sleep(120)   # 2 dakikada bir yenile

def benzer_bul(soru, kac=5):
    with _tfidf_lock:
        if _vectorizer is None or _tfidf_mat is None:
            return []
        try:
            soru_v = _vectorizer.transform([soru])
            skorlar = cosine_similarity(soru_v, _tfidf_mat).flatten()
            en_iyi  = np.argsort(skorlar)[::-1][:kac]
            sonuclar = []
            conn = db_baglan()
            for idx in en_iyi:
                if skorlar[idx] < 0.05:
                    continue
                r = conn.execute(
                    "SELECT * FROM entryles WHERE id=?", (_tfidf_idler[idx],)
                ).fetchone()
                if r:
                    sonuclar.append(dict(r))
            conn.close()
            return sonuclar
        except Exception:
            return []

# ════════════════════════════════════════════════
#  YANIT ÜRETİCİ
# ════════════════════════════════════════════════

def hf_yanitla(soru, entryler, gecmis):
    if not HF_TOKEN:
        return lokal_yanitla(soru, entryler)

    baglam = "\n\n".join([
        f"[{i+1}] {e['baslik']}: {e['icerik'][:250]}"
        for i, e in enumerate(entryler)
    ]) if entryler else "İlgili entry bulunamadı."

    gecmis_str = "".join([
        f"Kullanıcı: {m[0]}\nBot: {m[1]}\n"
        for m in gecmis[-3:] if m[1]
    ])

    prompt = (
        "Sen Ekşi Sözlük tarzında, samimi ve esprili bir Türkçe asistansın.\n\n"
        f"Ekşi entry'leri:\n{baglam}\n\n"
        f"{gecmis_str}"
        f"Kullanıcı: {soru}\nBot:"
    )

    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers={
                "Authorization":  f"Bearer {HF_TOKEN}",
                "Content-Type":   "application/json",
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens":  300,
                    "temperature":     0.8,
                    "return_full_text": False,
                },
            },
            timeout=45,
        )
        if r.status_code == 200:
            return r.json()[0].get("generated_text", "").strip()
        elif r.status_code == 503:
            return "⏳ AI modeli ısınıyor, 20 saniye bekle ve tekrar dene."
        else:
            return lokal_yanitla(soru, entryler)
    except Exception:
        return lokal_yanitla(soru, entryler)

def lokal_yanitla(soru, entryler):
    if not entryler:
        return "bu konuda henüz yeterli entry yok, tarama devam ediyor 🔄"
    e = entryler[0]
    y = f"**{e['baslik']}** başlığından:\n\n{e['icerik'][:500]}"
    if len(e["icerik"]) > 500:
        y += "..."
    y += f"\n\n*— {e['yazar']}, {e['tarih']} · {e['favori']} fav*"
    if len(entryler) > 1:
        y += f"\n\n📌 +{len(entryler)-1} benzer entry daha bulundu."
    return y

def sayac_metni():
    t    = tarama
    sure = ""
    if t["baslangic"]:
        fark = datetime.now() - t["baslangic"]
        saat = int(fark.total_seconds() // 3600)
        dk   = int((fark.total_seconds() % 3600) // 60)
        sure = f"\n\n**Süre:** {saat}s {dk}dk"
    return (
        f"### 📊 Tarama Durumu\n"
        f"**Toplam entry:** {t['toplam']:,}\n\n"
        f"**Aranabilir:** {len(_tfidf_idler):,}\n\n"
        f"**Şu an:** {t['su_an']}\n\n"
        f"**Durum:** {'🟢 Aktif' if t['aktif'] else '🔴 Durdu'}"
        f"{sure}"
    )

# ════════════════════════════════════════════════
#  SOHBET
# ════════════════════════════════════════════════

def sohbet(mesaj, gecmis):
    if not mesaj or not mesaj.strip():
        return "", gecmis
    gecmis   = gecmis or []
    entryler = benzer_bul(mesaj)
    yanit    = hf_yanitla(mesaj, entryler, gecmis)
    gecmis   = gecmis + [[mesaj, yanit]]
    return "", gecmis

# ════════════════════════════════════════════════
#  GRADIO ARAYÜZÜ
# ════════════════════════════════════════════════

with gr.Blocks(
    title="🍋 Ekşi Sözlük AI",
    theme=gr.themes.Soft(primary_hue="lime"),
    css=(
        ".gradio-container{max-width:900px!important;margin:auto!important}"
        "footer{display:none!important}"
    ),
) as demo:

    gr.Markdown(
        "# 🍋 Ekşi Sözlük AI\n"
        "Ekşi Sözlük entry'lerinden öğrenen sohbet botu. "
        "Tarama arka planda sürekli devam ediyor."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, show_label=False)
            with gr.Row():
                mesaj_box = gr.Textbox(
                    placeholder="Bir şey sor... (Enter ile gönder)",
                    scale=5, show_label=False, container=False,
                )
                gonder_btn = gr.Button("Gönder 🚀", variant="primary", scale=1)
            temizle_btn = gr.Button("🗑️ Temizle", size="sm")

        with gr.Column(scale=1):
            sayac_md  = gr.Markdown(sayac_metni())
            yenile_btn = gr.Button("🔄 Yenile", size="sm")
            gr.Markdown(
                "---\n**Nasıl çalışır?**\n\n"
                "1. Ekşi'yi tarar\n"
                "2. Entry'leri kaydeder\n"
                "3. TF-IDF ile benzer entry bulur\n"
                "4. Ekşi ruhuyla yanıt verir"
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

# ════════════════════════════════════════════════
#  BAŞLATMA
# ════════════════════════════════════════════════

if __name__ == "__main__":
    db_olustur()
    threading.Thread(target=tarama_thread,        daemon=True).start()
    threading.Thread(target=tfidf_guncelle_thread, daemon=True).start()

    print("🌐 Uygulama başlıyor...")
    port = int(os.environ.get("PORT", 7860))
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
