"""
EKŞİ SÖZLÜK AI - ANA UYGULAMA
================================
Bu tek dosya her şeyi yapar:
1. Arka planda Ekşi'yi tarar
2. Önde sohbet botunu çalıştırır
3. Anlık tarama sayacını gösterir
"""

import gradio as gr
import requests
from bs4 import BeautifulSoup
import json
import time
import os
import random
import threading
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# ════════════════════════════════════════════════
#  AYARLAR
# ════════════════════════════════════════════════
GECIKME_MIN = 2.0
GECIKME_MAX = 4.0
DB_DOSYASI = "eksi.db"
EMBED_MODEL_ADI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9",
}

# ════════════════════════════════════════════════
#  VERİTABANI KURULUMU
# ════════════════════════════════════════════════

def db_baglan():
    conn = sqlite3.connect(DB_DOSYASI, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_olustur():
    conn = db_baglan()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entryles (
            id          TEXT PRIMARY KEY,
            baslik      TEXT,
            baslik_url  TEXT,
            icerik      TEXT,
            yazar       TEXT,
            tarih       TEXT,
            favori      TEXT,
            vektorlendi INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS durum (
            anahtar TEXT PRIMARY KEY,
            deger   TEXT
        );

        CREATE TABLE IF NOT EXISTS vektorler (
            entry_id TEXT PRIMARY KEY,
            vektor   BLOB
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Veritabanı hazır")

def durum_oku(anahtar, varsayilan="0"):
    conn = db_baglan()
    r = conn.execute(
        "SELECT deger FROM durum WHERE anahtar=?", (anahtar,)
    ).fetchone()
    conn.close()
    return r["deger"] if r else varsayilan

def durum_yaz(anahtar, deger):
    conn = db_baglan()
    conn.execute(
        "INSERT OR REPLACE INTO durum VALUES (?,?)", (anahtar, str(deger))
    )
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

def toplam_entry_say():
    conn = db_baglan()
    r = conn.execute("SELECT COUNT(*) as n FROM entryles").fetchone()
    conn.close()
    return r["n"]

# ════════════════════════════════════════════════
#  GLOBAL DURUM (tarama sayacı için)
# ════════════════════════════════════════════════
tarama_durumu = {
    "aktif": False,
    "toplam": 0,
    "su_an": "Bekleniyor...",
    "baslangic": None,
    "hata": None,
}

# ════════════════════════════════════════════════
#  EKŞİ TARAYICI
# ════════════════════════════════════════════════

def baslik_listesi_cek(sayfa=1):
    basliklar = []
    kaynaklar = [
        f"/basliklar/bugun?p={sayfa}",
        f"/basliklar/populer?p={sayfa}",
        f"/basliklar/gundem?p={sayfa}",
    ]
    for kaynak in kaynaklar:
        try:
            r = requests.get(
                "https://eksisozluk.com" + kaynak,
                headers=HEADERS, timeout=15
            )
            soup = BeautifulSoup(r.text, "html.parser")
            for item in soup.select("ul.topic-list li a"):
                href = item.get("href", "")
                baslik = item.get_text(strip=True)
                if href and baslik and "/entry/" not in href:
                    url = "https://eksisozluk.com" + href if href.startswith("/") else href
                    if not any(b["url"] == url for b in basliklar):
                        basliklar.append({"baslik": baslik, "url": url})
        except Exception:
            pass
        time.sleep(1)
    return basliklar

def entry_cek(baslik_url, sayfa=1):
    entryler = []
    sonraki = False
    try:
        r = requests.get(
            f"{baslik_url}?p={sayfa}",
            headers=HEADERS, timeout=15
        )
        if r.status_code == 429:
            print("  ⚠️  Rate limit! 60sn bekleniyor...")
            time.sleep(60)
            return entry_cek(baslik_url, sayfa)
        if r.status_code != 200:
            return [], False

        soup = BeautifulSoup(r.text, "html.parser")
        baslik_el = soup.select_one("h1#title a")
        baslik_adi = baslik_el.get_text(strip=True) if baslik_el else "?"

        for el in soup.select("ul#entry-item-list li[data-id]"):
            eid    = el.get("data-id", "")
            yazar  = el.get("data-author", "")
            icerik_el = el.select_one("div.content")
            icerik = icerik_el.get_text(" ", strip=True) if icerik_el else ""
            tarih_el  = el.select_one("footer .entry-date")
            tarih  = tarih_el.get_text(strip=True) if tarih_el else ""
            fav_el = el.select_one("footer .favorite-count")
            favori = fav_el.get_text(strip=True) if fav_el else "0"

            if icerik and len(icerik) > 15:
                entryler.append({
                    "id": eid, "baslik": baslik_adi,
                    "baslik_url": baslik_url, "icerik": icerik,
                    "yazar": yazar, "tarih": tarih, "favori": favori,
                })

        sonraki = bool(soup.select_one("div.pager a[rel='next']"))
    except Exception as e:
        tarama_durumu["hata"] = str(e)
    return entryler, sonraki

def tarama_thread():
    """Arka planda sürekli çalışır."""
    global tarama_durumu
    tarama_durumu["aktif"] = True
    tarama_durumu["baslangic"] = datetime.now()

    tamamlananlar = set(
        (durum_oku("tamamlanan_urls", "")).split(",")
    )
    tarama_durumu["toplam"] = toplam_entry_say()

    sayfa = 1
    while True:
        basliklar = baslik_listesi_cek(sayfa)
        if not basliklar:
            sayfa = 1
            time.sleep(300)
            continue

        for b in basliklar:
            if b["url"] in tamamlananlar:
                continue

            s = 1
            while True:
                tarama_durumu["su_an"] = (
                    f"'{b['baslik'][:35]}' — sayfa {s}"
                )
                entryler, sonraki = entry_cek(b["url"], s)
                if entryler:
                    entry_kaydet(entryler)
                    tarama_durumu["toplam"] += len(entryler)

                if not sonraki or not entryler:
                    break
                s += 1
                time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))

            tamamlananlar.add(b["url"])
            durum_yaz(
                "tamamlanan_urls",
                ",".join(list(tamamlananlar)[-5000:])
            )
            time.sleep(random.uniform(GECIKME_MIN, GECIKME_MAX))

        sayfa += 1

# ════════════════════════════════════════════════
#  EMBEDDİNG VE ARAMA
# ════════════════════════════════════════════════

embed_model = None
bellek_vektorler = []
bellek_entryle_idler = []

def model_yukle():
    global embed_model
    print("📥 Embedding modeli yükleniyor...")
    embed_model = SentenceTransformer(EMBED_MODEL_ADI)
    print("✅ Model hazır!")

def vektorleme_thread():
    """Yeni entry'leri arka planda vektörleştirir."""
    global bellek_vektorler, bellek_entryle_idler
    while embed_model is None:
        time.sleep(5)

    while True:
        try:
            conn = db_baglan()
            satirlar = conn.execute(
                """SELECT id, baslik, icerik FROM entryles
                   WHERE vektorlendi=0 LIMIT 100"""
            ).fetchall()
            conn.close()

            if not satirlar:
                time.sleep(30)
                continue

            metinler = [
                f"{r['baslik']} | {r['icerik'][:400]}"
                for r in satirlar
            ]
            yeni_vektorler = embed_model.encode(metinler, show_progress_bar=False)

            bellek_vektorler.extend(yeni_vektorler)
            bellek_entryle_idler.extend([r["id"] for r in satirlar])

            conn = db_baglan()
            conn.executemany(
                "UPDATE entryles SET vektorlendi=1 WHERE id=?",
                [(r["id"],) for r in satirlar]
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Vektörleme hatası: {e}")
            time.sleep(10)

def benzer_bul(soru, kac=5):
    if not bellek_vektorler or embed_model is None:
        return []

    soru_v = embed_model.encode([soru])
    matris = np.array(bellek_vektorler)
    norm_m = matris / (np.linalg.norm(matris, axis=1, keepdims=True) + 1e-8)
    norm_s = soru_v / (np.linalg.norm(soru_v) + 1e-8)
    skorlar = np.dot(norm_m, norm_s.T).flatten()
    en_iyi = np.argsort(skorlar)[::-1][:kac]

    sonuclar = []
    conn = db_baglan()
    for idx in en_iyi:
        if skorlar[idx] < 0.25:
            continue
        eid = bellek_entryle_idler[idx]
        r = conn.execute(
            "SELECT * FROM entryles WHERE id=?", (eid,)
        ).fetchone()
        if r:
            sonuclar.append(dict(r))
    conn.close()
    return sonuclar

# ════════════════════════════════════════════════
#  SOHBET
# ════════════════════════════════════════════════

def llm_yanitla(soru, entryler, gecmis):
    """HuggingFace ücretsiz API ile yanıt üret."""
    bagiam = "\n\n".join([
        f"[{i+1}] Başlık: {e['baslik']}\n"
        f"İçerik: {e['icerik'][:300]}\n"
        f"Yazar: {e['yazar']} | Favori: {e['favori']}"
        for i, e in enumerate(entryler)
    ]) if entryler else "İlgili entry bulunamadı."

    gecmis_metni = "".join([
        f"{'Kullanıcı' if m['role']=='user' else 'Bot'}: {m['content']}\n"
        for m in gecmis[-4:]
    ])

    prompt = f"""Sen Ekşi Sözlük ruhunu bilen, samimi ve esprili bir Türkçe asistansın.
Aşağıdaki Ekşi entry'lerini kullanarak soruyu yanıtla.
Ekşi'nin dilini kullan: samimi, bazen alaycı, içten.

Ekşi Entry'leri:
{bagiam}

{gecmis_metni}Kullanıcı: {soru}
Bot:"""

    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 350,
                    "temperature": 0.8,
                    "return_full_text": False,
                }
            },
            timeout=45
        )
        if r.status_code == 200:
            yanit = r.json()[0].get("generated_text", "").strip()
            return yanit
        elif r.status_code == 503:
            return "⏳ AI modeli ısınıyor, 20 saniye bekle ve tekrar gönder."
        else:
            return lokal_yanitla(soru, entryler)
    except Exception:
        return lokal_yanitla(soru, entryler)

def lokal_yanitla(soru, entryler):
    """API olmadan direkt entry göster."""
    if not entryler:
        return (
            "bu konuda henüz yeterli entry taranmadı. "
            "tarama devam ediyor, biraz sonra tekrar dene! 🔄"
        )
    e = entryler[0]
    yanit = f"**{e['baslik']}** başlığından:\n\n"
    yanit += f"{e['icerik'][:500]}{'...' if len(e['icerik'])>500 else ''}\n\n"
    yanit += f"*— {e['yazar']}, {e['tarih']} · {e['favori']} favori*"
    if len(entryler) > 1:
        yanit += f"\n\n📌 +{len(entryler)-1} benzer entry daha bulundu."
    return yanit

def sayac_guncelle():
    """Anlık tarama durumunu döndürür."""
    t = tarama_durumu
    sure = ""
    if t["baslangic"]:
        fark = datetime.now() - t["baslangic"]
        saat = int(fark.total_seconds() // 3600)
        dk   = int((fark.total_seconds() % 3600) // 60)
        sure = f" · {saat}s {dk}dk çalışıyor"

    vek = len(bellek_vektorler)
    return (
        f"### 📊 Tarama Durumu\n"
        f"**Toplam entry:** {t['toplam']:,}\n\n"
        f"**Aranabilir entry:** {vek:,}\n\n"
        f"**Şu an:** {t['su_an']}\n\n"
        f"**Durum:** {'🟢 Aktif' if t['aktif'] else '🔴 Durdu'}{sure}"
    )

# ════════════════════════════════════════════════
#  SOHBET FONKSİYONU
# ════════════════════════════════════════════════

def sohbet_ve_goster(mesaj_input, gecmis):
    """
    mesaj_input : str  – kullanıcının yazdığı metin
    gecmis      : list – chatbot'un kendi tuttuğu mesaj geçmişi
                         [{"role": "user"|"assistant", "content": "..."}, ...]
    Döndürür    : (boş string, güncellenmiş gecmis)
    """
    if not mesaj_input or not mesaj_input.strip():
        return "", gecmis

    gecmis = gecmis or []

    entryler = benzer_bul(mesaj_input)
    if HF_TOKEN:
        yanit = llm_yanitla(mesaj_input, entryler, gecmis)
    else:
        yanit = lokal_yanitla(mesaj_input, entryler)

    gecmis = gecmis + [
        {"role": "user",      "content": mesaj_input},
        {"role": "assistant", "content": yanit},
    ]
    return "", gecmis

# ════════════════════════════════════════════════
#  GRADIO ARAYÜZÜ
# ════════════════════════════════════════════════

with gr.Blocks(
    title="🍋 Ekşi Sözlük AI",
    theme=gr.themes.Soft(primary_hue="lime"),
    css="""
    .gradio-container{max-width:860px!important;margin:auto!important}
    footer{display:none!important}
    """
) as demo:

    gr.Markdown("""
# 🍋 Ekşi Sözlük AI
Ekşi Sözlük entry'lerinden beslenen sohbet botu.
Tarama arka planda devam ediyor — ne kadar çok entry birikirse o kadar akıllı olur.
""")

    with gr.Row():
        with gr.Column(scale=3):
            # type="messages" → {"role":..., "content":...} formatını kabul eder
            chatbot = gr.Chatbot(
                label="Sohbet",
                height=480,
                show_label=False,
                type="messages",
            )
            with gr.Row():
                mesaj = gr.Textbox(
                    placeholder="Bir şey sor... (Giriş)",
                    scale=5, show_label=False, container=False,
                )
                gonder = gr.Button("Gönder", variant="primary", scale=1)
            temizle = gr.Button("🗑️ Temizle", size="sm")

        with gr.Column(scale=1):
            sayac = gr.Markdown(sayac_guncelle())
            yenile = gr.Button("🔄 Yenile", size="sm")
            gr.Markdown("""
---
**Nasıl çalışır?**

1. Sunucu açıldığında taramayı başlatır
2. Her entry veritabanına kaydolur
3. Sen soru sordukça en alakalı entry'leri bulur
4. Ekşi ruhuyla yanıt verir
""")

    # chatbot hem input hem output — gecmis_state'e artık gerek yok
    mesaj.submit(
        fn=sohbet_ve_goster,
        inputs=[mesaj, chatbot],
        outputs=[mesaj, chatbot],
    )
    gonder.click(
        fn=sohbet_ve_goster,
        inputs=[mesaj, chatbot],
        outputs=[mesaj, chatbot],
    )
    temizle.click(
        fn=lambda: ("", []),
        inputs=None,
        outputs=[mesaj, chatbot],
    )
    yenile.click(
        fn=sayac_guncelle,
        inputs=None,
        outputs=[sayac],
    )

# ════════════════════════════════════════════════
#  BAŞLATMA
# ════════════════════════════════════════════════

if __name__ == "__main__":
    db_olustur()

    # Embedding modelini ayrı thread'de yükle (uygulamayı bloklamasın)
    threading.Thread(target=model_yukle, daemon=True).start()

    # Taramayı başlat
    threading.Thread(target=tarama_thread, daemon=True).start()

    # Vektörlemeyi başlat
    threading.Thread(target=vektorleme_thread, daemon=True).start()

    print("🌐 Uygulama başlıyor...")
    port = int(os.environ.get("PORT", 7860))

    demo.queue()   # queue her zaman açık olmalı
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
