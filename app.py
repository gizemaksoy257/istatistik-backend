from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

MAIL_USER = os.environ.get('MAIL_USER')
MAIL_PASS = os.environ.get('MAIL_PASS')
HOCA_MAIL = os.environ.get('HOCA_MAIL')

orders = {}

def send_mail(to, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = MAIL_USER
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(MAIL_USER, MAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Mail hatası: {e}")
        return False

def dosyadan_df(dosya):
    fname = dosya.filename.lower()
    if fname.endswith('.csv'):
        return pd.read_csv(dosya)
    elif fname.endswith(('.xlsx', '.xls')):
        return pd.read_excel(dosya)
    return None

# ============================================================
# NORMALLİK TESTİ - AI Advisor'dan çağrılır
# ============================================================
@app.route('/normallik-testi', methods=['POST'])
def normallik_testi():
    try:
        dosya = request.files.get('dataFile')
        if not dosya:
            return jsonify({"hata": "Dosya bulunamadı"}), 400

        df = dosyadan_df(dosya)
        if df is None:
            return jsonify({"hata": "Desteklenmeyen format. Excel veya CSV yükleyin."}), 400

        sayisal_kolonlar = df.select_dtypes(include=[np.number]).columns.tolist()
        if not sayisal_kolonlar:
            return jsonify({"hata": "Dosyada sayısal veri bulunamadı."}), 400

        sonuclar = []
        genel_normal = True

        for kolon in sayisal_kolonlar[:10]:
            veri = df[kolon].dropna().values
            if len(veri) < 3:
                continue
            if len(veri) <= 5000:
                stat, p = stats.shapiro(veri)
                test_adi = "Shapiro-Wilk"
            else:
                stat, p = stats.kstest(veri, 'norm', args=(np.mean(veri), np.std(veri)))
                test_adi = "Kolmogorov-Smirnov"

            normal = p > 0.05
            if not normal:
                genel_normal = False

            sonuclar.append({
                "kolon": kolon,
                "n": int(len(veri)),
                "ortalama": round(float(np.mean(veri)), 3),
                "std": round(float(np.std(veri)), 3),
                "test": test_adi,
                "istatistik": round(float(stat), 4),
                "p_degeri": round(float(p), 4),
                "normal": normal,
                "yorum": "Normal dağılım ✓" if normal else "Normal dağılım değil ✗"
            })

        grup_sayisi = len(sayisal_kolonlar)
        oneriler = []

        if genel_normal:
            if grup_sayisi == 2:
                oneriler = [
                    {"test": "t-test", "neden": "2 grup + normal dağılım → parametrik test"},
                    {"test": "levene", "neden": "Varyans homojenliğini kontrol et"},
                ]
            elif grup_sayisi >= 3:
                oneriler = [
                    {"test": "anova", "neden": "3+ grup + normal dağılım → ANOVA"},
                    {"test": "levene", "neden": "ANOVA için varyans homojenliği şart"},
                ]
            oneriler.append({"test": "pearson", "neden": "Normal dağılımda korelasyon için Pearson"})
        else:
            if grup_sayisi == 2:
                oneriler = [
                    {"test": "mann-whitney", "neden": "2 grup + normal dağılım yok → non-parametrik"},
                ]
            elif grup_sayisi >= 3:
                oneriler = [
                    {"test": "kruskal-wallis", "neden": "3+ grup + normal dağılım yok → non-parametrik ANOVA"},
                ]
            oneriler.append({"test": "spearman", "neden": "Normal dağılım yoksa korelasyon için Spearman"})

        oneriler.append({"test": "shapiro-wilk", "neden": "Normallik testi (rapor için)"})

        return jsonify({
            "basarili": True,
            "genel_normal": genel_normal,
            "kolonlar": sonuclar,
            "onerilen_testler": oneriler,
            "ozet": "Veriler normal dağılım gösteriyor." if genel_normal else "Veriler normal dağılım göstermiyor."
        })

    except Exception as e:
        print(f"Normallik testi hatası: {e}")
        return jsonify({"hata": str(e)}), 500


# ============================================================
# İSTATİSTİK TESTLERİ
# ============================================================
def run_tests(df, selected_tests):
    results = []
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        return [{"hata": "Sayısal sütun bulunamadı"}]

    col1 = df[cols[0]].dropna().values
    col2 = df[cols[1]].dropna().values if len(cols) > 1 else col1

    for test in selected_tests:
        try:
            if test == "shapiro-wilk":
                stat, p = stats.shapiro(col1)
                results.append({"test": "Shapiro-Wilk", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": "Normal dağılım var" if p > 0.05 else "Normal dağılım YOK"})
            elif test == "levene":
                stat, p = stats.levene(col1, col2)
                results.append({"test": "Levene", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": "Varyanslar homojen" if p > 0.05 else "Varyanslar homojen DEĞİL"})
            elif test == "t-test":
                stat, p = stats.ttest_ind(col1, col2)
                results.append({"test": "T-Testi", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": "Anlamlı fark VAR" if p < 0.05 else "Anlamlı fark YOK"})
            elif test == "anova":
                gruplar = [df[c].dropna().values for c in cols[:3]]
                stat, p = stats.f_oneway(*gruplar)
                results.append({"test": "ANOVA", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": "Anlamlı fark VAR" if p < 0.05 else "Anlamlı fark YOK"})
            elif test == "kruskal-wallis":
                gruplar = [df[c].dropna().values for c in cols[:3]]
                stat, p = stats.kruskal(*gruplar)
                results.append({"test": "Kruskal-Wallis", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": "Anlamlı fark VAR" if p < 0.05 else "Anlamlı fark YOK"})
            elif test == "mann-whitney":
                stat, p = stats.mannwhitneyu(col1, col2)
                results.append({"test": "Mann-Whitney U", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": "Anlamlı fark VAR" if p < 0.05 else "Anlamlı fark YOK"})
            elif test == "pearson":
                stat, p = stats.pearsonr(col1, col2)
                results.append({"test": "Pearson", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": f"r={round(float(stat),4)}, {'anlamlı' if p < 0.05 else 'anlamlı değil'}"})
            elif test == "spearman":
                stat, p = stats.spearmanr(col1, col2)
                results.append({"test": "Spearman", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": f"r={round(float(stat),4)}, {'anlamlı' if p < 0.05 else 'anlamlı değil'}"})
            elif test == "ki-kare":
                stat, p = stats.chisquare(col1)
                results.append({"test": "Ki-Kare", "istatistik": round(float(stat), 4), "p_degeri": round(float(p), 4),
                                 "yorum": "Anlamlı fark VAR" if p < 0.05 else "Anlamlı fark YOK"})
            elif test == "cronbach":
                n = len(cols)
                item_var = df[cols].var(axis=0, ddof=1).sum()
                total_var = df[cols].sum(axis=1).var(ddof=1)
                alpha = (n / (n - 1)) * (1 - item_var / total_var)
                results.append({"test": "Cronbach Alpha", "istatistik": round(float(alpha), 4), "p_degeri": None,
                                 "yorum": "Yüksek güvenilirlik" if alpha > 0.80 else ("Kabul edilebilir" if alpha > 0.70 else "Düşük güvenilirlik")})
            elif test in ["dogrusal-regresyon", "lojistik-regresyon"]:
                slope, intercept, r, p, se = stats.linregress(col1, col2)
                results.append({"test": "Doğrusal Regresyon", "istatistik": round(float(r**2), 4), "p_degeri": round(float(p), 4),
                                 "yorum": f"R²={round(float(r**2),4)}, {'anlamlı' if p < 0.05 else 'anlamlı değil'}"})
            else:
                results.append({"test": test, "yorum": "Bu test uzman ekip tarafından yapılacak."})
        except Exception as e:
            results.append({"test": test, "hata": str(e)})

    return results


# ============================================================
# SİPARİŞ AL
# ============================================================
@app.route('/siparis', methods=['POST'])
def siparis_al():
    try:
        ad_soyad = request.form.get('fullName')
        email    = request.form.get('email')
        calisma  = request.form.get('studyType')
        ozet     = request.form.get('summary')
        testler  = request.form.getlist('tests')
        not_ek   = request.form.get('extraNote', '')

        if not all([ad_soyad, email, calisma, testler]):
            return jsonify({"hata": "Eksik bilgi"}), 400

        dosya = request.files.get('dataFile')
        test_sonuclari = []
        if dosya:
            df = dosyadan_df(dosya)
            if df is not None:
                test_sonuclari = run_tests(df, testler)

        siparis_id = str(uuid.uuid4())[:8].upper()
        siparis = {
            "id": siparis_id,
            "ad_soyad": ad_soyad,
            "email": email,
            "calisma_turu": calisma,
            "ozet": ozet,
            "testler": testler,
            "not_ek": not_ek,
            "test_sonuclari": test_sonuclari,
            "tarih": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "durum": "beklemede"
        }
        orders[siparis_id] = siparis

        if HOCA_MAIL:
            sonuc_html = "".join([
                f"<tr><td><b>{s.get('test','')}</b></td><td>{s.get('istatistik','')}</td><td>{s.get('p_degeri','')}</td><td>{s.get('yorum','')}</td></tr>"
                for s in test_sonuclari
            ])
            hoca_mail = f"""
            <div style="font-family:Arial,sans-serif;">
            <h2 style="color:#3b82f6;">Yeni Sipariş: #{siparis_id}</h2>
            <p><b>Ad Soyad:</b> {ad_soyad}</p>
            <p><b>E-posta:</b> {email}</p>
            <p><b>Çalışma Türü:</b> {calisma}</p>
            <p><b>Özet:</b> {ozet}</p>
            <p><b>Seçili Testler:</b> {', '.join(testler)}</p>
            <p><b>Ek Not:</b> {not_ek}</p>
            <h3>Otomatik Analiz Sonuçları:</h3>
            <table border="1" cellpadding="8" style="border-collapse:collapse;">
                <tr style="background:#3b82f6;color:white;"><th>Test</th><th>İstatistik</th><th>P Değeri</th><th>Yorum</th></tr>
                {sonuc_html}
            </table>
            <br>
            <a href="https://istatistik-backend.onrender.com/onayla/{siparis_id}"
               style="background:#10b981;color:white;padding:10px 20px;border-radius:8px;text-decoration:none;margin-right:10px;">✅ Onayla</a>
            <a href="https://istatistik-backend.onrender.com/reddet/{siparis_id}"
               style="background:#f59e0b;color:white;padding:10px 20px;border-radius:8px;text-decoration:none;">⚠️ Reddet</a>
            </div>
            """
            send_mail(HOCA_MAIL, f"Yeni Sipariş - #{siparis_id}", hoca_mail)

        return jsonify({
            "basarili": True,
            "siparis_id": siparis_id,
            "mesaj": "Siparişiniz alındı, en kısa sürede sonuçlarınız tarafınıza iletilecektir."
        })

    except Exception as e:
        print(f"Sipariş hatası: {e}")
        return jsonify({"hata": str(e)}), 500


# ============================================================
# ONAYLA
# ============================================================
@app.route('/onayla/<siparis_id>', methods=['GET'])
def onayla(siparis_id):
    if siparis_id not in orders:
        return "Sipariş bulunamadı", 404

    siparis = orders[siparis_id]
    siparis['durum'] = 'onaylandi'

    if siparis.get('email'):
        sonuc_html = "".join([
            f"<tr><td><b>{s.get('test','')}</b></td><td>{s.get('istatistik','')}</td><td>{s.get('p_degeri','')}</td><td>{s.get('yorum','')}</td></tr>"
            for s in siparis.get('test_sonuclari', [])
        ])
        mail = f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
        <div style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);padding:30px;border-radius:12px 12px 0 0;text-align:center;">
            <h1 style="color:white;margin:0;">Akıllı İstatistik</h1>
        </div>
        <div style="background:#fff;padding:30px;border-radius:0 0 12px 12px;border:1px solid #e2e8f0;">
            <h2>Merhaba {siparis['ad_soyad']},</h2>
            <p style="color:#64748b;">Analiz talebiniz tamamlandı.</p>
            <div style="background:#f4f7f9;padding:15px;border-radius:8px;margin:20px 0;">
                <p><b>Sipariş No:</b> {siparis_id}</p>
                <p><b>Çalışma Türü:</b> {siparis['calisma_turu']}</p>
                <p><b>Yapılan Testler:</b> {', '.join(siparis['testler'])}</p>
            </div>
            <h3>Analiz Sonuçlarınız:</h3>
            <table border="1" cellpadding="10" style="border-collapse:collapse;width:100%;font-size:14px;">
                <tr style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:white;">
                    <th>Test</th><th>İstatistik</th><th>P Değeri</th><th>Yorum</th>
                </tr>
                {sonuc_html}
            </table>
            <br>
            <p style="color:#64748b;font-size:13px;">📧 akilliistatistik@gmail.com</p>
            <p style="color:#64748b;font-size:13px;">Akıllı İstatistik Ekibi</p>
        </div></div>
        """
        send_mail(siparis['email'], f"Analiz Sonuçlarınız Hazır - #{siparis_id}", mail)

    return f"""
    <html><body style="font-family:Arial;text-align:center;padding:50px;background:#f4f7f9;">
    <div style="max-width:500px;margin:0 auto;background:white;padding:40px;border-radius:16px;">
    <div style="font-size:60px;">✅</div>
    <h2 style="color:#10b981;">Sonuçlar Gönderildi</h2>
    <p style="color:#64748b;">{siparis['ad_soyad']} adresine mail iletildi.</p>
    </div></body></html>
    """


# ============================================================
# REDDET
# ============================================================
@app.route('/reddet/<siparis_id>', methods=['GET'])
def reddet(siparis_id):
    if siparis_id not in orders:
        return "Sipariş bulunamadı", 404

    siparis = orders[siparis_id]
    siparis['durum'] = 'reddedildi'

    if MAIL_USER:
        mail = f"""
        <div style="font-family:Arial,sans-serif;">
        <h2 style="color:#f59e0b;">⚠️ İnceleme Gerekiyor - #{siparis_id}</h2>
        <p><b>Müşteri:</b> {siparis['ad_soyad']}</p>
        <p><b>E-posta:</b> {siparis['email']}</p>
        <p><b>Çalışma Türü:</b> {siparis['calisma_turu']}</p>
        <p><b>Testler:</b> {', '.join(siparis['testler'])}</p>
        <p style="color:#ef4444;">Müşteri ile iletişime geçmeniz gerekmektedir.</p>
        </div>
        """
        send_mail(MAIL_USER, f"⚠️ İnceleme Gerekiyor - #{siparis_id}", mail)

    return f"""
    <html><body style="font-family:Arial;text-align:center;padding:50px;background:#f4f7f9;">
    <div style="max-width:500px;margin:0 auto;background:white;padding:40px;border-radius:16px;">
    <div style="font-size:60px;">⚠️</div>
    <h2 style="color:#f59e0b;">Sipariş Reddedildi</h2>
    <p style="color:#64748b;">Müşteriye mail GÖNDERİLMEDİ.</p>
    <p style="color:#64748b;">Ekibinize bildirim iletildi: <b>{siparis['email']}</b></p>
    </div></body></html>
    """


@app.route('/', methods=['GET'])
def home():
    return jsonify({"durum": "Akıllı İstatistik Backend Çalışıyor!"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
