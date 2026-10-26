"""Modelleri çapraz doğrulamayla karşılaştırır, en iyisini test eder ve kaydeder.

Çalıştırma (proje klasöründen):  python src/egit.py
Çıktılar:
    models/spam_model.joblib        seçilen model (TF-IDF + sınıflandırıcı pipeline'ı)
    reports/metrics.json            tüm sayısal sonuçlar
    reports/figures/*.png           model karşılaştırması ve hata matrisi
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from veri import PROJE_KOKU, RASTGELE_TOHUM, ayir, temizle, veriyi_yukle

MODEL_DOSYASI = PROJE_KOKU / "models" / "spam_model.joblib"
RAPOR_KLASORU = PROJE_KOKU / "reports"
GRAFIK_KLASORU = RAPOR_KLASORU / "figures"

# Seçim ölçütü: spam sınıfının F1 puanı. Veri dengesiz (~%13 spam) olduğu için
# doğruluk (accuracy) yanıltıcıdır: her şeye "normal" diyen model bile ~%87 alır.
SPAM_F1 = make_scorer(f1_score, pos_label="spam")

# Karşılaştırılan modeller ve denenen ayarlar.
# Her pipeline TF-IDF'i çapraz doğrulamanın her turunda yalnızca o turun eğitim parçasında öğrenir.
ADAYLAR = {
    "Naive Bayes": (
        MultinomialNB(),
        {"tfidf__ngram_range": [(1, 1), (1, 2)],
         "model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, solver="liblinear"),
        {"tfidf__ngram_range": [(1, 1), (1, 2)],
         "model__C": [1, 10, 100, 1000],
         "model__class_weight": [None, "balanced"]},
    ),
    "Linear SVM": (
        LinearSVC(random_state=RASTGELE_TOHUM),
        {"tfidf__ngram_range": [(1, 1), (1, 2)],
         "model__C": [0.1, 1, 10],
         "model__class_weight": [None, "balanced"]},
    ),
}


def pipeline_kur(siniflandirici):
    return Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)), ("model", siniflandirici)])


def spam_skoru(model, X):
    """ROC için sürekli bir 'spam olma' puanı: SVM/LogReg'de karar fonksiyonu, NB'de olasılık."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, list(model.classes_).index("spam")]


def test_olcutleri(y_gercek, y_tahmin):
    matris = confusion_matrix(y_gercek, y_tahmin, labels=["ham", "spam"])
    return {
        "dogruluk": accuracy_score(y_gercek, y_tahmin),
        "spam_precision": precision_score(y_gercek, y_tahmin, pos_label="spam"),
        "spam_recall": recall_score(y_gercek, y_tahmin, pos_label="spam"),
        "spam_f1": f1_score(y_gercek, y_tahmin, pos_label="spam"),
        "hata_matrisi": {
            "dogru_normal": int(matris[0, 0]), "yanlis_alarm": int(matris[0, 1]),
            "kacirilan_spam": int(matris[1, 0]), "yakalanan_spam": int(matris[1, 1]),
        },
    }


def main():
    df, veri_ozeti = temizle(veriyi_yukle())
    X_egitim, X_test, y_egitim, y_test = ayir(df)
    print(f"Temiz kayıt: {veri_ozeti['temiz_kayit']}  (silinen tekrar: {veri_ozeti['tekrar_eden']})")
    print(f"Eğitim: {len(X_egitim)}  Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASTGELE_TOHUM)

    # 1) Başlangıç modeli: varsayılan ayarlı Naive Bayes (eski main.py'deki ilk model)
    baslangic = Pipeline([("tfidf", TfidfVectorizer()), ("model", MultinomialNB())])
    baslangic.fit(X_egitim, y_egitim)

    # 2) Her aday için ayar araması; seçim YALNIZCA eğitim verisindeki CV puanıyla yapılır
    sonuclar = {}
    for ad, (siniflandirici, ayarlar) in ADAYLAR.items():
        arama = GridSearchCV(pipeline_kur(siniflandirici), ayarlar, scoring=SPAM_F1, cv=cv, n_jobs=-1)
        arama.fit(X_egitim, y_egitim)
        en_iyi = arama.best_index_
        sonuclar[ad] = {
            "arama": arama,
            "cv_f1_ort": float(arama.cv_results_["mean_test_score"][en_iyi]),
            "cv_f1_std": float(arama.cv_results_["std_test_score"][en_iyi]),
            "en_iyi_ayarlar": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in arama.best_params_.items()},
        }
        print(f"{ad:20s} CV spam F1 = {sonuclar[ad]['cv_f1_ort']:.3f} ± {sonuclar[ad]['cv_f1_std']:.3f}"
              f"  {sonuclar[ad]['en_iyi_ayarlar']}")

    secilen_ad = max(sonuclar, key=lambda ad: sonuclar[ad]["cv_f1_ort"])
    secilen_model = sonuclar[secilen_ad]["arama"].best_estimator_
    print(f"\nSeçilen model: {secilen_ad}")

    # 3) Test verisi yalnızca bir kez, seçimden sonra kullanılır
    y_tahmin = secilen_model.predict(X_test)
    print("\nTest sonuçları:")
    print(classification_report(y_test, y_tahmin, digits=3))

    metrikler = {
        "veri": veri_ozeti | {"egitim": len(X_egitim), "test": len(X_test)},
        "cv_karsilastirma": {ad: {k: v for k, v in s.items() if k != "arama"} for ad, s in sonuclar.items()},
        "secilen_model": secilen_ad,
        "test": {
            "baslangic_nb_alpha_1": test_olcutleri(y_test, baslangic.predict(X_test)),
            "secilen_model": test_olcutleri(y_test, y_tahmin),
        },
    }

    MODEL_DOSYASI.parent.mkdir(parents=True, exist_ok=True)
    GRAFIK_KLASORU.mkdir(parents=True, exist_ok=True)
    joblib.dump(secilen_model, MODEL_DOSYASI)

    # Grafik 3: ROC eğrileri (test verisi). Eşik değiştikçe yakalanan spam oranı (TPR) ile
    # yanlış alarm oranı (FPR) arasındaki denge; AUC = eğrinin altındaki alan (1 = kusursuz).
    y_test_ikili = (y_test == "spam").astype(int)
    modeller = {"Başlangıç NB (alpha=1)": baslangic} | {ad: s["arama"].best_estimator_ for ad, s in sonuclar.items()}
    fig, ax = plt.subplots(figsize=(5, 4.5))
    metrikler["test_roc_auc"] = {}
    for ad, model in modeller.items():
        skor = spam_skoru(model, X_test)
        fpr, tpr, _ = roc_curve(y_test_ikili, skor)
        auc = roc_auc_score(y_test_ikili, skor)
        metrikler["test_roc_auc"][ad] = auc
        ax.plot(fpr, tpr, lw=2.5 if ad == secilen_ad else 1.2, label=f"{ad} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax.set(xlabel="Yanlış alarm oranı (FPR)", ylabel="Yakalanan spam oranı (TPR)", title="ROC eğrileri (test verisi)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFIK_KLASORU / "roc_egrisi.png", dpi=150)
    plt.close(fig)

    # Grafik 4: öğrenme eğrisi. Eğitim verisi büyüdükçe eğitim ve doğrulama F1'i nasıl değişiyor?
    # İki eğri birbirine yaklaşıyorsa model ezberlemiyor; doğrulama eğrisi hâlâ yükseliyorsa
    # daha fazla veri işe yarar.
    boyutlar, egitim_puan, dogrulama_puan = learning_curve(
        secilen_model, X_egitim, y_egitim, cv=cv, scoring=SPAM_F1,
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for puan, etiket, renk in [(egitim_puan, "Eğitim", "#4C78A8"), (dogrulama_puan, "Doğrulama (CV)", "#E45756")]:
        ax.plot(boyutlar, puan.mean(axis=1), "o-", color=renk, label=etiket)
        ax.fill_between(boyutlar, puan.mean(axis=1) - puan.std(axis=1), puan.mean(axis=1) + puan.std(axis=1),
                        color=renk, alpha=0.15)
    ax.set(xlabel="Eğitim mesajı sayısı", ylabel="Spam F1", title=f"Öğrenme eğrisi: {secilen_ad}", ylim=(0.5, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFIK_KLASORU / "ogrenme_egrisi.png", dpi=150)
    plt.close(fig)
    metrikler["ogrenme_egrisi"] = {"egitim_boyutu": boyutlar.tolist(),
                                   "egitim_f1": egitim_puan.mean(axis=1).tolist(),
                                   "dogrulama_f1": dogrulama_puan.mean(axis=1).tolist()}

    (RAPOR_KLASORU / "metrics.json").write_text(
        json.dumps(metrikler, indent=2, ensure_ascii=False), encoding="utf-8")

    # Grafik 1: modellerin CV puanları (hata çubuğu = 5 katın standart sapması)
    adlar = list(sonuclar)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(adlar, [sonuclar[a]["cv_f1_ort"] for a in adlar],
           yerr=[sonuclar[a]["cv_f1_std"] for a in adlar], capsize=6,
           color=["#4C78A8" if a != secilen_ad else "#E45756" for a in adlar])
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("5-katlı CV spam F1")
    ax.set_title("Model karşılaştırması (eğitim verisi)")
    fig.tight_layout()
    fig.savefig(GRAFIK_KLASORU / "model_karsilastirma.png", dpi=150)
    plt.close(fig)

    # Grafik 2: seçilen modelin test hata matrisi
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_tahmin, labels=["ham", "spam"], display_labels=["Normal", "Spam"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"Test hata matrisi: {secilen_ad}")
    fig.tight_layout()
    fig.savefig(GRAFIK_KLASORU / "hata_matrisi.png", dpi=150)
    plt.close(fig)

    print(f"Model kaydedildi: {MODEL_DOSYASI.relative_to(PROJE_KOKU)}")
    print(f"Sonuçlar: reports/metrics.json, grafikler: reports/figures/")


if __name__ == "__main__":
    main()
