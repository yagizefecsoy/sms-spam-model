"""Kaydedilmiş modelle yeni mesajları sınıflandırır.

Kullanım (proje klasöründen, önce python src/egit.py çalıştırılmış olmalı):
    python src/tahmin.py "Congratulations! You won a free ticket, call now"
    python src/tahmin.py            # etkileşimli mod, çıkmak için q
"""
import sys
from pathlib import Path

import joblib

MODEL_DOSYASI = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"


def modeli_yukle():
    if not MODEL_DOSYASI.exists():
        sys.exit("Model bulunamadı. Önce eğitin: python src/egit.py")
    return joblib.load(MODEL_DOSYASI)


def sinifla(model, mesaj):
    return "SPAM" if model.predict([mesaj])[0] == "spam" else "NORMAL"


def main():
    model = modeli_yukle()

    if len(sys.argv) > 1:
        for mesaj in sys.argv[1:]:
            print(f"{sinifla(model, mesaj):6s} | {mesaj}")
        return

    while True:
        mesaj = input("\nİngilizce mesaj yaz (çıkmak için q): ").strip()
        if mesaj.lower() == "q":
            break
        if not mesaj:
            print("Lütfen bir mesaj yaz.")
            continue
        print("Sonuç:", sinifla(model, mesaj))


if __name__ == "__main__":
    main()
