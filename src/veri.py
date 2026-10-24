"""Veri setini okuma, temizleme ve eğitim/test ayrımı."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJE_KOKU = Path(__file__).resolve().parent.parent
VERI_DOSYASI = PROJE_KOKU / "data" / "SMSSpamCollection"
RASTGELE_TOHUM = 42


def veriyi_yukle(dosya_yolu=VERI_DOSYASI):
    """UCI SMS Spam Collection dosyasını (etiket<TAB>mesaj) okur."""
    if not Path(dosya_yolu).exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {dosya_yolu}\n"
            "README'deki 'Veri seti' bölümüne göre indirip data/ klasörüne koyun."
        )
    return pd.read_csv(
        dosya_yolu,
        sep="\t",
        header=None,
        names=["etiket", "mesaj"],
        quoting=3,
        encoding="utf-8",
    )


def temizle(df):
    """Eksik ve birebir tekrar eden kayıtları kaldırır, özet bilgiyi döndürür."""
    ozet = {
        "ham_kayit": len(df),
        "eksik_deger": int(df.isnull().sum().sum()),
        "tekrar_eden": int(df.duplicated().sum()),
    }
    # Tekrarlar ayrımdan ÖNCE silinir; aksi halde aynı mesaj hem eğitimde hem testte olabilir.
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    ozet["temiz_kayit"] = len(df)
    ozet["etiket_dagilimi"] = df["etiket"].value_counts().to_dict()
    return df, ozet


def ayir(df, test_orani=0.20):
    """Sınıf oranlarını koruyan eğitim/test ayrımı."""
    return train_test_split(
        df["mesaj"],
        df["etiket"],
        test_size=test_orani,
        random_state=RASTGELE_TOHUM,
        stratify=df["etiket"],
    )
