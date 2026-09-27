from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

dosya_yolu = Path(__file__).parent / "SMSSpamCollection"

df = pd.read_csv(
    dosya_yolu,
    sep="\t",
    header=None,
    names=["etiket", "mesaj"],
    quoting=3,
    encoding="utf-8"
)

print("İlk 5 kayıt:")
print(df.head())

print("\nToplam kayıt:", len(df))

print("\nEtiket dağılımı:")
print(df["etiket"].value_counts())

print("\nEksik değer sayıları:")
print(df.isnull().sum())

print("\nTekrar eden kayıt sayısı:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

print("\nTemizlik sonrası toplam kayıt:", len(df))

print("\nTemizlik sonrası etiket dağılımı:")
print(df["etiket"].value_counts())

X = df["mesaj"]   # Girdi: mesaj metinleri
y = df["etiket"]  # Doğru cevap: ham veya spam

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nEğitim mesajı sayısı:", len(X_train))
print("Test mesajı sayısı:", len(X_test))

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nEğitim matrisi:", X_train_tfidf.shape)
print("Test matrisi:", X_test_tfidf.shape)

print("\nİlk 20 özellik:")
print(vectorizer.get_feature_names_out()[:20])

model = MultinomialNB()

# Eğitim mesajları ve doğru etiketleriyle modeli eğit
model.fit(X_train_tfidf, y_train)

# Eğitimde görmediği test mesajlarının etiketlerini tahmin et
tahminler = model.predict(X_test_tfidf)

karsilastirma = pd.DataFrame({
    "Gerçek etiket": y_test.to_numpy(),
    "Model tahmini": tahminler
})

print("\nİlk 10 tahmin:")
print(karsilastirma.head(10))

dogruluk = accuracy_score(y_test, tahminler)
print(f"\nDoğruluk: %{dogruluk * 100:.2f}")

print("\nSınıflandırma raporu:")
print(classification_report(y_test, tahminler, digits=3, zero_division=0))

matris = confusion_matrix(y_test, tahminler, labels=["ham", "spam"])

print("\nHata matrisi:")
print(pd.DataFrame(
    matris,
    index=["Gerçek normal", "Gerçek spam"],
    columns=["Tahmin normal", "Tahmin spam"]
))

# Metni sayılara dönüştürme ve model eğitimini birleştir
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

# Karşılaştırılacak alpha değerleri
parametreler = {
    "nb__alpha": [0.1, 0.3, 0.5, 1.0]
}

# Spam sınıfının F1 puanına göre seçim yap
spam_f1 = make_scorer(f1_score, pos_label="spam")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

arama = GridSearchCV(
    pipeline,
    parametreler,
    scoring=spam_f1,
    cv=cv
)

# Ham eğitim metinlerini veriyoruz; test verisini kullanmıyoruz
arama.fit(X_train, y_train)

print("\nAlpha karşılaştırması:")
for parametre, puan in zip(
    arama.cv_results_["params"],
    arama.cv_results_["mean_test_score"]
):
    print(f"Alpha: {parametre['nb__alpha']} | Ortalama spam F1: {puan:.3f}")

print("\nSeçilen ayarlar:", arama.best_params_)
print("En iyi ortalama F1:", round(arama.best_score_, 3))

en_iyi_model = arama.best_estimator_

# Pipeline ham metni alır, TF-IDF dönüşümünü kendisi yapar
yeni_tahminler = en_iyi_model.predict(X_test)

print("\nİyileştirilmiş modelin test sonuçları:")
print(classification_report(
    y_test, yeni_tahminler,
    digits=3,
    zero_division=0
))

yeni_matris = confusion_matrix(
    y_test, yeni_tahminler,
    labels=["ham", "spam"]
)

print(pd.DataFrame(
    yeni_matris,
    index=["Gerçek normal", "Gerçek spam"],
    columns=["Tahmin normal", "Tahmin spam"]
))

while True:
    mesaj = input("\nİngilizce mesaj yaz (çıkmak için q): ").strip()

    if mesaj.lower() == "q":
        break

    if not mesaj:
        print("Lütfen bir mesaj yaz.")
        continue

    tahmin = en_iyi_model.predict([mesaj])[0]

    if tahmin == "spam":
        print("Sonuç: SPAM")
    else:
        print("Sonuç: NORMAL")