# SMS Spam Detection

İngilizce SMS mesajlarını **spam** veya **normal (ham)** olarak sınıflandıran, Python ile geliştirilmiş bir makine öğrenmesi projesi.

## Yöntem

- Pandas ile veri okuma ve eksik değer kontrolü
- Birebir tekrar eden kayıtların kaldırılması
- Sınıf oranlarını koruyan %80 eğitim / %20 test ayrımı (`random_state=42`)
- TF-IDF ile metinlerin sayısal özelliklere dönüştürülmesi
- Multinomial Naive Bayes ile başlangıç modeli
- Pipeline ve 5 katlı StratifiedKFold ile alpha karşılaştırması
- Eğitim verisindeki ortalama spam F1 puanına göre model seçimi
- Test değerlendirmesi ve terminalden etkileşimli mesaj sınıflandırma

TF-IDF her çapraz doğrulama turunda yalnızca ilgili eğitim parçasında öğrenilir. Alpha seçenekleri: 0.1, 0.3, 0.5, 1.0.

## Kurulum ve çalıştırma

Python 3.11 veya üzeri kullanın. Proje klasöründe bir sanal ortam oluşturun:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Bağımlılıkları kurun:

```bash
python -m pip install -r requirements.txt
```

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) sayfasından veri setini indirin. ZIP içindeki `SMSSpamCollection` dosyasını, adını değiştirmeden `main.py` ile aynı klasöre koyun. Veri dosyası depoya dahil edilmemiştir.

```bash
python main.py
```

Program önce modelleri eğitir ve sonuçları yazdırır, ardından İngilizce mesaj girmenizi bekler. Çıkmak için `q` yazın. Her çalıştırmada eğitim tekrar yapılır; eğitilmiş model diske kaydedilmez.

## Veri seti

Kullanılan dosyada 5.574 kayıt vardır. 403 birebir tekrar kaldırıldıktan sonra 5.171 kayıt kalır: 4.518 normal ve 653 spam. Eğitimde 4.136, testte 1.035 mesaj kullanılır.

Kaynak: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CC84.

Veri seti lisansı: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Bu lisans bilgisi veri setine aittir.

## Test sonuçları

Aşağıdaki sonuçlar geliştirme sırasında alınan test çıktılarıdır. Seçilen alpha: **0.1**. Çapraz doğrulama ortalama spam F1: **0.925**.

| Ölçüt | Başlangıç (alpha=1.0) | Seçilen model (alpha=0.1) |
|---|---:|---:|
| Genel doğruluk | %95.27 | %98.16 |
| Spam precision | %100.00 | %96.67 |
| Spam recall | %62.60 | %88.55 |
| Spam F1 | 0.770 | 0.924 |
| Yakalanan spam | 82 | 116 |
| Kaçırılan spam | 49 | 15 |
| Normal mesajda yanlış alarm | 0 | 4 |

Seçilen modelin hata matrisi:

| Gerçek sınıf | Tahmin normal | Tahmin spam |
|---|---:|---:|
| Normal | 900 | 4 |
| Spam | 15 | 116 |

## Sınırlar

Bu bir öğrenme projesidir. Sonuçlar bu veri seti ve bölünmesi için geçerlidir; güncel veya farklı kaynaklardan gelen mesajlarda aynı başarı garanti edilmez. İngilizce verilerle eğitilmiştir; Türkçe için değerlendirilmemiştir. Yakın benzer mesajlar birebir tekrar temizliğinden sonra kalabilir. Başlangıç ve seçilen model aynı test grubunda raporlanmıştır; alpha seçimi eğitim içindeki çapraz doğrulamayla yapılmıştır. Üretim kullanımı öncesinde yeni ve bağımsız verilerle değerlendirilmelidir.
