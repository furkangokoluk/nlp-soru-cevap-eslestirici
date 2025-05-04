# Türkçe Soru-Cevap Eşleştirici

Bu proje, Türkçe metinler üzerinde Word2Vec ve TF-IDF tabanlı doğal dil işleme tekniklerini kullanarak soru-cevap eşleştirme sistemi geliştirmeyi amaçlamaktadır.

## Veri Seti Kullanım Amacı

Bu veri seti şu amaçlarla kullanılabilir:
- Türkçe soru-cevap sistemleri geliştirme
- Metin benzerliği analizi
- Doğal dil işleme modelleri eğitimi
- Türkçe metin ön işleme pipeline'ı geliştirme
- Soru-cevap eşleştirme algoritmaları değerlendirme

## Özellikler

- Türkçe metin ön işleme (Zemberek kullanarak)
- Lemmatization ve stemming işlemleri
- TF-IDF tabanlı kelime benzerliği analizi
- Word2Vec model eğitimi (8 farklı konfigürasyon)
- Soru-cevap eşleştirme simülasyonu

## Gereksinimler ve Kurulum

### Sistem Gereksinimleri
- Python 3.8 veya üzeri
- Java Runtime Environment (Zemberek için)
- 4GB minimum RAM (Word2Vec eğitimi için)

### Gerekli Kütüphaneler
```bash
numpy>=1.21.0
pandas>=1.3.0
gensim>=4.1.0
scikit-learn>=0.24.0
zemberek-python>=0.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Kurulum Adımları

1. Projeyi klonlayın:
```bash
git clone [repo-url]
cd [repo-directory]
```

2. Virtual environment oluşturun (opsiyonel ama önerilen):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Zemberek'i yükleyin:
```bash
pip install zemberek-python
```

## Model Oluşturma Adımları

1. **Veri Ön İşleme**
```bash
python preprocessing.py
```
- Ham veriyi yükler
- Metin temizleme işlemlerini gerçekleştirir
- Lemmatization ve stemming uygular
- İşlenmiş verileri kaydeder

2. **TF-IDF Model Eğitimi**
```bash
python tfidf_model.py
```
- İşlenmiş verileri yükler
- TF-IDF vektörlerini hesaplar
- Modeli kaydeder (.pkl formatında)
- Benzerlik skorlarını hesaplar

3. **Word2Vec Model Eğitimi**
```bash
python word2vec_model.py
```
- 16 farklı parametre kombinasyonu için modeller eğitir
- Her modeli .model formatında kaydeder
- Vektörleri .pkl formatında kaydeder
- Performans metriklerini hesaplar

4. **Tüm Pipeline'ı Çalıştırma**
```bash
python main.py
```
- Tüm adımları sırayla gerçekleştirir
- Sonuçları ve metrikleri kaydeder

## Kullanım ve Verimlilik Özellikleri

Projeyi çalıştırmak için:

```bash
python main.py
```

### Akıllı Model Yönetimi
- Modeller sadece gerektiğinde (yeni veri eklendiğinde) güncellenir
- TF-IDF ve Word2Vec modelleri otomatik olarak kaydedilir/yüklenir
- Geribildirim değişiklikleri hash değeri ile kontrol edilir

### İşlem Adımları
1. Veri ön işleme ve Zipf analizi
2. TF-IDF model eğitimi/yükleme
3. Word2Vec modelleri eğitimi/yükleme
4. Performans metrikleri hesaplama

## Proje Yapısı

- `preprocessing.py`: Metin ön işleme işlemleri
- `tfidf_model.py`: TF-IDF tabanlı analiz
- `word2vec_model.py`: Word2Vec model eğitimi ve analizi
- `main.py`: Ana program akışı

## Çıktılar

Proje çalıştırıldığında aşağıdaki dosyalar oluşturulur:

- `data/processed_data.csv`: İşlenmiş veri seti
- `data/lemma_data.csv`: Lemmatize edilmiş veriler
- `data/stem_data.csv`: Stem'lenmiş veriler
- `results/tfidf_*_results.csv`: TF-IDF analiz sonuçları
- `models/*.model`: Eğitilmiş Word2Vec modelleri
- `models/word2vec_results_summary.csv`: Word2Vec sonuç özeti
