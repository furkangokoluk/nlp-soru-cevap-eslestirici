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
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
gensim>=4.3.1
nltk>=3.8.1
zemberek-python>=0.1.0
tqdm>=4.65.0
matplotlib>=3.7.1
seaborn>=0.12.2
fpdf>=1.7.2
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

3. **Word2Vec Model Eğitimi**
```bash
python word2vec_model.py
```
- 16 farklı parametre kombinasyonu için modeller eğitir
- Her modeli .model formatında kaydeder
- Vektörleri .pkl formatında kaydeder
- Performans metriklerini hesaplar

4. **Benzerlik Analizi**
```bash
python similarity_analysis.py
```
- TF-IDF ve Word2Vec modellerini yükler
- Metin benzerliği hesaplar
- Jaccard benzerlik matrisi oluşturur
- Benzerlik sonuçlarını görselleştirir
- Sonuçları CSV formatında kaydeder

5. **Model Değerlendirmesi**
```bash
python model_evaluation.py
```
- Benzerlik sonuçlarını yükler
- Otomatik puan hesaplama
- Model performans karşılaştırması
- Jaccard benzerlik analizi
- Model yapılandırma analizi
- Değerlendirme sonuçlarını görselleştirir

6. **Tüm Pipeline'ı Çalıştırma**
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
5. Benzerlik analizi ve model değerlendirmesi
6. Soru-cevap eşleştirme ve geribildirim sistemi

### Yeni Özellikler
- **Otomatik Model Değerlendirmesi**: Model performansını otomatik olarak değerlendirir ve puanlar
- **Benzerlik Analizi**: TF-IDF ve Word2Vec modelleri arasında benzerlik karşılaştırması yapar
- **Jaccard Benzerlik Matrisi**: Modeller arası tutarlılığı görselleştirir
- **Geribildirim Sistemi**: Kullanıcı geribildirimlerine göre modelleri günceller
- **Performans Metrikleri**: Detaylı model performans analizi ve görselleştirme

## Proje Yapısı

- `preprocessing.py`: Metin ön işleme işlemleri
- `tfidf_model.py`: TF-IDF tabanlı analiz
- `word2vec_model.py`: Word2Vec model eğitimi ve analizi
- `similarity_analysis.py`: Metin benzerliği analizi ve karşılaştırma
- `model_evaluation.py`: Model performans değerlendirmesi ve metrikler
- `qa_matcher.py`: Soru-cevap eşleştirme ve geribildirim sistemi
- `test_feedback.py`: Geribildirim sistemi testleri
- `main.py`: Ana program akışı

## Çıktılar

Proje çalıştırıldığında aşağıdaki dosyalar oluşturulur:

- `data/processed_data.csv`: İşlenmiş veri seti
- `data/lemma_data.csv`: Lemmatize edilmiş veriler
- `data/stem_data.csv`: Stem'lenmiş veriler
- `results/tfidf_*_results.csv`: TF-IDF analiz sonuçları
- `results/similarity_results.csv`: Benzerlik analizi sonuçları
- `results/model_similarity_matrix.png`: Model benzerlik matrisi görselleştirmesi
- `results/zipf_analysis_results.csv`: Zipf yasası analiz sonuçları
- `results/zipf_*_data.png`: Zipf yasası görselleştirmeleri
- `evaluation/evaluation_results.csv`: Model değerlendirme sonuçları
- `evaluation/model_scores.png`: Model performans karşılaştırma grafiği
- `evaluation/jaccard_similarity_matrix.png`: Jaccard benzerlik matrisi görselleştirmesi
- `evaluation/jaccard_analysis.csv`: Jaccard benzerlik analizi sonuçları
- `evaluation/model_configuration_analysis.csv`: Model yapılandırma analizi
- `models/*.model`: Eğitilmiş Word2Vec modelleri
- `models/word2vec_results_summary.csv`: Word2Vec sonuç özeti
