import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from preprocessing import TextPreprocessor
from tfidf_model import TFIDFModel
from word2vec_model import Word2VecModel
from qa_matcher import QAMatcher
import pandas as pd

def create_directories():
    """Gerekli dizinleri oluştur"""
    for directory in ['data', 'models', 'results']:
        os.makedirs(directory, exist_ok=True)


def analyze_zipf_law(texts, title, output_path):
    """Zipf yasası analizi yap ve log-log grafiği çiz"""
    # Dosya zaten varsa, analizi atla
    if os.path.exists(output_path):
        print(f"Zipf grafiği zaten mevcut, atlanıyor: {output_path}")
        return None, "Mevcut dosya kullanıldı."
    
    print(f"Zipf analizi yapılıyor: {title}")
    start_time = time.time()
    
    all_text = ' '.join(texts)
    words = re.findall(r'\b\w+\b', all_text)
    word_counts = Counter(words)
    frequencies = sorted(word_counts.values(), reverse=True)
    ranks = np.arange(1, len(frequencies) + 1)

    # Log-log grafiği
    plt.figure(figsize=(12, 8))
    plt.loglog(ranks, frequencies, 'bo', markersize=3, alpha=0.5)
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)
    mask = ~np.isnan(log_frequencies)
    log_ranks_filtered = log_ranks[mask]
    log_frequencies_filtered = log_frequencies[mask]
    coefficients = np.polyfit(log_ranks_filtered, log_frequencies_filtered, 1)
    polynomial = np.poly1d(coefficients)
    slope = coefficients[0]

    plt.loglog(ranks, np.exp(polynomial(log_ranks)), 'r-', linewidth=2)
    plt.title(f'{title} - Zipf Yasası (Log-Log)', fontsize=16)
    plt.xlabel('Kelime Sırası (Rank)')
    plt.ylabel('Kelime Frekansı')
    plt.text(0.7, 0.9, f'Eğim (α) = {slope:.3f}', transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    interpretation = ""
    if -1.2 < slope < -0.8:
        interpretation = "Doğal dile oldukça uygun."
    elif -1.5 < slope < -0.5:
        interpretation = "Kısmen doğal dil özellikleri taşıyor."
    else:
        interpretation = "Doğal dilden sapmalar içeriyor."

    processing_time = time.time() - start_time
    print(f"{title} -> Eğim: {slope:.3f} | Yorum: {interpretation} | Süre: {processing_time:.2f} saniye")
    return slope, interpretation


def analyze_zipf_for_all_data(processed_file):
    """Tüm veri tipleri için Zipf analizi yap"""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Orijinal veri için Zipf analizi
    print("\nZipf yasası analizleri başlıyor...")
    
    # Verileri oku
    df = pd.read_csv(processed_file)
    
    # Ham metin (temizlenmiş ama lemma/stem yapılmamış)
    raw_texts = df['clean_question'].tolist() + df['clean_answer'].tolist()
    raw_output_path = os.path.join(results_dir, 'zipf_raw_data.png')
    
    # Lemma ve stem verileri
    lemma_texts = df['lemma_question'].tolist() + df['lemma_answer'].tolist()
    stem_texts = df['stem_question'].tolist() + df['stem_answer'].tolist()
    lemma_output_path = os.path.join(results_dir, 'zipf_lemma_data.png')
    stem_output_path = os.path.join(results_dir, 'zipf_stem_data.png')
    
    # Analizler
    raw_slope, raw_interp = analyze_zipf_law(raw_texts, 'Ham Veri Sonuçları', raw_output_path)
    lemma_slope, lemma_interp = analyze_zipf_law(lemma_texts, 'Lemmatization Sonuçları', lemma_output_path)
    stem_slope, stem_interp = analyze_zipf_law(stem_texts, 'Stemming Sonuçları', stem_output_path)
    
    # Sonuçları bir DataFrame'e dönüştür
    if raw_slope is not None and lemma_slope is not None and stem_slope is not None:
        results_df = pd.DataFrame({
            'Veri Tipi': ['Ham Veri', 'Lemmatization', 'Stemming'],
            'Eğim': [raw_slope, lemma_slope, stem_slope],
            'Yorum': [raw_interp, lemma_interp, stem_interp]
        })
        results_df.to_csv(os.path.join(results_dir, 'zipf_analysis_results.csv'), index=False)
        print("\nZipf analizi sonuçları:")
        print(results_df)


def main():
    # Başlangıç zamanını kaydet
    main_start_time = time.time()
    
    # Gerekli dizinleri oluştur
    create_directories()

    # 1. Veri ön işleme (sadece ilk kez)
    processed_file = 'data/processed_data.csv'
    if not os.path.exists(processed_file):
        print("Veri ön işleme başlıyor...")
        preprocessor = TextPreprocessor()
        preprocessor.process_dataset(
            'squad-tr-train-v1.0.0-excluded.json',
            'data'
        )
    else:
        print("Önceden işlenmiş veri bulundu, atlanıyor: {}".format(processed_file))

    # 2. TF-IDF analizi (sadece sonuç dosyası yoksa)
    for prefix in ['lemma', 'stem']:
        result_csv = f'results/tfidf_{prefix}_results.csv'
        matrix_csv = f'results/tfidf_{"lemmatized" if prefix=='lemma' else 'stemmed'}.csv'
        if not os.path.exists(result_csv) or not os.path.exists(matrix_csv):
            print(f"TF-IDF analizi ({prefix}) başlıyor...")
            tfidf_model = TFIDFModel()
            tfidf_model.process_data(processed_file, 'results')
            break  # process_data hazırda hem lemma hem stem'i çalıştırır
        else:
            print(f"{prefix} için TF-IDF sonuç dosyası bulundu, atlanıyor: {result_csv}")

    # 3. Zipf yasası analizi
    zipf_results = 'results/zipf_analysis_results.csv'
    if not os.path.exists(zipf_results):
        print("Zipf yasası analizi başlıyor...")
        analyze_zipf_for_all_data(processed_file)
    else:
        print(f"Zipf analizi sonuçları bulundu, atlanıyor: {zipf_results}")
    
    # 4. Word2Vec model eğitimi (sadece sonuç dosyası yoksa)
    w2v_summary = 'models/word2vec_results_summary.csv'
    if not os.path.exists(w2v_summary):
        print("Word2Vec model eğitimi başlıyor...")
        word2vec_model = Word2VecModel()
        word2vec_model.train_all_configurations(processed_file, 'models')
    else:
        print(f"Word2Vec özet sonucu bulundu, atlanıyor: {w2v_summary}")

    # 5. Soru-Cevap Eşleştirici (kaydedilmiş model varsa onu kullanır)
    print("Soru-Cevap Eşleştirici başlatılıyor...")
    qa_matcher = QAMatcher(model_type='tfidf')
    qa_matcher.load_data(processed_file)
    
    # Eğitim süresini ölç
    start_time = time.time()
    qa_matcher.train()
    training_time = time.time() - start_time
    print(f"QA Matcher eğitimi/yüklenmesi {training_time:.2f} saniye sürdü.")

    # Test sorusu
    test_question = "Python nedir?"
    print(f"\nTest sorusu: {test_question}")
    
    # Sorgu süresini ölç
    start_time = time.time()
    results = qa_matcher.find_best_answers(test_question)
    query_time = time.time() - start_time
    print(f"Sorgu süresi: {query_time:.4f} saniye")

    print("\nEn iyi 3 cevap:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Cevap: {result['answer']}")
        print(f"   Benzerlik: {result['similarity']:.4f}")
    
    # Geribildirim ekleme testi
    if results:
        print("\nGeribildirim ekleme testi:")
        # İlk geribildirim
        updated = qa_matcher.add_feedback(test_question, results[0]['answer'], True)
        print(f"İlk geribildirim eklendi, güncelleme gerekli mi: {updated}")
        
        # Aynı geribildirim tekrar eklenirse güncelleme gerekmez
        updated = qa_matcher.add_feedback(test_question, results[0]['answer'], True)
        print(f"Aynı geribildirim tekrar eklendi, güncelleme gerekli mi: {updated}")
        
        # Model güncelleme testi (gerçek uygulamada minimum 10 geribildirim gerekir)
        print("\nModel güncelleme testi:")
        updated = qa_matcher.update_model()
        print(f"Model güncelleme sonucu: {updated}")

    # 6. Sonuç analizi çıktısı
    print("\nTF-IDF Sonuçları:")
    for prefix in ['lemma', 'stem']:
        result_csv = f'results/tfidf_{prefix}_results.csv'
        try:
            df = pd.read_csv(result_csv)
            print(f"\n{prefix} için sonuçlar:")
            print(df.head())
        except FileNotFoundError:
            print(f"{prefix} sonuçları okunamadı: {result_csv}")

    print("\nWord2Vec Sonuçları:")
    try:
        df_w2v = pd.read_csv(w2v_summary)
        if not df_w2v.empty:
            print(df_w2v.sort_values('similarity', ascending=False).head(10))
        else:
            print("Word2Vec sonuçları boş!")
    except FileNotFoundError:
        print(f"Word2Vec sonuç dosyası bulunamadı: {w2v_summary}")

    # Toplam işlem süresini göster
    total_time = time.time() - main_start_time
    print(f"\nToplam işlem süresi: {total_time:.2f} saniye")
    print("\nİşlem tamamlandı!")

if __name__ == "__main__":
    main()
