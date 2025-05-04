from qa_matcher import QAMatcher
import time

def test_feedback_system():
    print("Soru-Cevap Eşleştirici başlatılıyor...")
    qa_matcher = QAMatcher(model_type='tfidf')
    qa_matcher.load_data('data/processed_data.csv')
    
    # Eğitim süresini ölç
    start_time = time.time()
    qa_matcher.train()
    training_time = time.time() - start_time
    print(f"QA Matcher eğitimi/yüklenmesi {training_time:.2f} saniye sürdü.\n")

    # Test soruları ve cevapları
    test_cases = [
        ("Python nedir?", "Python, yüksek seviyeli, genel amaçlı bir programlama dilidir.", True),
        ("Java ile Python arasındaki fark nedir?", "Java derlenmiş bir dil iken Python yorumlanmış bir dildir.", True),
        ("Yapay zeka nedir?", "Yapay zeka, insan zekasını taklit eden ve öğrenebilen bilgisayar sistemleridir.", True),
        ("Derin öğrenme nedir?", "Derin öğrenme, yapay sinir ağları kullanan bir makine öğrenmesi alt dalıdır.", True),
        ("Doğal dil işleme nedir?", "Doğal dil işleme, bilgisayarların insan dilini anlama ve işleme yeteneğidir.", True),
        ("Makine öğrenmesi nedir?", "Makine öğrenmesi, bilgisayarların veriden öğrenmesini sağlayan bir yapay zeka dalıdır.", True),
        ("TF-IDF nedir?", "TF-IDF, belgelerdeki kelimelerin önemini ölçen bir ağırlıklandırma yöntemidir.", True),
        ("Word2Vec nedir?", "Word2Vec, kelimeleri vektör uzayında temsil eden bir doğal dil işleme tekniğidir.", True),
        ("Veri madenciliği nedir?", "Veri madenciliği, büyük veri setlerinden anlamlı bilgiler çıkarma sürecidir.", True),
        ("Büyük veri nedir?", "Büyük veri, geleneksel yöntemlerle işlenemeyecek kadar büyük ve karmaşık veri setleridir.", True),
    ]

    print("Geribildirim ekleme testi başlıyor...")
    for i, (question, answer, is_relevant) in enumerate(test_cases, 1):
        print(f"\nTest {i}/10:")
        print(f"Soru: {question}")
        
        # Önce en iyi cevapları bul
        results = qa_matcher.find_best_answers(question)
        print(f"İlk benzerlik skoru: {results[0]['similarity']:.4f}")
        
        # Geribildirim ekle
        updated = qa_matcher.add_feedback(question, answer, is_relevant)
        print(f"Geribildirim eklendi, güncelleme gerekli mi: {updated}")

    print("\nTüm geribildirimler eklendi. Model güncelleniyor...")
    updated = qa_matcher.update_model()
    print(f"Model güncelleme sonucu: {updated}")

    # Model güncellemesinden sonra aynı soruları tekrar test et
    print("\nGüncellenen model ile test:")
    for question, answer, _ in test_cases:
        results = qa_matcher.find_best_answers(question)
        print(f"\nSoru: {question}")
        print(f"En iyi cevap: {results[0]['answer']}")
        print(f"Benzerlik skoru: {results[0]['similarity']:.4f}")

if __name__ == "__main__":
    test_feedback_system()
