import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import os
import json
import pickle
import hashlib

class QAMatcher:
    def __init__(self, model_type='tfidf', vector_size=300):
        """
        Soru-cevap eşleştirici sınıfı
        
        Args:
            model_type (str): 'tfidf' veya 'word2vec'
            vector_size (int): Word2Vec için vektör boyutu
        """
        self.model_type = model_type
        self.vector_size = vector_size
        self.questions = []
        self.answers = []
        self.vectorizer = None
        self.word2vec_model = None
        self.question_vectors = None
        self.answer_vectors = None
        self.feedback_data = []
        self.feedback_hash = None  # Geri bildirim verilerinin hash değeri
    
    def load_data(self, data_path):
        """Veri setini yükle"""
        df = pd.read_csv(data_path)
        self.questions = df['question'].tolist()
        self.answers = df['answer'].tolist()
    
    def train(self):
        """Modeli eğit veya yükle"""
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model_type == 'tfidf':
            tfidf_model_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
            tfidf_vectors_path = os.path.join(model_dir, 'tfidf_vectors.pkl')
            
            # Eğer model dosyası varsa, yükle
            if os.path.exists(tfidf_model_path) and os.path.exists(tfidf_vectors_path):
                print("Kaydedilmiş TF-IDF modeli yükleniyor...")
                with open(tfidf_model_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(tfidf_vectors_path, 'rb') as f:
                    vectors_data = pickle.load(f)
                    self.question_vectors = vectors_data['question_vectors']
                    self.answer_vectors = vectors_data['answer_vectors']
            else:
                print("TF-IDF modeli eğitiliyor...")
                self.vectorizer = TfidfVectorizer()
                self.question_vectors = self.vectorizer.fit_transform(self.questions)
                self.answer_vectors = self.vectorizer.transform(self.answers)
                
                # Modeli ve vektörleri kaydet
                with open(tfidf_model_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                with open(tfidf_vectors_path, 'wb') as f:
                    vectors_data = {
                        'question_vectors': self.question_vectors,
                        'answer_vectors': self.answer_vectors
                    }
                    pickle.dump(vectors_data, f)
                print("TF-IDF modeli kaydedildi:", tfidf_model_path)
        else:  # word2vec
            word2vec_model_path = os.path.join(model_dir, f'word2vec_qa_matcher_vs{self.vector_size}.model')
            word2vec_vectors_path = os.path.join(model_dir, f'word2vec_vectors_vs{self.vector_size}.pkl')
            
            # Eğer model dosyası varsa, yükle
            if os.path.exists(word2vec_model_path) and os.path.exists(word2vec_vectors_path):
                print("Kaydedilmiş Word2Vec modeli yükleniyor...")
                self.word2vec_model = Word2Vec.load(word2vec_model_path)
                with open(word2vec_vectors_path, 'rb') as f:
                    vectors_data = pickle.load(f)
                    self.question_vectors = vectors_data['question_vectors']
                    self.answer_vectors = vectors_data['answer_vectors']
            else:
                print("Word2Vec modeli eğitiliyor...")
                # Metinleri kelime listelerine dönüştür
                sentences = [text.split() for text in self.questions + self.answers]
                
                # Word2Vec modelini eğit
                self.word2vec_model = Word2Vec(
                    sentences=sentences,
                    vector_size=self.vector_size,
                    window=5,
                    min_count=2,
                    workers=4
                )
                
                # Soru ve cevapları vektörleştir
                self.question_vectors = np.array([
                    self._get_document_vector(text) for text in self.questions
                ])
                self.answer_vectors = np.array([
                    self._get_document_vector(text) for text in self.answers
                ])
                
                # Modeli ve vektörleri kaydet
                self.word2vec_model.save(word2vec_model_path)
                with open(word2vec_vectors_path, 'wb') as f:
                    vectors_data = {
                        'question_vectors': self.question_vectors,
                        'answer_vectors': self.answer_vectors
                    }
                    pickle.dump(vectors_data, f)
                print("Word2Vec modeli kaydedildi:", word2vec_model_path)
    
    def _get_document_vector(self, text):
        """Metni vektöre dönüştür (Word2Vec için)"""
        words = text.split()
        vectors = []
        for word in words:
            try:
                vectors.append(self.word2vec_model.wv[word])
            except KeyError:
                continue
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)
    
    def find_best_answers(self, question, top_n=3):
        """Yeni bir soru için en iyi cevapları bul"""
        if self.model_type == 'tfidf':
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.answer_vectors)[0]
        else:
            question_vector = self._get_document_vector(question)
            similarities = cosine_similarity([question_vector], self.answer_vectors)[0]
        
        # En yüksek benzerlik skorlarına sahip cevapları bul
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'answer': self.answers[idx],
                'similarity': float(similarities[idx])
            })
        return results
    
    def add_feedback(self, question, selected_answer, is_correct):
        """Kullanıcı geribildirimini kaydet"""
        feedback = {
            'question': question,
            'selected_answer': selected_answer,
            'is_correct': is_correct,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.feedback_data.append(feedback)
        
        # Geribildirimleri dosyaya kaydet
        feedback_path = 'feedback.json'
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            
        # Yeni hash değerini hesapla
        new_hash = self._calculate_feedback_hash()
        
        # Hash değeri değiştiyse güncelleme gerektiğini belirt
        if self.feedback_hash != new_hash:
            self.feedback_hash = new_hash
            print("Yeni geribildirim eklendi, model güncellenebilir.")
            return True
        return False
        
    def _calculate_feedback_hash(self):
        """Geri bildirim verilerinin hash değerini hesapla"""
        if not self.feedback_data:
            return None
        
        # Geri bildirim verilerini string olarak birleştir ve hash'le
        feedback_str = json.dumps(self.feedback_data, sort_keys=True)
        return hashlib.md5(feedback_str.encode()).hexdigest()
    
    def update_model(self):
        """Geribildirimlere göre modeli güncelle"""
        if len(self.feedback_data) < 10:  # Minimum geribildirim sayısı
            print("Güncelleme için yeterli geribildirim yok (minimum 10 gerekli).")
            return False
        
        # Doğru eşleştirmeleri yeni veri olarak ekle
        correct_matches = [
            (fb['question'], fb['selected_answer'])
            for fb in self.feedback_data
            if fb['is_correct']
        ]
        
        if not correct_matches:
            print("Güncelleme için doğru eşleştirme bulunamadı.")
            return False
            
        # Hash değerini kontrol et - değişiklik yoksa güncelleme yapma
        current_hash = self._calculate_feedback_hash()
        if current_hash == self.feedback_hash and self.feedback_hash is not None:
            print("Geribildirim verilerinde değişiklik yok, güncelleme atlanıyor.")
            return False
        
        # Yeni hash değerini kaydet
        self.feedback_hash = current_hash
        
        # Yeni verileri ekle
        new_questions, new_answers = zip(*correct_matches)
        self.questions.extend(new_questions)
        self.answers.extend(new_answers)
        
        # Model dosyalarını sil (yeniden oluşturulacak)
        model_dir = 'models'
        if self.model_type == 'tfidf':
            for file_name in ['tfidf_vectorizer.pkl', 'tfidf_vectors.pkl']:
                file_path = os.path.join(model_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:  # word2vec
            for file_name in [f'word2vec_qa_matcher_vs{self.vector_size}.model', 
                         f'word2vec_vectors_vs{self.vector_size}.pkl']:
                file_path = os.path.join(model_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Modeli yeniden eğit
        self.train()
        
        print(f"Model {len(correct_matches)} yeni örnekle güncellendi.")
        return True

if __name__ == "__main__":
    # Test
    matcher = QAMatcher(model_type='tfidf')
    matcher.load_data('data/processed_data.csv')
    matcher.train()
    
    # Test sorusu
    test_question = "Python nedir?"
    results = matcher.find_best_answers(test_question)
    
    print(f"\nSoru: {test_question}")
    print("\nEn iyi 3 cevap:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Cevap: {result['answer']}")
        print(f"   Benzerlik: {result['similarity']:.4f}")
    
    # Geribildirim ekleme testi
    print("\nGeribildirim ekleme testi:")
    if results:
        updated = matcher.add_feedback(test_question, results[0]['answer'], True)
        print(f"Geribildirim eklendi, güncelleme gerekli mi: {updated}")
        
        # Aynı geribildirim tekrar eklenirse güncelleme gerekmez
        updated = matcher.add_feedback(test_question, results[0]['answer'], True)
        print(f"Aynı geribildirim tekrar eklendi, güncelleme gerekli mi: {updated}")