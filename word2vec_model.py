import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import pickle
import time

class Word2VecModel:
    def __init__(self):
        self.models = {}
        
    def prepare_sentences(self, texts):
        """Metinleri kelime listelerine dönüştür"""
        return [text.split() for text in texts]
    
    def train_model(self, sentences, sg, vector_size, window, name):
        """Belirli parametrelerle Word2Vec modeli eğit"""
        model = Word2Vec(
            sentences=sentences,
            sg=sg,  # 0 for CBOW, 1 for Skip-gram
            vector_size=vector_size,
            window=window,
            min_count=2,  # En az 2 kez geçen kelimeleri al
            workers=4,
            epochs=20,  # Epoch sayısını artır
            negative=10,  # Negatif örnekleme sayısını artır
            sample=0.001,  # Yüksek frekanslı kelimeleri daha fazla örnekle
            alpha=0.025,  # Öğrenme oranı
            min_alpha=0.0001  # Minimum öğrenme oranı
        )
        self.models[name] = model
        return model
    
    def get_document_vector(self, model, text):
        """Metni vektöre dönüştür"""
        words = text.split()
        vectors = []
        for word in words:
            try:
                vectors.append(model.wv[word])
            except KeyError:
                continue
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(model.vector_size)
    
    def evaluate_qa_matching(self, model, questions, answers, top_n=3):
        """Soru-cevap eşleştirme performansını değerlendir"""
        # Soru ve cevapları vektörleştir
        question_vectors = np.array([self.get_document_vector(model, q) for q in questions])
        answer_vectors = np.array([self.get_document_vector(model, a) for a in answers])
        
        # Her soru için en benzer cevapları bul
        results = []
        for i, question in enumerate(questions):
            similarities = cosine_similarity([question_vectors[i]], answer_vectors)[0]
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            
            # Benzerlik skorunu normalize et
            max_similarity = similarities[top_indices[0]]
            if max_similarity > 0:
                similarities = similarities / max_similarity
            
            question_results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Sadece anlamlı benzerlikleri al
                    question_results.append({
                        'answer': answers[idx],
                        'similarity': float(similarities[idx])
                    })
            results.append({
                'question': question,
                'best_answers': question_results
            })
        
        return results
    
    def save_vectors(self, model, questions, answers, model_name, output_path):
        """Model vektörlerini kaydet"""
        vectors_path = os.path.join(output_path, f"{model_name}_vectors.pkl")
        
        # Soru ve cevap vektörlerini hesapla
        question_vectors = np.array([self.get_document_vector(model, q) for q in questions])
        answer_vectors = np.array([self.get_document_vector(model, a) for a in answers])
        
        # Vektörleri kaydet
        vectors_data = {
            'question_vectors': question_vectors,
            'answer_vectors': answer_vectors
        }
        
        with open(vectors_path, 'wb') as f:
            pickle.dump(vectors_data, f)
        
        return question_vectors, answer_vectors
    
    def load_vectors(self, model_name, output_path):
        """Kaydedilmiş vektörleri yükle"""
        vectors_path = os.path.join(output_path, f"{model_name}_vectors.pkl")
        
        if os.path.exists(vectors_path):
            with open(vectors_path, 'rb') as f:
                vectors_data = pickle.load(f)
                return vectors_data['question_vectors'], vectors_data['answer_vectors']
        
        return None, None
    
    def train_all_configurations(self, data_path, output_path):
        """Tüm konfigürasyonları eğit ve sonuçları kaydet"""
        # Dizini oluştur
        os.makedirs(output_path, exist_ok=True)
        
        # Sonuç dosyasını kontrol et
        summary_path = os.path.join(output_path, "word2vec_results_summary.csv")
        if os.path.exists(summary_path):
            print(f"Kaydedilmiş Word2Vec sonuçları bulundu: {summary_path}")
            print("Mevcut sonuçlar kullanılacak. Yeniden eğitim için dosyayı silin.")
            return pd.read_csv(summary_path)
        
        # Veri setini yükle
        df = pd.read_csv(data_path)
        
        # Lemma ve stem verilerini hazırla
        data_types = {
            'lemma': {
                'questions': df['lemma_question'].tolist(),
                'answers': df['lemma_answer'].tolist()
            },
            'stem': {
                'questions': df['stem_question'].tolist(),
                'answers': df['stem_answer'].tolist()
            }
        }
        
        # Tüm konfigürasyonlar
        configs = [
            {'sg': 0, 'vector_size': 300, 'window': 4},  # CBOW, 300d, window=4
            {'sg': 0, 'vector_size': 300, 'window': 2},  # CBOW, 300d, window=2
            {'sg': 0, 'vector_size': 100, 'window': 4},  # CBOW, 100d, window=4
            {'sg': 0, 'vector_size': 100, 'window': 2},  # CBOW, 100d, window=2
            {'sg': 1, 'vector_size': 300, 'window': 4},  # Skip-gram, 300d, window=4
            {'sg': 1, 'vector_size': 300, 'window': 2},  # Skip-gram, 300d, window=2
            {'sg': 1, 'vector_size': 100, 'window': 4},  # Skip-gram, 100d, window=4
            {'sg': 1, 'vector_size': 100, 'window': 2},  # Skip-gram, 100d, window=2
        ]
        
        results = []
        test_questions = df['question'].tolist()[:10]  # İlk 10 soru için test
        
        # Toplam işlem süresini ölç
        total_start_time = time.time()
        
        for data_type, data in data_types.items():
            sentences = self.prepare_sentences(data['questions'] + data['answers'])
            
            for config in configs:
                model_name = f"{data_type}_sg{config['sg']}_vs{config['vector_size']}_w{config['window']}"
                model_path = os.path.join(output_path, f"{model_name}.model")
                
                # Eğer model dosyası varsa, yükle
                if os.path.exists(model_path):
                    print(f"Kaydedilmiş model yükleniyor: {model_name}")
                    model = Word2Vec.load(model_path)
                    self.models[model_name] = model
                    
                    # Vektörleri yükle veya yeniden hesapla
                    question_vectors, answer_vectors = self.load_vectors(model_name, output_path)
                    if question_vectors is None or answer_vectors is None:
                        print(f"Vektörler bulunamadı, yeniden hesaplanıyor: {model_name}")
                        question_vectors, answer_vectors = self.save_vectors(
                            model, test_questions, data['answers'], model_name, output_path
                        )
                else:
                    # Modeli eğit
                    print(f"Yeni model eğitiliyor: {model_name}")
                    start_time = time.time()
                    
                    model = self.train_model(
                        sentences,
                        config['sg'],
                        config['vector_size'],
                        config['window'],
                        model_name
                    )
                    
                    training_time = time.time() - start_time
                    print(f"{model_name} eğitimi {training_time:.2f} saniye sürdü.")
                    
                    # Modeli kaydet
                    model.save(model_path)
                    
                    # Vektörleri hesapla ve kaydet
                    question_vectors, answer_vectors = self.save_vectors(
                        model, test_questions, data['answers'], model_name, output_path
                    )
                
                # Soru-cevap eşleştirme performansını değerlendir
                start_time = time.time()
                qa_results = self.evaluate_qa_matching(
                    model,
                    test_questions,
                    data['answers']
                )
                eval_time = time.time() - start_time
                print(f"Değerlendirme süresi: {eval_time:.2f} saniye")
                
                # Sonuçları kaydet
                for result in qa_results:
                    results.append({
                        'model': model_name,
                        'question': result['question'],
                        'best_answer': result['best_answers'][0]['answer'] if result['best_answers'] else '',
                        'similarity': result['best_answers'][0]['similarity'] if result['best_answers'] else 0.0
                    })
                
                # Örnek sonuçları göster
                print(f"\n{model_name} için örnek sonuçlar:")
                for result in qa_results[:3]:  # İlk 3 sonucu göster
                    print(f"\nSoru: {result['question']}")
                    if result['best_answers']:
                        print("En iyi cevap:", result['best_answers'][0]['answer'])
                        print("Benzerlik:", result['best_answers'][0]['similarity'])
                    else:
                        print("Uygun cevap bulunamadı")
        
        # Tüm sonuçları bir DataFrame'e dönüştür ve kaydet
        results_df = pd.DataFrame(results)
        
        # Eğer sonuçlar boş değilse kaydet
        if not results_df.empty:
            results_df.to_csv(summary_path, index=False)
            print("\nWord2Vec sonuçları kaydedildi:")
            print(results_df.sort_values('similarity', ascending=False).head(10))
        else:
            print("\nUyarı: Hiçbir sonuç bulunamadı!")
        
        # Toplam işlem süresini göster
        total_time = time.time() - total_start_time
        print(f"\nToplam işlem süresi: {total_time:.2f} saniye")
        
        return results_df

if __name__ == "__main__":
    # Dizinleri oluştur
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Word2Vec modelini oluştur ve eğit
    model = Word2VecModel()
    model.train_all_configurations('data/processed_data.csv', 'models')
    
    print("\nWord2Vec işlemi tamamlandı!")
    print("\nKaydedilmiş modelleri test etmek için tekrar çalıştırın.")
    print("Modeller kaydedilmişse, işlem çok daha hızlı olacaktır.")