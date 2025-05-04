import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import time

class TFIDFModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.95,
            stop_words=None,
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform(self, texts):
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix

    def evaluate_qa_matching(self, questions, answers, top_n=3):
        question_vectors = self.vectorizer.transform(questions)
        answer_vectors = self.vectorizer.transform(answers)
        results = []

        for i, question in enumerate(questions):
            similarities = cosine_similarity(question_vectors[i:i+1], answer_vectors)[0]
            top_indices = np.argsort(similarities)[-top_n:][::-1]

            max_similarity = similarities[top_indices[0]]
            if max_similarity > 0:
                similarities = similarities / max_similarity

            question_results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    question_results.append({
                        'answer': answers[idx],
                        'similarity': float(similarities[idx])
                    })

            results.append({
                'question': question,
                'best_answers': question_results
            })

        return results

    def save_model(self, prefix, output_path):
        os.makedirs(output_path, exist_ok=True)
        model_path = os.path.join(output_path, f'tfidf_{prefix}_model.pkl')
        model_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'feature_names': self.feature_names
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"TF-IDF modeli kaydedildi: {model_path}")

    def load_model(self, prefix, output_path):
        model_path = os.path.join(output_path, f'tfidf_{prefix}_model.pkl')
        if os.path.exists(model_path):
            print(f"Kaydedilmiş TF-IDF modeli yükleniyor: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.tfidf_matrix = model_data['tfidf_matrix']
                self.feature_names = model_data['feature_names']
            return True
        return False

    def process_data(self, data_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        df = pd.read_csv(data_path)
        results = {}

        for prefix in ['lemma', 'stem']:
            print(f"\n{prefix} verileri için TF-IDF işlemi...")

            result_csv = f'{output_path}/tfidf_{prefix}_results.csv'
            compressed_output_path = f'{output_path}/{prefix}_sparse_tfidf.csv.gz'

            if os.path.exists(result_csv) and os.path.exists(compressed_output_path) and self.load_model(prefix, output_path):
                print(f"Kaydedilmiş TF-IDF sonuçları bulundu, işlem atlanıyor.")
                continue

            print(f"TF-IDF modeli yeniden hesaplanıyor...")
            start_time = time.time()

            questions = df[f'{prefix}_question'].tolist()
            answers = df[f'{prefix}_answer'].tolist()
            all_texts = questions + answers

            tfidf_matrix = self.fit_transform(all_texts)
            processing_time = time.time() - start_time
            print(f"TF-IDF işlemi {processing_time:.2f} saniye sürdü.")
            self.save_model(prefix, output_path)

            # 🔻 Yeni optimizasyon: sparse matris, sıkıştırılmış CSV
            print(f"TF-IDF matrisini sıkıştırılmış CSV olarak kaydediliyor...")
            rows, cols = tfidf_matrix.nonzero()
            values = tfidf_matrix.data
            feature_names = self.feature_names

            sparse_records = [
                {'doc_id': row, 'term': feature_names[col], 'tfidf': float(val)}
                for row, col, val in zip(rows, cols, values)
            ]
            sparse_df = pd.DataFrame(sparse_records)
            sparse_df.to_csv(compressed_output_path, index=False, compression='gzip')
            print(f"Sparse TF-IDF verisi kaydedildi: {compressed_output_path}")

            num_documents = tfidf_matrix.shape[0]
            num_terms = tfidf_matrix.shape[1]
            sparsity = 100.0 * (1.0 - tfidf_matrix.nnz / float(num_documents * num_terms))
            print(f"Matris boyutları: {num_documents} belge x {num_terms} terim")
            print(f"Seyreklik (sparsity): {sparsity:.2f}%")

            qa_results = self.evaluate_qa_matching(questions[:10], answers)
            results_df = pd.DataFrame([
                {
                    'question': result['question'],
                    'best_answer': result['best_answers'][0]['answer'] if result['best_answers'] else '',
                    'similarity': result['best_answers'][0]['similarity'] if result['best_answers'] else 0.0
                }
                for result in qa_results
            ])
            results_df.to_csv(result_csv, index=False)
            results[prefix] = qa_results

            print(f"\n{prefix} için örnek soru-cevap eşleştirmeleri:")
            for result in qa_results[:3]:
                print(f"\nSoru: {result['question']}")
                if result['best_answers']:
                    print("En iyi cevap:", result['best_answers'][0]['answer'])
                    print("Benzerlik:", result['best_answers'][0]['similarity'])
                else:
                    print("Uygun cevap bulunamadı")

        return results

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    model = TFIDFModel()

    start_time = time.time()
    model.process_data('data/processed_data.csv', 'results')
    total_time = time.time() - start_time

    print(f"\nToplam işlem süresi: {total_time:.2f} saniye")
    print("\nTF-IDF işlemi tamamlandı!")
    print("\nKaydedilmiş modelleri test etmek için tekrar çalıştırın.")
