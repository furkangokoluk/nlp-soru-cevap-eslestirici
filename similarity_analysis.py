import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class SimilarityAnalyzer:
    def __init__(self):
        self.results_dir = 'results'
        self.models_dir = 'models'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_tfidf_model(self, prefix):
        """TF-IDF modelini yükle"""
        model_path = os.path.join(self.results_dir, f'tfidf_{prefix}_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_word2vec_model(self, model_name):
        """Word2Vec modelini yükle"""
        model_path = os.path.join(self.models_dir, f'{model_name}.model')
        if os.path.exists(model_path):
            from gensim.models import Word2Vec
            return Word2Vec.load(model_path)
        return None
    
    def get_document_vector(self, model, text):
        """Metni vektöre dönüştür (Word2Vec için)"""
        if pd.isna(text):  # NaN kontrolü
            return np.zeros(model.vector_size)
            
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
    
    def calculate_similarities(self, input_text, data_path):
        """Tüm modeller için benzerlik hesapla"""
        df = pd.read_csv(data_path)
        results = defaultdict(list)
        
        # NaN değerleri temizle
        for prefix in ['lemma', 'stem']:
            df[f'{prefix}_question'] = df[f'{prefix}_question'].fillna('')
            df[f'{prefix}_answer'] = df[f'{prefix}_answer'].fillna('')
        
        # TF-IDF modelleri için
        for prefix in ['lemma', 'stem']:
            model_data = self.load_tfidf_model(prefix)
            if model_data:
                vectorizer = model_data['vectorizer']
                input_vector = vectorizer.transform([input_text])
                texts = df[f'{prefix}_question'].tolist() + df[f'{prefix}_answer'].tolist()
                text_vectors = vectorizer.transform(texts)
                
                similarities = cosine_similarity(input_vector, text_vectors)[0]
                top_indices = np.argsort(similarities)[-5:][::-1]
                
                for idx in top_indices:
                    results[f'tfidf_{prefix}'].append({
                        'text': texts[idx],
                        'similarity': float(similarities[idx])
                    })
        
        # Word2Vec modelleri için
        configs = [
            {'sg': 0, 'vector_size': 300, 'window': 4},  # CBOW
            {'sg': 0, 'vector_size': 300, 'window': 2},
            {'sg': 0, 'vector_size': 100, 'window': 4},
            {'sg': 0, 'vector_size': 100, 'window': 2},
            {'sg': 1, 'vector_size': 300, 'window': 4},  # Skip-gram
            {'sg': 1, 'vector_size': 300, 'window': 2},
            {'sg': 1, 'vector_size': 100, 'window': 4},
            {'sg': 1, 'vector_size': 100, 'window': 2}
        ]
        
        for prefix in ['lemma', 'stem']:
            for config in configs:
                model_name = f"{prefix}_sg{config['sg']}_vs{config['vector_size']}_w{config['window']}"
                model = self.load_word2vec_model(model_name)
                
                if model:
                    input_vector = self.get_document_vector(model, input_text)
                    texts = df[f'{prefix}_question'].tolist() + df[f'{prefix}_answer'].tolist()
                    text_vectors = np.array([self.get_document_vector(model, text) for text in texts])
                    
                    similarities = cosine_similarity([input_vector], text_vectors)[0]
                    top_indices = np.argsort(similarities)[-5:][::-1]
                    
                    for idx in top_indices:
                        results[model_name].append({
                            'text': texts[idx],
                            'similarity': float(similarities[idx])
                        })
        
        return results
    
    def calculate_jaccard_similarity(self, set1, set2):
        """İki küme arasındaki Jaccard benzerliğini hesapla"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    def create_jaccard_matrix(self, results):
        """Jaccard benzerlik matrisi oluştur"""
        model_names = list(results.keys())
        matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    texts1 = {r['text'] for r in results[model1]}
                    texts2 = {r['text'] for r in results[model2]}
                    matrix[i, j] = self.calculate_jaccard_similarity(texts1, texts2)
        
        return matrix, model_names
    
    def plot_jaccard_matrix(self, matrix, model_names):
        """Jaccard benzerlik matrisini görselleştir"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=model_names, yticklabels=model_names)
        plt.title('Model Benzerlik Matrisi (Jaccard)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_similarity_matrix.png'))
        plt.close()
    
    def analyze_similarities(self, input_text, data_path):
        """Tüm analizleri gerçekleştir"""
        print(f"\nGiriş metni: {input_text}")
        
        # Benzerlikleri hesapla
        results = self.calculate_similarities(input_text, data_path)
        
        # Sonuçları kaydet
        for model_name, model_results in results.items():
            print(f"\n{model_name} için en benzer 5 metin:")
            for i, result in enumerate(model_results, 1):
                print(f"{i}. Benzerlik: {result['similarity']:.4f}")
                print(f"   Metin: {result['text']}")
        
        # Jaccard matrisi oluştur ve görselleştir
        matrix, model_names = self.create_jaccard_matrix(results)
        self.plot_jaccard_matrix(matrix, model_names)
        
        # Sonuçları DataFrame'e dönüştür ve kaydet
        all_results = []
        for model_name, model_results in results.items():
            for result in model_results:
                all_results.append({
                    'model': model_name,
                    'text': result['text'],
                    'similarity': result['similarity']
                })
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(self.results_dir, 'similarity_results.csv'), index=False)
        
        return results_df

if __name__ == "__main__":
    analyzer = SimilarityAnalyzer()
    
    # Örnek bir giriş metni
    input_text = "Python programlama dili nedir?"
    
    # Analizi gerçekleştir
    results = analyzer.analyze_similarities(input_text, 'data/processed_data.csv')
    
    print("\nAnaliz tamamlandı! Sonuçlar 'results' dizinine kaydedildi.") 