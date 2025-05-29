import pandas as pd
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from similarity_analysis import SimilarityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ModelEvaluator:
    def __init__(self):
        self.results_dir = 'results'
        self.evaluation_dir = 'evaluation'
        os.makedirs(self.evaluation_dir, exist_ok=True)
        self.similarity_analyzer = SimilarityAnalyzer()
        
    def load_similarity_results(self):
        """Benzerlik sonuçlarını yükle"""
        results_path = os.path.join(self.results_dir, 'similarity_results.csv')
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        return None
    
    def calculate_automatic_score(self, input_text, result_text, similarity_score):
        """Otomatik puan hesapla"""
        # TF-IDF vektörlerini oluştur
        vectorizer = TfidfVectorizer()
        try:
            vectors = vectorizer.fit_transform([input_text, result_text])
            semantic_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            semantic_similarity = 0
        
        # Puan hesaplama kriterleri
        base_score = similarity_score * 5  # Benzerlik skorunu 5'lik ölçeğe dönüştür
        
        # Anlamsal benzerlik puanı
        semantic_score = semantic_similarity * 5
        
        # Final puan (benzerlik ve anlamsal puanın ortalaması)
        final_score = (base_score + semantic_score) / 2
        
        # Puanı 1-5 aralığına yuvarla
        return max(1, min(5, round(final_score)))
    
    def evaluate_model_results(self, results_df):
        """Model sonuçlarını otomatik değerlendir"""
        print("\nModel Değerlendirmesi Başlatılıyor...")
        
        evaluations = defaultdict(list)
        models = results_df['model'].unique()
        
        # Giriş metnini al (ilk sonucun metnini referans olarak kullan)
        input_text = results_df.iloc[0]['text']
        
        for model in models:
            print(f"\nModel: {model} değerlendiriliyor...")
            
            model_results = results_df[results_df['model'] == model].head(5)
            
            for idx, row in model_results.iterrows():
                # Otomatik puan hesapla
                score = self.calculate_automatic_score(input_text, row['text'], row['similarity'])
                evaluations[model].append(score)
                
                print(f"Sonuç {idx % 5 + 1}:")
                print(f"Metin: {row['text']}")
                print(f"Benzerlik: {row['similarity']:.4f}")
                print(f"Otomatik Puan: {score}")
        
        return evaluations
    
    def calculate_model_scores(self, evaluations):
        """Model puanlarını hesapla"""
        scores = {}
        for model, scores_list in evaluations.items():
            scores[model] = {
                'mean_score': np.mean(scores_list),
                'std_score': np.std(scores_list),
                'scores': scores_list
            }
        return scores
    
    def plot_model_scores(self, scores):
        """Model puanlarını görselleştir"""
        models = list(scores.keys())
        mean_scores = [scores[model]['mean_score'] for model in models]
        std_scores = [scores[model]['std_score'] for model in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, mean_scores, yerr=std_scores, capsize=5)
        
        plt.title('Model Performans Karşılaştırması')
        plt.xlabel('Model')
        plt.ylabel('Ortalama Puan')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 5)
        
        # Puanları barların üzerine yaz
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.evaluation_dir, 'model_scores.png'))
        plt.close()
    
    def save_evaluation_results(self, scores):
        """Değerlendirme sonuçlarını kaydet"""
        results = []
        for model, score_data in scores.items():
            results.append({
                'model': model,
                'mean_score': score_data['mean_score'],
                'std_score': score_data['std_score'],
                'scores': ','.join(map(str, score_data['scores']))
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.evaluation_dir, 'evaluation_results.csv'), index=False)
        
        # Sonuçları ekrana yazdır
        print("\nDeğerlendirme Sonuçları:")
        print("="*50)
        for model, score_data in scores.items():
            print(f"\nModel: {model}")
            print(f"Ortalama Puan: {score_data['mean_score']:.2f} ± {score_data['std_score']:.2f}")
            print(f"Puanlar: {score_data['scores']}")
    
    def calculate_jaccard_similarity(self, set1, set2):
        """İki küme arasındaki Jaccard benzerliğini hesapla"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    def create_jaccard_matrix(self, results_df):
        """18x18 Jaccard benzerlik matrisi oluştur"""
        # Her model için ilk 5 sonucu al
        model_results = {}
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model].head(5)
            model_results[model] = set(model_data['text'].tolist())
        
        # Model isimlerini al
        models = list(model_results.keys())
        n_models = len(models)
        
        # Jaccard matrisini oluştur
        jaccard_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    jaccard_matrix[i, j] = 1.0  # Köşegen değerleri 1
                else:
                    jaccard_matrix[i, j] = self.calculate_jaccard_similarity(
                        model_results[models[i]],
                        model_results[models[j]]
                    )
        
        return jaccard_matrix, models
    
    def plot_jaccard_matrix(self, matrix, model_names):
        """Jaccard benzerlik matrisini görselleştir"""
        plt.figure(figsize=(15, 12))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=model_names, yticklabels=model_names)
        plt.title('Model Sıralama Tutarlılığı (Jaccard Benzerliği)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.evaluation_dir, 'jaccard_similarity_matrix.png'))
        plt.close()
    
    def analyze_jaccard_matrix(self, matrix, model_names):
        """Jaccard matrisini analiz et ve sonuçları kaydet"""
        # Model gruplarını belirle
        model_groups = {
            'tfidf': [m for m in model_names if m.startswith('tfidf_')],
            'word2vec_cbow': [m for m in model_names if '_sg0_' in m],
            'word2vec_skipgram': [m for m in model_names if '_sg1_' in m]
        }
        
        # Grup içi ve gruplar arası ortalama benzerlikleri hesapla
        analysis_results = []
        
        for group1_name, group1_models in model_groups.items():
            for group2_name, group2_models in model_groups.items():
                group1_indices = [model_names.index(m) for m in group1_models]
                group2_indices = [model_names.index(m) for m in group2_models]
                
                # Gruplar arası benzerlikleri hesapla
                similarities = []
                for i in group1_indices:
                    for j in group2_indices:
                        if i != j:  # Kendisiyle karşılaştırmayı hariç tut
                            similarities.append(matrix[i, j])
                
                if similarities:
                    analysis_results.append({
                        'group1': group1_name,
                        'group2': group2_name,
                        'mean_similarity': np.mean(similarities),
                        'std_similarity': np.std(similarities)
                    })
        
        # Sonuçları DataFrame'e dönüştür ve kaydet
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(os.path.join(self.evaluation_dir, 'jaccard_analysis.csv'), index=False)
        
        # Sonuçları ekrana yazdır
        print("\nJaccard Benzerlik Analizi:")
        print("="*50)
        for _, row in analysis_df.iterrows():
            print(f"\n{row['group1']} vs {row['group2']}:")
            print(f"Ortalama Benzerlik: {row['mean_similarity']:.3f} ± {row['std_similarity']:.3f}")
        
        return analysis_df
    
    def run_evaluation(self):
        """Değerlendirme sürecini başlat"""
        results_df = self.load_similarity_results()
        if results_df is None:
            print("Benzerlik sonuçları bulunamadı!")
            return
        
        # Model değerlendirmesi
        evaluations = self.evaluate_model_results(results_df)
        scores = self.calculate_model_scores(evaluations)
        self.plot_model_scores(scores)
        self.save_evaluation_results(scores)
        
        # Jaccard analizi
        jaccard_matrix, model_names = self.create_jaccard_matrix(results_df)
        self.plot_jaccard_matrix(jaccard_matrix, model_names)
        self.analyze_jaccard_matrix(jaccard_matrix, model_names)
    
    def analyze_model_configurations(self, eval_df):
        """Model yapılandırmalarının etkisini analiz et"""
        # Model adlarından yapılandırma bilgilerini çıkar
        configs = []
        for model_name in eval_df['model']:
            if model_name.startswith('tfidf_'):
                configs.append({
                    'type': 'tfidf',
                    'preprocessing': model_name.split('_')[1],
                    'score': eval_df[eval_df['model'] == model_name]['mean_score'].iloc[0]
                })
            else:
                # Word2Vec model adı formatı: prefix_sg{0/1}_vs{size}_w{window}
                parts = model_name.split('_')
                configs.append({
                    'type': 'word2vec',
                    'preprocessing': parts[0],
                    'algorithm': 'CBOW' if parts[1] == 'sg0' else 'Skip-gram',
                    'vector_size': int(parts[2][2:]),
                    'window': int(parts[3][1:]),
                    'score': eval_df[eval_df['model'] == model_name]['mean_score'].iloc[0]
                })
        
        config_df = pd.DataFrame(configs)
        
        # Analiz sonuçlarını kaydet
        config_df.to_csv(os.path.join(self.evaluation_dir, 'model_configuration_analysis.csv'), index=False)
        
        # Özet istatistikler
        print("\nModel Yapılandırma Analizi:")
        print("\nTF-IDF vs Word2Vec:")
        print(config_df.groupby('type')['score'].mean())
        
        print("\nÖn İşleme Etkisi:")
        print(config_df.groupby('preprocessing')['score'].mean())
        
        if 'algorithm' in config_df.columns:
            print("\nAlgoritma Etkisi (Word2Vec):")
            print(config_df[config_df['type'] == 'word2vec'].groupby('algorithm')['score'].mean())
            
            print("\nVektör Boyutu Etkisi (Word2Vec):")
            print(config_df[config_df['type'] == 'word2vec'].groupby('vector_size')['score'].mean())
            
            print("\nPencere Boyutu Etkisi (Word2Vec):")
            print(config_df[config_df['type'] == 'word2vec'].groupby('window')['score'].mean())
        
        return config_df

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
    
    # Model yapılandırmalarını analiz et
    results_df = pd.read_csv(os.path.join(evaluator.evaluation_dir, 'evaluation_results.csv'))
    config_analysis = evaluator.analyze_model_configurations(results_df)
    
    print("\nDeğerlendirme tamamlandı! Sonuçlar 'evaluation' dizinine kaydedildi.")