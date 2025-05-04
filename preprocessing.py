import pandas as pd
import numpy as np
from zemberek import TurkishMorphology, TurkishSentenceNormalizer
import re

class TextPreprocessor:
    def __init__(self):
        self.morphology = TurkishMorphology.create_with_defaults()
        self.normalizer = TurkishSentenceNormalizer(self.morphology)
        
        # Türkçe stopwords listesi
        self.stopwords = {
            've', 'veya', 'ya', 'de', 'da', 'ki', 'ile', 'mi', 'mı', 'mu', 'mü',
            'bu', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
            'için', 'gibi', 'kadar', 'değil', 'daha', 'her', 'bir', 'çok',
            'ama', 'fakat', 'lakin', 'ancak', 'oysa',
            'ne', 'nasıl', 'niye', 'neden', 'kim', 'hangi',
            'şey', 'şeyler', 'ise', 'isem', 'isen', 'imiş',
            'var', 'yok', 'evet', 'hayır', 'tamam',
            'olarak', 'üzere', 'göre',
            'tüm', 'bütün', 'tümü', 'hepsi',
            'kendi', 'kendisi', 'kendileri',
            'hiç', 'hiçbir', 'birşey', 'birkaç',
            'bazı', 'bazıları', 'birçok',
            'dolayı', 'ötürü', 'yüzünden',
            'önce', 'sonra', 'şimdi', 'daha',
            'artık', 'sadece', 'yalnız', 'yalnızca'
        }
        
    def clean_text(self, text):
        """Temel metin temizleme işlemleri"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = ' '.join([word for word in text.split() if word not in self.stopwords])
        return text.strip()
    
    def lemmatize(self, text):
        """Zemberek kullanarak lemmatization"""
        try:
            analysis = self.morphology.analyze(text)
            results = self.morphology.disambiguate(text, analysis)
            lemmas = [result.lemma for result in results]
            return ' '.join(lemmas)
        except:
            return text
    
    def stem(self, text):
        """Zemberek kullanarak stemming"""
        try:
            analysis = self.morphology.analyze(text)
            results = self.morphology.disambiguate(text, analysis)
            stems = [result.stem for result in results]
            return ' '.join(stems)
        except:
            return text
    
    def process_dataset(self, data_path, output_path):
        """Veri setini işle ve lemma/stem versiyonlarını kaydet"""
        # Veri setini yükle
        df = pd.read_json(data_path)
        
        # Soru-cevap çiftlerini çıkar
        questions = []
        answers = []
        
        for article in df['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    questions.append(qa['question'])
                    answers.append(qa['answers'][0]['text'])
        
        # DataFrame oluştur
        df = pd.DataFrame({
            'question': questions,
            'answer': answers
        })
        
        # Temizleme ve normalizasyon
        print("Metinleri temizleme ve normalizasyon...")
        df['clean_question'] = df['question'].apply(self.clean_text)
        df['clean_answer'] = df['answer'].apply(self.clean_text)
        
        # Lemmatization
        print("Lemmatization işlemi...")
        df['lemma_question'] = df['clean_question'].apply(self.lemmatize)
        df['lemma_answer'] = df['clean_answer'].apply(self.lemmatize)
        
        # Stemming
        print("Stemming işlemi...")
        df['stem_question'] = df['clean_question'].apply(self.stem)
        df['stem_answer'] = df['clean_answer'].apply(self.stem)
        
        # Tüm veriyi sakla
        df.to_csv(f'{output_path}/processed_data.csv', index=False)

        # Raporla uyumlu olarak sadece lemmatize/stem verileri ayrı ayrı kaydet
        df[['lemma_question', 'lemma_answer']].to_csv(f'{output_path}/cleaned_lemmatized.csv', index=False)
        df[['stem_question', 'stem_answer']].to_csv(f'{output_path}/cleaned_stemmed.csv', index=False)
        
        print("Tüm veriler başarıyla kaydedildi.")
        return df

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.process_dataset('squad-tr-train-v1.0.0-excluded.json', 'data')
