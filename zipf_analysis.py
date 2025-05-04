import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

def analyze_zipf_law(texts, title, output_path):
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

    print(f"{title} -> Eğim: {slope:.3f} | Yorum: {interpretation}")
    return slope, interpretation

def analyze_lemmatization_and_stemming():
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Verileri oku
    lemma_df = pd.read_csv('data/lemma_data.csv')
    stem_df = pd.read_csv('data/stem_data.csv')

    lemma_texts = lemma_df['lemma_question'].tolist() + lemma_df['lemma_answer'].tolist()
    stem_texts = stem_df['stem_question'].tolist() + stem_df['stem_answer'].tolist()

    # Analizler
    analyze_zipf_law(lemma_texts, 'Lemmatization Sonuçları', os.path.join(results_dir, 'zipf_lemma_data.png'))
    analyze_zipf_law(stem_texts, 'Stemming Sonuçları', os.path.join(results_dir, 'zipf_stem_data.png'))

if __name__ == "__main__":
    analyze_lemmatization_and_stemming()
