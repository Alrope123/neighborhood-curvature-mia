from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
import os
import json
import numpy as np
import random

def iterate_files(root_dir):
    file_paths = []
    file_names = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file_path[len(root_dir):])
    return zip(file_paths, file_names)


if __name__ == '__main__':

    doc_dir = "/data/pile/train/00.jsonl"
    target_docs = []
    with open(doc_dir, 'r') as f:
        for line in f:
            dp = json.loads(line)
            if dp['meta']['pile_set_name'] == "ArXiv":
                target_docs.append(dp['text'])
    
    random_indices = [random.randint(0, len(target_docs)) for _ in range(100)]
    target_docs = [target_docs[i] for i in random_indices]

    k = 5
    top_k_texts = [None] * k
    top_k_scores = [-float('inf')] * k

    results = {doc: {'top_k_texts': [None] * k, 'top_k_scores': [-float('inf')] * k} for doc in target_docs}

    tfidf_vectorizer = TfidfVectorizer().fit(target_docs)  # Fit on the target docs first
    target_vectors = tfidf_vectorizer.transform(target_docs)

    data_dir = "/gscratch/h2lab/alrope/data/redpajama/arxiv/"
    for i, (data_path, filename) in tqdm(enumerate(iterate_files(data_dir))):
        # DEBUG
        # if i > 3:
        #     break

        doc_pool = []
        with open(data_path, 'r') as f:
            for line in f:
                doc_pool.append(json.loads(line)['text'])  

        # Transform the batch using the same vectorizer (this ensures consistent feature space)
        batch_vectors = tfidf_vectorizer.transform(doc_pool)

        for doc, target_vector in zip(target_docs, target_vectors):
            cosine_similarities = linear_kernel(target_vector, batch_vectors).flatten()

            for i, score in enumerate(cosine_similarities):
                if score > min(results[doc]['top_k_scores']):
                    # Get the index to replace based on the smallest score among top_k_scores
                    replace_index = results[doc]['top_k_scores'].index(min(results[doc]['top_k_scores']))
                    
                    # Replace the score and text in the top_k lists
                    results[doc]['top_k_scores'][replace_index] = score
                    results[doc]['top_k_texts'][replace_index] = doc_pool[i]

    # Sort each result list
    for doc in results:
        sorted_results = sorted(zip(results[doc]['top_k_texts'], results[doc]['top_k_scores']), key=lambda x: x[1], reverse=True)
        results[doc] = sorted_results

    out_dir = "/gscratch/h2lab/alrope/neighborhood-curvature-mia/debug/out"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'same_arxiv_document.json'), 'w') as f:
        json.dump(results, f)    
