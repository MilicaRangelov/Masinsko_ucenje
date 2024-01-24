import csv
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')


def generate_embeddings(strings):
    tokenized_strings = [word_tokenize(string.lower()) for string in strings]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for tokens in tokenized_strings:
        filtered_tokens.append([word for word in tokens if word not in stop_words])
    model = Word2Vec(filtered_tokens,  vector_size=20,
                     window=5, min_count=1, workers=4)
    embeddings = []
    for tokens in filtered_tokens:
        embedding = []
        for word in tokens:
            
            embedding.append(model.wv[word])

        if len(embedding) == 0:
            embedding = np.zeros(20)
        else:
            embedding = np.array(embedding).sum(axis=0)
            embedding = embedding / np.linalg.norm(embedding)
            #embedding=embedding[-1]
        
        embeddings.append(embedding)

    return embeddings





df = pd.read_csv('unique-categories.sorted-by-count.csv')
strings = df['keyword'].tolist()
embeddings = generate_embeddings(strings)


embedding_df = pd.DataFrame(embeddings,
                            columns=['embedding'+str(i) for i in range(20)])
embedding_df['keyword'] = strings


embedding_df.to_csv(
    'unique-categories.sorted-by-count.embeddings.csv', index=False)
