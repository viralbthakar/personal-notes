import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


data_dict = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "text": [
        "The sun was setting over the ocean, casting a golden glow over the water.",
        "The smell of freshly baked bread filled the air as I walked into the bakery.",
        "The sound of laughter and chatter filled the room as friends gathered for a dinner party.",
        "The leaves rustled in the wind as I walked through the forest, admiring the autumn colors.",
        "The city skyline looked breathtaking from the top of the skyscraper.",
        "The crowd cheered as the winning goal was scored in the championship game.",
        "The aroma of coffee wafted through the cafe as people chatted over their drinks.",
        "The waves crashed against the shore, creating a soothing sound.",
        "The stars twinkled in the night sky, creating a beautiful sight to behold.",
        "The sound of rain tapping against the window was a comforting lullaby."
    ]
}

data_copy_dict = data_dict.copy()


# Define a function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if not token in stop_words]
    text = ' '.join(tokens)
    return text


# Preprocess the text in the data_dict
data_dict['text'] = [preprocess_text(text) for text in data_dict['text']]
print(data_dict)

# Train a Word2Vec model on the text data
sentences = [sentence.split() for sentence in data_dict['text']]
model = Word2Vec(sentences, min_count=1, vector_size=100)

# Compute the sentence embeddings using the Word2Vec model
embeddings = []
for sentence in sentences:
    sentence_embeddings = [model.wv[word]
                           for word in sentence if word in model.wv.key_to_index]
    if len(sentence_embeddings) > 0:
        embeddings.append(sum(sentence_embeddings) / len(sentence_embeddings))
    else:
        embeddings.append([0]*100)

# Calculate pairwise cosine similarity between the sentence embeddings
cosine_sim = cosine_similarity(embeddings)

# Print the similarity matrix
print(cosine_sim)


# Apply k-means clustering to the sentence embeddings
kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)

# Print the cluster assignments for each sentence
for i, cluster_label in enumerate(kmeans.labels_):
    print(
        f"Sentence {i+1} {data_copy_dict['text'][i]} belongs to cluster {cluster_label+1}")


for i in range(kmeans.n_clusters):
    cluster_sentences = []
    for j in range(len(sentences)):
        if kmeans.labels_[j] == i:
            cluster_sentences.append(data_copy_dict['text'][j])
    print("Cluster", i, ":", cluster_sentences)
