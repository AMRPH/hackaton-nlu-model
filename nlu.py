import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

print("Hello World")

# Load the dataset
data = pd.read_csv('nlu_test.csv')
data = data.iloc[:10, -1]

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf = vectorizer.fit_transform(data)
print(tfidf.shape)

'''
# Apply Non-Negative Matrix Factorization (NMF) for topic modeling
nmf_model = NMF(n_components=20, random_state=0)
nmf_model.fit(tfidf)

# Get the top 10 keywords for each topic
num_keywords = 5
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf_model.components_):
    print("Topic %d:" % (topic_idx))
    print(", ".join([feature_names[i] for i in topic.argsort()[:-num_keywords - 1:-1]]))
    print("\n")
'''