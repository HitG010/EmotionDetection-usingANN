import spacy
import csv
import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
stemmer = PorterStemmer()
nltk.download('stopwords')

def remove_stop_words(words):
    stop_word = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_word]
    new_string = " ".join(filtered_words) 
    
    return new_string


def vectorize(sentances):
    # Create a Vectorizer Object
    vectorizer = CountVectorizer()
    
    vectorizer.fit(sentances)
    
    vocab = vectorizer.vocabulary_
    # Printing the identified Unique words along with their indices
    print("Vocabulary: ", vocab)
 
    # Encode the Document
    vector = vectorizer.transform(sentances)
    
    # Summarizing the Encoded Texts
    print("Encoded Document is:")
    print(vector.toarray())
 
    return vector.toarray(), vocab


nlp=spacy.load("en_core_web_sm")

def lemminization(text):
    text = text.lower()
    doc = nlp(text)
    words = []
    for sentance in doc.sents:
        for token in sentance:
            if token.is_punct:
                continue
            words.append(token.lemma_)  #Lemma is the root form of the word
    
    return words


def preProcess(document):
    sentances = []
    for doc in document:
        words = lemminization(doc)
        words = remove_stop_words(words)
        print(words)
        sentances.append(words)
    vector, vocab = vectorize(sentances)
    return vector, vocab

count = 5000
document = pd.read_csv("goemotions_1.csv", delimiter=",", nrows=count)
document = document[["text", 'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral']]
# print(document.head())

ans = ['emotions']

emotions = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral']
# for row in document[emotions].iterrows():
#     # print('h')
#     # print(row[1])
#     row = row[1]
#     for ele in range(28):
#         # print(row[ele])
#         if row[ele] == 1:
#             ans.append(ele)

# ans = ans[1:count]
# print(ans)
# print(len(ans))
# document = pd.concat([document, pd.DataFrame(ans)], axis=1)
# document = document.drop(columns=emotions)

print(document)

sentances, vocab = preProcess(document["text"])
# print(sentances)
# document = document.drop(columns=["text"])
# document = pd.concat([document, pd.DataFrame(sentances)], axis=1)

print(document)

# Implementing softmax function

# Defining model

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(1024, activation='relu', input_shape=(document.shape[1],)),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(27, activation='softmax')
# ])

# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
# )
sentances = sentances.tolist()
print(type(sentances[1:]))
# model.fit(sentances[1:], ans, epochs=100)

X_test = []
input_text = ["I failed in the exam", "I am happy", "I am sad", "I am angry", "I am excited", "I am nervous", "I am confused", "I am curious", "I am surprised", "I am disgusted", "I am annoyed", "I am grateful", "I am proud", "I am relieved", "I am remorseful", "I am caring", "I am loving", "I am optimistic", "I am disappointed", "I am approving", "I am admiring", "I am amused", "I am embarrassed", "I am fearful", "I am grieving", "I am joyful", "I am neutral", "I am realizing"]

for text in input_text:
    words = lemminization(text)
    words = remove_stop_words(words)
    indices = []
    for word in words.split():
        if word in vocab:
            indices.append(vocab[word])
    vector_input = []
    for i in range(len(vocab)):
        if i in indices:
            vector_input.append(1)
        else:
            vector_input.append(0)
    X_test.append(vector_input)

# indices.sort()
vector_input = []
for i in range(len(vocab)):
    if i in indices:
        vector_input.append(1)
    else:
        vector_input.append(0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.model_selection import train_test_split
for i in range(28):
    X_train, x_t, y_train, y_test = train_test_split(sentances[1:], document[emotions[i]].tolist()[1:], test_size=0.4, random_state=0)

    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(x_t)
    print(y_pred)

    from sklearn import metrics
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    # print(X_test)
    print(gnb.predict(X_test))



# pred = model.predict(X_test)

# # diff = []
# # for i in range(28):
# #     a = 0
# #     if (pred[0][i]>1):
# #         a = pred[0][i] - 1
# #     else:
# #         a = 1 - pred[0][i]
# #     diff.append(a)
    

# print(pred)
# for i in range(len(pred)):
#     print(input_text[i], " : ", emotions[np.argmax(pred[i])])


        

# # print(document)
# # document.to_csv("goemotions_2.csv", index=False)




