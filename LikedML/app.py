import spacy
import csv
import pandas as pd
import nltk
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
    
    # Printing the identified Unique words along with their indices
    print("Vocabulary: ", vectorizer.vocabulary_)
 
    # Encode the Document
    vector = vectorizer.transform(sentances)
    
    # Summarizing the Encoded Texts
    print("Encoded Document is:")
    print(vector.toarray())
 
    return vector.toarray(), vectorizer.vocabulary_


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
    for i in range(0,len(document)):
        doc = document[i]
        print(doc)
        words = lemminization(doc)
        words = remove_stop_words(words)
        print(words)
        sentances.append(words)
    vector, vocab = vectorize(sentances)
    return vector, vocab


document = pd.read_csv("a1_RestaurantReviews_HistoricDump.tsv", sep='\t', header=None)
document.rename(columns={0: 'Review', 1:'Liked'}, inplace=True)
print(document.head())

vector, vocab = preProcess(document['Review'])
print(vector)
print(vector.shape)


document['vector'] = vector.tolist()
print(document['vector'])
# document.drop(columns=[0], inplace=True)
document.to_csv("a1_RestaurantReviews_HistoricDump_vectorized.tsv", sep='\t', header=None, index=False)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

X_train = vector[1:]
y_train = document['Liked'][1:]
print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=0)

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(y_pred)

from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

a = input("Enter a review: ")
a = lemminization(a)
a = remove_stop_words(a)
a = a.split()

for i in range(len(a)):
    if a[i] in vocab:
        a[i] = vocab[a[i]]
    else:
        a[i] = 0    
        


a = preProcess(a)
print(gnb.predict(a))