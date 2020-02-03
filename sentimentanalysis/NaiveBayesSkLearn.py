"""
Naive Bayes classifier for sentiment analysis
Using sklearn inbuilt MultinomialNB
"""

import re
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
stop_words = set(stopwords.words('english')) # nltk inbuilt stop words
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

import pandas as pd
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics


training_data=[]
trainingDataSet = []


def process(sentence):
    """
    removes stop words and perform lemmatization on text
    :param sentence:
    :return: processed sentence
    """
    tokens = word_tokenize(sentence)
    filtered_tokens = [word for word in tokens if word not in ["?", "'s", ",", "``", '()']]
    lower_word = [w.lower() for w in filtered_tokens]
    filtered_sentence = [w for w in lower_word if w not in stop_words]
    lemma_words = [wordnet_lemmatizer.lemmatize(w) for w in filtered_sentence]
    stem_word = [porter_stemmer.stem(w) for w in lemma_words]
    return " ".join(stem_word)

def extract_data(file,clss):
    """
    extract text from files
    :param file:
    :param clss:
    :return: extracted sentences
    """
    textData=[]
    with open(file) as textfile:
        text=textfile.read()
        textData.append(text.split('.'))
    for textlist in textData:
        for text in textlist:
            if (len(text)>3):
                out =re.sub('[+�]+', '', text)
                out=out.replace('[','').replace(']','').replace('\n','')
                trainingDataSet.append(out)
                training_data.append({'class':clss,'sentence':process(out)})
    return training_data


extract_data('positive.txt',1)
extract_data('negative.txt',0)

df= pd.DataFrame(training_data)
data = shuffle(df, random_state=22)
X_train, X_test, y_train, y_test = train_test_split(df['sentence'].values,
                 df['class'].values,
                 test_size=0.2)

vect = CountVectorizer(stop_words='english')
tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)

nb = MultinomialNB()
nb.fit(tf_train, y_train) # training

predictions = nb.predict(tf_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))


