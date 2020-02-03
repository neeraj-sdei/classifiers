"""
Naive Bayes classifier for sentiment analysis
Algorithm is based on term frequencies(i.e. frequency of a word in the document)
"""

import re
import pickle
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

training_data=[]


def extrac_data(file,clss):
    textData=[]
    with open(file) as textfile:
        text=textfile.read()
        textData.append(text.split('.'))
    for textlist in textData:
        for text in textlist:
            if (len(text)>3):
                out =re.sub('[+�]+', '', text)
                out=out.replace('[','').replace(']','').replace('\n','')
                training_data.append({'class':clss,'sentence':out})
    return training_data


extrac_data('positive.txt','pos')
extrac_data('negative.txt','neg')


class NaiveBayesClassifier:
    """
    class that will perform naive bayes algo based on frequency of words
    """
    def __init__(self):
            self.corpus_words=[]
            self.class_words=[]
            self.classes=[]

    def fit(self,training_data):
        self.classes=list(set([a['class'] for a in training_data]))
        for clss in self.classes:
            print(clss)
            print('Training in progress........')
            freq_dist={}
            for sent in training_data:
                if clss==sent['class']:
                    tokens=word_tokenize(sent['sentence'])
                    filtered_tokens=[word for word in tokens if word not in ["?", "'s",",","``",'()']]
                    lower_word=[w.lower() for w in filtered_tokens]
                    filtered_sentence = [w for w in lower_word if w not in stop_words]
                    lemma_words=[wordnet_lemmatizer.lemmatize(w) for w in filtered_sentence]
                    stem_word=[porter_stemmer.stem(w) for w in lemma_words]
                    for w in stem_word:
                        if w not in self.corpus_words:
                            self.corpus_words.append(w)
                    for w in stem_word:
                        if w in freq_dist:
                            freq_dist[w]+=1
                        else:
                            freq_dist[w]=1
            self.class_words.append([freq_dist,clss])
        print('Training completed ')

    def PredictResult(self,sent,clss):
        count = 0
        tokens = word_tokenize(sent)
        lower_word = [w.lower() for w in tokens]
        filtered_sentence = [w for w in lower_word if w not in stop_words]
        lemma_words=[wordnet_lemmatizer.lemmatize(w) for w in filtered_sentence]
        stem_word=[porter_stemmer.stem(w) for w in lemma_words]
        for w in stem_word:
            if w in self.corpus_words:
                for val in self.class_words:
                    if val[1]==clss:
                        temp=val[0]
                        if w in temp.keys():
                            w_freq=temp[w]
                            count+=w_freq/1000
            else:
                count+=0
        return count
    def prediction(self,sent):
        for clss in self.classes:
            weight=self.PredictResult(sent,clss)
            print(clss,weight)

Classifier=NaiveBayesClassifier()
Classifier.fit(training_data)
save_classifier = open("./naivebayes.pickle", "wb") # saving the model
pickle.dump(Classifier, save_classifier)
save_classifier.close()
classifier_f = open("./naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
testsent='karmen moves like rhythm itself'
print(testsent)
classifier.prediction(testsent) # predicting
