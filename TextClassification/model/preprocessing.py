""" Data processing and stopwords remove"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

"""importing function to create feature and label sets"""
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
#Remove Stopwords
stop_words = set(stopwords.words('english'))


class Preprocessing:
    """Class for Data processing, including cleaning anfd feature extraction """

    def __init__(self, data_frame=None):
        self.data = data_frame

    def remove_stopwords(self, text):
        """Removes stopwords from the given text"""
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)

    def clean_text(self, text):
        """function for text cleaning and remove non alphabetic character """
        text = text.lower()
        text = re.sub("[^a-zA-Z]"," ",text)
        text = ' '.join(text.split())
        return text
    
    def stemming(self, sentence):
        """function used for improve the text """


        stemmer = SnowballStemmer("english")
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    def split(self, test_size=0.2):
        """function for Splits the data into training and testing sets"""
        X = self.data['text']
        y = self.data['label']
        return train_test_split(X, y, test_size=0.20, random_state=123)

    def process_data(self):
        """function for Processes the data by cleaning the text, removing stopwords, and stemming"""


        self.data['text'] = self.data['text'].apply(lambda x: self.remove_stopwords(x))
        self.data['text'] = self.data['text'].apply(lambda x: self.clean_text(x))
        self.data['text'] = self.data['text'].apply(self.stemming)
        return self.split()
    
    def process_text(self, text):
        """Processes a single text input through the cleaning, stopword removal, and stemming steps"""
        text = self.remove_stopwords(text)
        text = self.clean_text(text)
        text = self.stemming(text)
        return text
