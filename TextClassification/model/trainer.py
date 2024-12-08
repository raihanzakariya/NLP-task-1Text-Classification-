"""Text classifcation traininng and evaluation implementation"""
# train model with pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import joblib
import os
from model.preprocessing import Preprocessing, stop_words


class Trainer:
    """Class to train and evaluate model"""

    def __init__(self, data_frame):
        """Initializes the trainer with a given Dataframe"""
        self.data_frame = data_frame
        self.pre_proc = Preprocessing(self.data_frame)

    def train(self):
        """Prepare the data, trains the model, and evaluate its performance"""
        X_train, X_test, y_train, y_test = self.pre_proc.process_data()
        print('Training Data :', X_train.shape)
        print('Testing Data : ', X_test.shape)

        pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=list(stop_words))),
                ('clf', LinearSVC()),
            ])

        pipeline.fit(X_train, y_train)
        print ("Training Metrics")
        self.evaluate(pipeline, X_train, y_train)
        print ("Evaluation Metrics")
        self.evaluate(pipeline, X_test, y_test)
        self.save_model(pipeline)
    
    def evaluate(self, pipeline, X_test, y_test):
        """Evaluate the model using accuracy """
        predictions = pipeline.predict(X_test)
        print('Accuracy = ', accuracy_score(y_test,predictions))
        print('F1 score is ',f1_score(y_test, predictions, average="micro"))

    def save_model(self, pipeline, path="outputs/model"):
        """Saves the trained model to a specified directory"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(pipeline, path+"/text_clf.joblib")
