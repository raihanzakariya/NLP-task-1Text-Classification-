import joblib
from model.preprocessing import Preprocessing


class TextClassifier:
    """Class to text Classification  for processing"""

    def __init__(self, model_path="outputs/model/text_clf.joblib"):
        """Initializes the TextClassifier with a pre-trained model"""
        self.text_clf = joblib.load(model_path)
        self.proc = Preprocessing()

    def predict(self, input_text):
        """"""
        text = self.proc.process_text(input_text)
        return self.text_clf.predict([text])[0]
