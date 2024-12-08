# Text Classification Model
Given a set of data(text) and label(category), This project creates a text classifier model. It follows below steps in order to achieve this:

  - Clean and process data.
  - Select a model to train.
  - Analyse evaluation metric after training.


## Experiments
Performed multiple experiments with different model to chose best performing model, refer to `notebook/experiment_text_classification.ipynb`.
Found `LinearSVC()` to be the best peroming model.

## Results
- Evaluation:
  - Accuracy: 88.6%
  - F1Score: 88.6%

- for detailed report refer to `report/TextClassication_report.pdf`




## Setup

- Virtual environment
```
  python -m venv </path/to/venv>
  source </path/to/venv/bin/activate>
```

- Install depencies
```
  pip install -r requirements.txt
```

## Usage

- Load dataset
```
  python text_classification.py -d </path/to/input/data/csv/file>
```

- Visualize data distribution
```
  python text_classification.py -d ../Data/dataset.csv -p
```

- Train and evaluate classifcation model
```
  python text_classification.py -d ../Data/dataset.csv -t
```

- Infer trained model ("outputs/model/text_clf.joblib")
```
  python text_classification.py -m outputs/model/text_clf.pkl -i <"product description">
```

## Demo
Infer model on a given product description

```
python text_classification.py -m outputs/model/text_clf.pkl -i "one of Prada's most functional designs, this belt bag is made from weather-resistant shell fabric with zip compartments for storing your daily belongings. It's designed for navigating your day hands-free- try styling yours diagonally across the body."
```

Output
```
[nltk_data] Downloading package stopwords to
[nltk_data]     /home/dsj3kor/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[I] Prediction: Bags
```
