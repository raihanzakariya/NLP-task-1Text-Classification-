"""Text classifcation driver code."""

from dataset.data_loader import DataLoader
from dataset.data_processor import DataProcessor
from model.trainer import Trainer
from model.infer import TextClassifier

import argparse
parser = argparse.ArgumentParser("Text Classification")
parser.add_argument("-d", "--data", type=str, help="path to data(csv-file)")
parser.add_argument("-m", "--model", type=str, help="path to trained model")
parser.add_argument("-t", "--train", action="store_true", help="trainig mode")
parser.add_argument("-i", "--infer", type=str, help="infer text on trained model")
parser.add_argument("-p", "--plot", action="store_true", help="plot data hist")


def main(args):
    """driver function for text-classification."""
    data = None
    if args.data:
        data = DataLoader(args.data)
        data.load_csv()
        data.clean(keep_columns=["Description", "SubCategory"], set_columns=["text", "label"])

    if args.plot:
        data.plot_data("label")

    if args.train:
        trainer = Trainer(data.data_frame)
        trainer.train()
    elif args.infer:
        clf = TextClassifier(args.model)
        ans = clf.predict(args.infer)
        print (f"[I] Prediction: {ans}")

if __name__=="__main__":
    args = parser.parse_args()
    main(args)
