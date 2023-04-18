import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, path):
        df = pd.read_csv(path, index_col=0)
        self.labels = None
        self.y = None
        self.num_classes = 0
        if "category" in df.columns:
            unique_labels = df["category"].unique()
            self.labels = dict(zip(unique_labels, range(len(unique_labels))))
            self.num_classes = len(self.labels)
            df["category"] = df["category"].map(self.labels)
            self.y = df["category"].values
            self.x = df.drop("category", axis=1).values
        else:
            self.x = df.values

    def to_one_hot(self):
        if self.y is None:
            raise ValueError("No labels found in dataset")
        else:
            self.y = np.eye(self.num_classes)[self.y]

    def get_cat_to_label(self):
        return dict(zip(self.labels.values(), self.labels.keys()))


def write_to_csv(path, labels, mapping):
    df = pd.DataFrame({
        "Id": list(range(len(labels))),
        "Category": [mapping[i] for i in labels]
    })
    df.set_index("Id", inplace=True)
    df.to_csv(path)