import pandas as pd
import numpy as np

class Dataset:
    def __init__(self,path):
        df = pd.read_csv(path,index_col=0)
        self.labels = None
        self.y = None
        self.num_classes = 0
        if 'category' in df.columns:
            # Create a dictionary of labels and their corresponding integer mapping
            unique_labels = df['category'].unique()
            self.labels = dict(zip(unique_labels,range(len(unique_labels))))
            self.num_classes = len(self.labels)
            # Replace the labels with their corresponding integer mapping
            df['category'] = df['category'].map(self.labels)
            self.y = df['category'].values
            self.x = df.drop('category',axis=1).values
        else:
            self.x = df.values

    def to_one_hot(self):
        if self.y is None:
            raise Exception('No labels found in dataset')
        new_y = np.zeros((self.y.shape[0],self.num_classes))
        new_y[:,self.y] = 1
        self.y = new_y
