import pandas as pd

class Dataset:
    def __init__(self,path):
        df = pd.read_csv(path,index_col=0)
        self.labels = None
        self.y = None
        if 'category' in df.columns:
            # Create a dictionary of labels and their corresponding integer mapping
            unique_labels = df['category'].unique()
            self.labels = dict(zip(unique_labels,range(len(unique_labels))))

            self.y = df['category'].values
            self.x = df.drop('category',axis=1).values
        else:
            self.x = df.values
