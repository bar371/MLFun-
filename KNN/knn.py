import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class KNNClassifer() :

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self , sample1 , sample2 ):
        return np.sqrt(np.sum(np.power(sample1 - sample2, 2)))


    def predict(self, X_test, k):
        rets = []
        for sample in X_test.values:
            args = np.argsort([self.euclidean_distance(sample ,t ) for t in self.X_train.values])
            y_sorted = self.y_train[args][0:k]
            choice = 0 if np.sum(y_sorted) / len(y_sorted) < 0.5 else 1
            rets.append(choice)
        return rets

    def getAccuracy(self, pred, y_test):
        assert (len(pred) == len(y_test))
        print(np.sum([1 for p,y in zip(pred, y_test) if p==y]) / len(pred))
        

def pre_process(data_path):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    X = df.drop(columns='diabetes')
    X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    y = df['diabetes'].values
    X_train, X_test ,y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, stratify=y)
    return X_train, X_test ,y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = pre_process('../DataSets/diabetes_data.csv')
    knn = KNNClassifer()
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test, k=5)
    knn.getAccuracy(preds, y_test)




