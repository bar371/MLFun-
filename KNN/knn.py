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

    def predict(self, X_test, k):
        rets = []
        for sample in X_test:
            args = np.argsort([abs(t-sample) for t in self.X_train])
            y_sorted = self.y_train[args][0:k]
            choice = 0 if np.sum(y_sorted) / len(y_sorted) < 0.5 else 1
            print(choice)
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
    X_score = X.sum(1)
    print(X_score.shape)
    y = df['diabetes'].values
    X_train, X_test ,y_train, y_test = train_test_split(X_score, y, random_state=1, test_size=0.2, stratify=y)
    return X_train, X_test ,y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = pre_process('diabetes_data.csv')
    knn = KNNClassifer()
    knn.fit(X_train, y_train)
    preds = knn.predict(X_train, k=1)
    knn.getAccuracy(preds, y_train)




