import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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