import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('mushrooms.csv')

def df_proc(df):
    # not using 'stalk-root' as this has missing values
    columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
    df.pop('stalk-root')
    df = pd.get_dummies(df, columns=columns)
    y = df.pop('class').values
    X = df.values
    return X, y

def rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rfc = RandomForestClassifier()
    rfc.fit_transform(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    print('Accuracy for this RF classifier is: {}'.format(acc))


if __name__ == '__main__':
    X, y = df_proc(df)
    rf(X, y)
