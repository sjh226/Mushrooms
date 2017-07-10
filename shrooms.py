import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('mushrooms.csv')

def df_proc(df):
    # not using 'stalk-root' as this has missing values
    # trying to only use easily identifyable features
    # 'gill-attachment', 'gill-spacing', 'gill-size',
    # 'stalk-surface-above-ring','stalk-surface-below-ring'
    # 'spore-print-color', 'population'
    y = df['class'].values
    columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises',\
       'odor', 'gill-color', 'stalk-shape', 'stalk-color-above-ring',\
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',\
       'ring-type', 'habitat']
    df = df[columns]
    df = pd.get_dummies(df, columns=columns)
    X = df.values
    return X, y, df

def rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rfc = RandomForestClassifier()
    rfc.fit_transform(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    print('Accuracy for this RF classifier is: {}'.format(acc))
    return rfc

def feat_imp(df, rfc):
    plt.close()
    plt.figure(figsize=(8, 8))
    zipped = list(zip(df.columns, rfc.feature_importances_))
    imp_dic = {item[1]:item[0] for item in zipped}
    left = list(range(0, len(rfc.feature_importances_), 1))
    plt.bar(left, rfc.feature_importances_, width=1, align='center')
    plt.xticks(left, df.columns, rotation='vertical', fontsize=7)
    plt.tight_layout()
    plt.savefig('feat_imp.png')


if __name__ == '__main__':
    X, y, dum_df = df_proc(df)
    rfc = rf(X, y)
    feat_imp(dum_df, rfc)
