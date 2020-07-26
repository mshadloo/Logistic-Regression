import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def create_data():
    cols = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
    df = pd.read_csv('../Titanic/data/titanic.csv')
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean());
    df['Parents/Children Aboard'] = df['Parents/Children Aboard'].fillna(0);
    df['Siblings/Spouses Aboard'] = df['Siblings/Spouses Aboard'].fillna(0);
    df = df.drop('Name', axis=1)
    df['Sex'] = df['Sex'].astype('category')
    df['Sex'] = df['Sex'].cat.codes
    data = df.values
    # normalization
    data = (data - np.min(data, axis=0, keepdims=True)) / (
                np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))
    print(data)
    train_data, test_data = train_test_split(data, test_size=0.2)

    train_x = train_data[:, 1:]
    train_y = train_data[:, 0:1]
    test_x = test_data[:, 1:]
    test_y = test_data[:, 0:1]
    return train_x, train_y, test_x, test_y

# test_y=test_y.reshape(test_y.shape[0],1)

