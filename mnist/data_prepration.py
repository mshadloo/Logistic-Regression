from mlxtend.data import loadlocal_mnist
import numpy as np

def create_data(num1,num2):
        X_train_orig, y_train_orig = loadlocal_mnist(
                images_path='./mnist/data/train-images-idx3-ubyte',
                labels_path='./mnist/data/train-labels-idx1-ubyte')
        X_test_orig, y_test_orig = loadlocal_mnist(
                images_path='./mnist/data/t10k-images-idx3-ubyte',
                labels_path='./mnist/data/t10k-labels-idx1-ubyte')

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for i,a in enumerate(y_train_orig):
             if a== num1 or a == num2:
                     X_train.append(X_train_orig[i,:])
                     if a== num1:
                             y_train.append(1)
                     else:
                             y_train.append(0)
        for i,a in enumerate(y_test_orig):
             if a== num1 or a == num2:
                     X_test.append(X_test_orig[i,:])
                     if a== num1:
                             y_test.append(1)
                     else:
                             y_test.append(0)

        return np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test)
def create_data2(num1,num2):
        X_train_orig, y_train_orig = loadlocal_mnist(
                images_path='./mnist/data/train-images-idx3-ubyte',
                labels_path='./mnist/data/train-labels-idx1-ubyte')


        X_train = []
        y_train = []

        for i,a in enumerate(y_train_orig):
             if a== num1 or a == num2:
                     X_train.append(X_train_orig[i,:])
                     if a== num1:
                             y_train.append(1)
                     else:
                             y_train.append(0)


        return np.array(X_train), np.array(y_train)