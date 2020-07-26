import numpy as np
from functions import sigmoid
class LogisticRegression:
    def __init__(self):
        self.parameters = {}

    def initialize_with_zeros(self,dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """

        w = np.zeros((dim, 1))
        b = 0


        assert (w.shape == (dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))

        self.parameters["w"]=w
        self.parameters["b"]=b
    def gradient(self, w, b, X, Y):

        m = X.shape[1]
        Z = np.dot(w.T, X) + b
        A = sigmoid(Z)
        cost = np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
        m = X.shape[1]
        dZ = A - Y
        dw = np.dot(X, dZ.T) / m
        db = np.sum(dZ) / m


        cost = np.squeeze(cost)
        return cost, dw, db

    def optimize(self, w, b, X, Y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            cost, dw, db = self.gradient(w, b, X, Y)

            w = w - learning_rate * dw
            b = b - learning_rate * db
            print(" iteration " + str(i)+": " + str(cost))
            costs.append(cost)
        grads = {"dw": dw,
                 "db": db}
        params = {"w": w,
                 "b": b}
        return costs, params, grads
    def fit(self, X, Y,  num_iterations, learning_rate):
        dim = X.shape[0]
        self.initialize_with_zeros(dim)
        w, b = self.parameters["w"], self.parameters["b"]
        costs, params, grads = self.optimize( w, b, X, Y, num_iterations, learning_rate)
        self.parameters["w"], self.parameters["b"] = params["w"], params["b"]
        return costs
    def predict(self, X):
        w, b = self.parameters["w"], self.parameters["b"]
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        Z = np.dot(w.T, X) + b
        A = sigmoid(Z)
        for i in range(A.shape[1]):
            Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
        return Y_prediction,A
    def evaluate(self, X, Y):
        Y_prediction,A = self.predict(X)


        accuracy = 100 * (1 - np.mean(np.abs(Y_prediction - Y)))
        cost = np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A)))/Y.shape[1]
        return accuracy, cost, Y_prediction


if __name__ == "__main__":
    model = LogisticRegression()
    w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    cost, dw, db = model.gradient(w,b ,X,Y)
    print("dw = " + str(dw))
    print("db = " + str(db))
    print("cost = " + str(cost))
    costs, params, grads = model.optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009)

    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))

    model.parameters["w"]= np.array([[0.1124579], [0.23106775]])
    model.parameters["b"] = -0.3
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    print("predictions = " + str(model.predict( X)))
