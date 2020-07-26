from mnist.data_prepration import create_data
from model import  LogisticRegression
import numpy as np

import matplotlib.pyplot as plt

X_train, Y_train, X_test, Y_test= create_data(1,7)
X_train = X_train.T/255
X_test = X_test.T/255
Y_test = Y_test.reshape(1, Y_test.shape[0])
Y_train = Y_train.reshape(1, Y_train.shape[0])


model = LogisticRegression()
costs = model.fit(X_train, Y_train, 10000, 0.4)
# Plot learning curve (with costs)
costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cross entropy loss')
plt.xlabel('iterations ')
plt.title("Learning rate =" + str(0.4))
plt.show()


accuracy_train,cost_train,prediction = model.evaluate(X_train, Y_train)
accuracy_test, cost_test, prediction = model.evaluate(X_test, Y_test)
print("accuracy on train set: " + str(accuracy_train))
print("cross entropy loss on train set: " + str(cost_train))
print("accuracy on test set: " + str(accuracy_test))
print("cross entropy loss on test set: " + str(cost_test))
#
# accuracy on train set: 99.75397862689321
# cross entropy loss on train set: 0.007362215273304598
# accuracy on test set: 99.44521497919555
# cross entropy loss on test set: 0.014537294410167503

# Y_prediction = accuracy_test[2]
# incorrects= []
# for i in range(Y_prediction.shape[1]):
#     if Y_prediction[0,i] != Y_test[0,i]:
#         incorrects.append((X_test[:,i], Y_test[0,i]))
#
# print(len(incorrects))
#
#
# ax = []
# fig=plt.figure(figsize=(10, 6))
# columns = 5
# rows = 2
# convert = lambda label: 7 if label== 0 else 1
# for i in range(1, columns*rows +1):
#     img = incorrects[i-1][0].reshape(28,28)
#
#     ax.append(fig.add_subplot(rows, columns, i))
#     ax[-1].set_title("True Label:" + str(convert(np.squeeze(incorrects[i-1][1])[()])))
#     plt.imshow(img)
# plt.show()

# learning_rates = [0.4, 0.1, 0.01]
# costs = {}
# accuracy_train = {}
# accuracy_test = {}
# for i in learning_rates:
#     print ("learning rate is: " + str(i))
#     model = LogisticRegression()
#     costs[str(i)] = model.fit(X_train, Y_train, 10000, i)
#     print ('\n' + "-------------------------------------------------------" + '\n')
#
#
# for i in learning_rates:
#     plt.plot(np.squeeze(costs[str(i)]), label= "Learning rate= " + str(i))
#
# plt.ylabel('Cross entropy loss')
# plt.xlabel('Iterations ')
#
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()

