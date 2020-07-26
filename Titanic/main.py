
from model import LogisticRegression
from Titanic.dataPrepration import create_data


train_x, train_y, test_x, test_y = create_data()
#reshape
train_x=train_x.T
train_y=train_y.T
test_x=test_x.T
test_y=test_y.T


model=LogisticRegression()
model.fit(train_x,train_y,20000,0.2)
accuracy_train, cost_train, prediction =  model.evaluate(train_x,train_y)
accuracy_test, cost_test, prediction =  model.evaluate(test_x,test_y)
print("accuracy on train set: " + str(accuracy_train))
print("cross entropy loss on train set: " + str(cost_train))
print("accuracy on test set: " + str(accuracy_test))
print("cross entropy loss on test set: " + str(cost_test))
