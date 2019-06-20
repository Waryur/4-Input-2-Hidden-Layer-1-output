import numpy as np
from matplotlib import pyplot as plt

#InputData = np.array([[0, 0, 0],
#                      [0, 0, 1],
#                      [0, 1, 0],
#                      [0, 1, 1],
#                      [1, 0, 0],
#                      [1, 0, 1],
#                      [1, 1, 0],
#                      [1, 1, 1]])

#TargetData = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])
InputData = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1],
                      
                      
                      [0, 1, 0, 0],
                      [0, 1, 0, 1],
                      [0, 1, 1, 0],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1],
                      [1, 0, 1, 0],
                      [1, 0, 1, 1],
                      [1, 1, 0, 0],
                      [1, 1, 0, 1],
                      
                      [1, 1, 1, 1]])

TargetData = np.array([[0], 
                       [1], 
                       
                       
                       [0], 
                       [1], 
                       [1], 
                       [0], 
                       [0], 
                       [1], 
                       [1], 
                       [0], 
                       [0], 
                       [1], 
                       
                       [0]])

TestData = np.array([[1, 1, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 1]])


def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

#w1 = np.random.randn(4, 3)
w1 = np.random.randn(4, 4)
b1 = np.random.randn(1, 4)

w2 = np.random.randn(4, 4)
b2 = np.random.randn(1, 4)

w3 = np.random.randn(1, 4)
b3 = np.random.randn()

iterations = 1500
lr = 0.1
costlist = []

for i in range(iterations):

    z1 = np.dot(InputData, w1.T) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2.T) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, w3.T) + b3
    a3 = sigmoid(z3)

    cost = np.square(a3 - TargetData)
    costlist.append(np.sum(cost))

    # z1, a1 = 8, 4
    # z2, a2 = 8, 4
    # z3, a3 = 8, 1
    # delta = 8, 1

    #backprop
    #dw3
    dcda3 = 2 * (a3 - TargetData)
    da3dz3 = sigmoid_p(z3)
    delta = dcda3 * da3dz3
    dz3dw3 = a2
    #dw2
    dz3da2 = w3
    da2dz2 = sigmoid_p(z2)
    dz2dw2 = a1
    #dw1
    dz2da2 = w2
    da2dz1 = sigmoid_p(z1)
    dz1dw1 = InputData

    dw3 = np.dot(delta.T, dz3dw3)
    db3 = delta
    w3 = w3 - lr * dw3
    b3 = b3 - lr * np.sum(db3, axis=0, keepdims=True)

    dw2 = np.dot((np.dot(delta, dz3da2) * da2dz2).T, dz2dw2)
    db2 = np.dot(delta, dz3da2) * da2dz2
    w2 = w2 - lr * dw2
    b2 = b2 - lr * np.sum(db2, axis=0, keepdims=True)

    dw1 = np.dot((np.dot((np.dot(delta, dz3da2) * da2dz2), dz2da2) * da2dz1).T, dz1dw1)
    db1 = np.dot((np.dot(delta, dz3da2) * da2dz2), dz2da2) * da2dz1
    w1 = w1 - lr * dw1
    b1 = b1 - lr * np.sum(db1, axis=0, keepdims=True)

print("B1: \n{}\n".format(b1))
print("B2: \n{}\n".format(b2))
print("B3: \n{}\n".format(b3))

z1 = np.dot(InputData, w1.T) + b1
a1 = sigmoid(z1)

z2 = np.dot(a1, w2.T) + b2
a2 = sigmoid(z2)

z3 = np.dot(a2, w3.T) + b3
a3 = sigmoid(z3)

cost = np.square(a3 - TargetData)
print("Prediction: \n{}\n".format(np.round(a3)))
print("Cost: \n{}\n".format(np.round(cost)))

z1 = np.dot(TestData, w1.T) + b1
a1 = sigmoid(z1)

z2 = np.dot(a1, w2.T) + b2
a2 = sigmoid(z2)

z3 = np.dot(a2, w3.T) + b3
a3 = sigmoid(z3)

print("Prediction: \n{}\n".format(np.round(a3)))

plt.plot(costlist)
plt.show()
