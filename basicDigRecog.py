import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time
class Nodes():
    def __init__(self, n_x, n_h):
        self.W = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)
        self.b = np.zeros((n_h, 1))
        self.Z = None
        self.A = None
    def sigmoid(self, z):
        return 1 /(1+np.exp(-z))
    def relu(self, z):
        return np.maximum(0,z)
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    
def one_hot(y):
    one_hot_y = np.eye(10)[y]
    return one_hot_y.T   
def get_accuracy(predictions, labels):
    preddigits = np.argmax(predictions, axis=0)
    truedigits = np.argmax(labels, axis=0)
    
    return np.sum(preddigits == truedigits) / truedigits.size 
#data load
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype("float32")
y = mnist.target.astype(np.int64)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
#for convention
x_train = x_train.T
x_test = x_test.T
y_train = one_hot(y_train)
y_test = one_hot(y_test)
HL1 = Nodes(784,128)
HL2 = Nodes(128,64)
HL3 = Nodes(64,10)
m = x_train.shape[1]
a = int(input("Enter Epochs: "))
lr = 0.5
t1 = time.time()
bestacc = 0
for i in range(a):
    #forw prop
    HL1.Z = np.dot(HL1.W, x_train) + HL1.b
    HL1.A = HL1.relu(HL1.Z)
    HL2.Z = np.dot(HL2.W, HL1.A) + HL2.b
    HL2.A = HL2.relu(HL2.Z)
    HL3.Z = np.dot(HL3.W, HL2.A) + HL3.b
    HL3.A = HL3.softmax(HL3.Z)
    #error
    DZ3 = HL3.A - y_train
    #back prop
    DW3 = (1/m) * np.dot(DZ3, HL2.A.T)
    DB3 = (1/m) * np.sum(DZ3, axis=1, keepdims=True)
    DZ2 = np.dot(HL3.W.T, DZ3) * (HL2.Z > 0)
    DW2 = (1/m) * np.dot(DZ2, HL1.A.T)
    DB2 = (1/m) * np.sum(DZ2, axis=1,keepdims=True)
    DZ1 = np.dot(HL2.W.T, DZ2) * (HL1.Z > 0)
    DW1 = (1/m) * np.dot(DZ1, x_train.T)
    DB1 = (1/m) * np.sum(DZ1, axis=1, keepdims=True)
    #update
    HL3.W = HL3.W-lr* DW3
    HL3.b = HL3.b-lr* DB3
    HL2.W = HL2.W-lr* DW2
    HL2.b = HL2.b-lr* DB2
    HL1.W = HL1.W-lr* DW1
    HL1.b = HL1.b-lr* DB1
    if i % 10 == 0:
        Z1_test = np.dot(HL1.W, x_test) + HL1.b
        A1_test = HL1.relu(Z1_test)
        Z2_test = np.dot(HL2.W, A1_test) + HL2.b
        A2_test = HL2.relu(Z2_test)
        Z3_test = np.dot(HL3.W, A2_test) + HL3.b
        A3_test = HL3.softmax(Z3_test)
        loss = -np.sum(y_test * np.log(A3_test + 1e-8)) / m
        
        train_acc = get_accuracy(HL3.A, y_train)
        test_acc = get_accuracy(A3_test, y_test)
        bestacc = max(bestacc, test_acc)

        print(f"Epoch {i} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Loss: {loss:.4f}")
        
        
timelapsed = time.time() - t1
print(f"Time elapsed: {timelapsed:.2f} seconds") 
#export
weights_path = Path(__file__).resolve().parent / "mnist_weights.npz"
np.savez(weights_path, 
         W1=HL1.W, b1=HL1.b, 
         W2=HL2.W, b2=HL2.b, 
         W3=HL3.W, b3=HL3.b,
         input_size=np.array([784]),
         hidden_sizes=np.array([128, 64]),
         output_size=np.array([10]),
         acc = bestacc * 100)
        
print(f"Model saved to {weights_path}")