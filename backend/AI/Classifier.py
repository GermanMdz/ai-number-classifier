import json
import os
import numpy as np
import pandas as pd
import time

class BaseClassifier:
    def __init__(self, weights=None, biases=None):
        self.weights = weights or []
        self.biases = biases or []
    
    def getOutput(self, x: np.ndarray):
        for b,w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w,x)+b)
        return x
        

class Classifier(BaseClassifier):
    def __init__(self, sizes, learning_rate=0.01, batch_size=16, epochs=10):
        """ Constructor de la red neuronal """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.weights = initialize_weights(sizes)
        self.biases = initialize_biases(sizes)
        
    def __backpropagation(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        a=x
        a_layers = [x]
        z_layers = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,a) + b
            z_layers.append(z)
            a = sigmoid(z)
            a_layers.append(a)
        # TODO extraer a una funcion, backward pass
        delta = (a_layers[-1] - y) * sigmoid_derivative(z_layers[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, a_layers[-2].transpose())
        for layer in range(2, self.num_layers):
            z = z_layers[-layer]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            grad_b[-layer] = delta
            grad_w[-layer] = np.dot(delta, a_layers[-layer - 1].transpose())
        return (grad_w, grad_b)
        
    def __update_parameters(self, mini_batch):
        total_gradient_weights = [np.zeros(w.shape) for w in self.weights]
        total_gradient_biases = [np.zeros(b.shape) for b in self.biases]
        for x,y in mini_batch:
            delta_gradient_weights, delta_gradient_biases = self.__backpropagation(x,y)
            total_gradient_weights = [tgw + dgw for tgw,dgw in zip(total_gradient_weights,delta_gradient_weights)]
            total_gradient_biases = [tgb + dgb for tgb,dgb in zip(total_gradient_biases,delta_gradient_biases)]
        self.weights = [w - gw * self.learning_rate for w,gw in zip(self.weights, total_gradient_weights)]
        self.biases = [b - gb * self.learning_rate for b,gb in zip(self.biases, total_gradient_biases)]

    def fit(self, x_train, y_train):
        x_train = [x.values.reshape(-1,1) for i,x in x_train.iterrows()]
        y_train = [to_categorical(n,self.sizes[-1]) for n in y_train]
        training_data = [(x, y) for (x, y) in zip(x_train, y_train)]
        n = len(training_data)
        time_start = time.time()
        for epoch in range(self.epochs):
            mini_batches = [training_data[start:start+self.batch_size] for start in range(0,n,self.batch_size)]
            for mini_batch in mini_batches:
                self.__update_parameters(mini_batch)
            print("Epoch {} complete".format(epoch))
        time_end = time.time()
        show_time(time_start,time_end)
        
                
    
    def predict(self, x_dataframe: pd.core.frame.DataFrame):
        predictions = np.zeros(shape = len(x_dataframe))
        x_dataframe.reset_index(inplace = True, drop = True)  # x_dataframe puede no comenzar por indice 0
        for i, sample in x_dataframe.iterrows() :
            prediction = self.getOutput(sample.values.reshape(-1, 1))
            predictions[i] = np.argmax(prediction)
        return predictions
    
    def to_json(self, path, name):
        with open(os.path.join(path, name), "w") as file:
            json_model = {}
            for layer in range(len(self.sizes)-1):
                json_model[f"layer{layer}"] = {
                    "weights": self.weights[layer].tolist(),
                    "biases": self.biases[layer].tolist()
                }
            json.dump(json_model, file)



# +-------------------+
# | Private functions |   
# +-------------------+

def initialize_weights(sizes):
    return [np.random.randn(x,y) for (x,y) in zip(sizes[1:],sizes[:-1])]

def initialize_biases(sizes):
    return [np.random.randn(n,1) for n in sizes[1:]]

def to_categorical(n,size):
    y = np.zeros(shape = (size,1))
    y[n,0] = 1
    return y


def sigmoid(z: np.ndarray):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0,z)

def show_time(start,end):
    x = end - start
    print("Time: {}min {}s".format(int(x//60), int(x%60)))
