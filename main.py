import numpy as np
import math

class Layer():
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs
    self.weights = np.random.rand(outputs, inputs)
    self.biases = np.random.rand(outputs)
    self.output_array = np.zeros(outputs)
    self.delta = np.zeros(outputs)
  
  def output(self):
    return self.output_array

class Network():
  def __init__(self, layers):
    self.layers = layers

  def set_input(self, inputs):
    self.inputs = inputs

  def set_output(self, output):
    self.outputs = output

  def output(self):
    return self.layers[-1].output()

  # Training
  def train(self, X, y, epoch, lr):
    for i in range(epoch):
      for X_, y_ in zip(X, y):
        self.set_input(X_)
        self.set_output(y_)
        self.forward_pass()
        self.backward_pass(lr)
      print('Epoch %d/%d' % (i+1, epoch))

  def predict(self, X):
    prediction = []
    for X_ in X:
      for layer in self.layers:
        self.forward(layer, X_)
        X_ = layer.output_array
      prediction.append(self.output())
    return prediction

  def test(self, X, y):
    correct = 0
    for X_, y_ in zip(X, y):
      for layer in self.layers:
        self.forward(layer, X_)
        X_ = layer.output_array
      if (self.output() == y_).all():
        correct += 1
    return correct / X.shape[0]

  # Forward pass
  def forward_pass(self):
    inputs = self.inputs
    for layer in self.layers:
      self.forward(layer, inputs)
      inputs = layer.output_array

  def forward(self, layer, inputs):
    for i in range(layer.outputs):
      temp = 0
      for j in range(inputs.shape[0]):
        temp += (layer.weights[i][j] * inputs[j])
      layer.output_array[i] = self.sigmoid(temp - layer.biases[i])

  def sigmoid(self, value):
    result = 1 / (1 + math.exp(-value))
    return result

  
  # Backward pass
  # TODO: create backward pass
  def backward_pass(self, lr):
    outputs = self.outputs
    inputs = self.inputs
    for i in range(len(self.layers)-1, -1, -1):
      if i == len(self.layers)-1:
        self.backward(self.layers[i], lr, self.layers[i-1].output_array, mode='output', outputs=outputs)
      elif i == 0:
        self.backward(self.layers[i], lr, inputs, mode='hidden', next_weights=self.layers[i+1].weights, next_delta=self.layers[i+1].delta)
      else:
        self.backward(self.layers[i], lr, self.layers[i-1].output_array, mode='hidden', next_weights=self.layers[i+1].weights, next_delta=self.layers[i+1].delta)


  def backward(self, layer, lr, prev_outputs, mode, outputs=None, next_weights=None, next_delta=None):
    # Calculate deltas
    if mode == 'output':
      for i in range(layer.outputs):
        layer.delta[i] = (outputs[i]-layer.output_array[i]) * layer.output_array[i] * (1-layer.output_array[i])
    elif mode == 'hidden':
      for i in range(layer.outputs):
        temp = 0
        for j in range(next_delta.shape[0]):
          temp += next_weights[j][i] * next_delta[j]
      layer.delta[i] = layer.output_array[i] * (1-layer.output_array[i]) * temp

    # Calculate new weight
    for i in range(layer.outputs):
      for j in range(layer.inputs):
        delta_w = lr * layer.delta[i] * prev_outputs[j]
      layer.weights[i] += delta_w
      

if __name__ == "__main__":
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
  from sklearn.datasets import load_iris

  iris=load_iris()
  x=iris.data
  y=iris.target
  # Split data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 3)

  # Normalize feature data
  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # One hot encode target values
  one_hot = OneHotEncoder()
  y_train_hot = np.array(one_hot.fit_transform(y_train.reshape(-1, 1)).todense())
  y_test_hot = np.array(one_hot.transform(y_test.reshape(-1, 1)).todense())

  network = Network(
    [
      Layer(4,64),
      Layer(64,128),
      Layer(128,64),
      Layer(64,3)
    ]
  )

  epoch = 10
  lr = 0.01
  network.train(X_train_scaled, y_train_hot, epoch, lr)

  pred = network.predict(X_test_scaled)
  
  for x_, y_ in zip(pred, y_test_hot):
    print(x_, y_)