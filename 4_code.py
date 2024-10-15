import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Neuron:
    def __init__(self, num_inputs, activation_function):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0
        self.activation_function = activation


    def forward(self, inputs):
        inputs = np.array(inputs)
        # Calculate the dot product of weights and inputs and add bias
        self.output = np.dot(self.weights, inputs) + self.bias
        if self.activation_function == 'relu':
            self.output = relu(self.output)
        return self.output

class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_function = activation_function

        # Creating the neurons
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.outputs = []

    def forward(self, inputs):
        # Clear the output list before each forward pass
        self.outputs = []
        # Pass inputs through each neuron and store the outputs
        for neuron in self.neurons:
            neuron_output = neuron.forward(inputs)
            self.outputs.append(neuron_output)
        if self.activation_function == 'softmax':
            self.outputs = softmax(self.outputs)
        return self.outputs

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_outputs = num_outputs

        self.layers = []
        
        # Add the first hidden layer
        self.layers.append(Layer(num_inputs, num_hidden_layer_neurons, activation_function='relu'))
        
        # Add additional hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(Layer(num_hidden_layer_neurons, num_hidden_layer_neurons))
        
        # Add the output layer
        self.layers.append(Layer(num_hidden_layer_neurons, num_outputs, activation_function='softmax'))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)  # The output of one layer is the input to the next
        
        return inputs  # Final output is from the output layer
