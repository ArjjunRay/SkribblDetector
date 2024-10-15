import numpy as np


class Neuron:
    """Edit this Neuron class to include whatever changes you made in the previous week"""
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0


    def forward(self, inputs):
        inputs = np.array(inputs)
        # Calculate the dot product of weights and inputs and add bias
        self.output = np.dot(self.weights, inputs) + self.bias
        return self.output

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

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
        
        return self.outputs
        
class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_outputs = num_outputs

        self.layers = []
        
        # Add the first hidden layer
        self.layers.append(Layer(num_inputs, num_hidden_layer_neurons))
        
        # Add additional hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(Layer(num_hidden_layer_neurons, num_hidden_layer_neurons))
        
        # Add the output layer
        self.layers.append(Layer(num_hidden_layer_neurons, num_outputs))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)  # The output of one layer is the input to the next
        
        return inputs  # Final output is from the output layer
