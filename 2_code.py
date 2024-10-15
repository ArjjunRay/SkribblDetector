import numpy as np 

class Neuron:
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
    
