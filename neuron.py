import numpy as np

def logistic_sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, input_len: int, inputs: list | None = None, activation_name: str = "tanh") -> None:
        """
        Initializes a Neuron instance with given inputs, optional weights, and an activation function.

        Parameters:
            input_len (int): The number of inputs 
            inputs (Optional[List[float]]): A list of input values for the neuron. These are the inputs from the previous layer 
                                            or the initial inputs to the network. They may be set, but are updated during forward pass
            activation_name (str): The name of the activation function to use. Supported values are "tanh" for the 
                                   hyperbolic tangent function and any other value defaults to the logistic sigmoid function. 
                                   This parameter determines how the neuron's output is calculated from its weighted inputs.

        Attributes:
            output (float): The output value of the neuron after applying the activation function to its weighted inputs. 
                            This is initially None and gets updated when `compute_output` is called.
            
        Note:
            The activation function is a critical component of the neuron, defining how inputs are transformed to an output. 
            The choice between "tanh" and the logistic sigmoid affects the range and behavior of the neuron's output.
        """
        # if you set weights it must match the length of inputs, if you don't you will get random weights
        self.input_len = input_len
        if inputs:
            assert input_len == len(inputs)
            self.inputs = np.array(inputs)
        else:
            self.inputs = None
        self.weights = np.random.uniform(-0.1, 0.1, input_len)
        self.activation_name = activation_name # we separate the name and the actual function here because comparing functions is tricky
        self.activation_function = np.tanh if activation_name == "tanh" else logistic_sigmoid
        self.output = None

    def __repr__(self) -> str:
        return str(self.output)

    def compute_output(self) -> float:
        if self.inputs is None:
            raise RuntimeError("Input not initialized. Run 'update_inputs' first before 'comput_output'")
        z = np.dot(self.weights, self.inputs)
        y = self.activation_function(z)
        self.output = y
        return y
    
    def derivative(self) -> float:
        if self.activation_name == "tanh":
            return 1 - np.tanh(self.output) ** 2
        else:
            return self.output * ( 1 - self.output )
        
    def update_inputs(self, new_inputs) -> float:
        assert len(new_inputs) == self.input_len
        self.inputs = np.array(new_inputs)
        return self.compute_output()
    
    def update_weights(self, lr, error_term):
        num_weights = len(self.weights)
        for weight in range(num_weights):
            #old_weight = self.weights[weight]
            delta = -lr * self.inputs[weight] * error_term
            self.weights[weight] += delta
            #new_weight = self.weights[weight]
            #print(f"old weight: {old_weight}, new weight: {round(new_weight, 4)}")
            


def main():
    inputs = [ 1.0, 1.0, 1.0 ]
    weights = [ 0.9, -0.6, -0.5 ]
    a = Neuron(len(inputs), inputs, weights)

if __name__ == "__main__":
    main()