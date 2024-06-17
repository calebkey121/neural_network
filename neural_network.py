try:
    from .neuron import Neuron  # Relative import for package use
except ImportError:
    from neuron import Neuron  # Absolute import as fallback for script use
import numpy as np

class NeuralNetwork():
    def __init__(self, layout, x_train, y_train, x_test, y_test, learning_rate=0.1) -> None:
        """
        Initializes a NeuralNetwork instance with specified parameters and default attributes.

        Parameters:
            layout (List[int]): List of the number of neurons per layer, e.g., [3, 4] indicates two layers with 3 neurons in the first and 4 in the second.
            x_train (List[float]): Training inputs to the network.
            y_train (List[float]): Hot (ground truth).
            x_test (List[float]): Testing inputs to the network.
            y_test (List[float]): Testing output values of the network (ground truth).
            learning_rate (float): Learning rate of the network, affecting the adjustment magnitude of weights during training.

        Attributes:
            network (List[List[Neuron]]): The network's layers, each containing its neurons, initialized based on the layout and weights.
            output (List[float]): Output from the last forward pass of the network. Initially None.
            error (float): Mean Squared Error (MSE) of the network after the last training step. Initially None.
        """
        # Parameters
        self.layout = layout
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        assert len(y_train[0]) == len(y_test[0]) == layout[-1] # must match size of output layer
        assert len(x_train) == len(y_train) # must have an output for every input
        assert len(x_test) == len(y_test) # likewise
        self.learning_rate = learning_rate

        # Attributes
        self.network = self.initialize_network() # gives us 2d list of neurons to run
        self.output_layer = None
        self.error_terms = None


    def initialize_network(self):
        """
        Initializes all neurons in the network.
        """
        network = []
        # in the forward pass we make sure to actually give it inputs
        num_layers = len(self.layout)
        input_len = len(self.x_train[0])
        for layer in range(num_layers):
            num_neurons = self.layout[layer]
            network.append([])
            for neuron in range(num_neurons):
                # we want tanh for hidden, sigmoid for last layer
                activation_name = "logistic_sigmoid" if layer == num_layers - 1 else "tanh"
                n = Neuron(input_len=input_len, activation_name=activation_name)
                network[layer].append(n)
            input_len = num_neurons + 1 # input for next layer (don't forget we're adding bias)
        return network


    def forward_pass(self, x):
        """
        Computes the output of the network given inputs
        """
        inputs = x
        num_layers = len(self.layout)
        for layer in range(num_layers):
            layer_output = [ 1.0 ] # initialize with 1 for bias term
            num_neurons = self.layout[layer]
            for neuron in range(num_neurons):
                output = self.network[layer][neuron].update_inputs(inputs)
                layer_output.append(output)
            inputs = layer_output
        self.output_layer = np.array(layer_output[1:]) # output layer (has a bias term for hidden layers not needed for output)
    

    def backward_pass(self, expected):
        """
        Determines all of the error terms needed for gradient descent 
        """
        der_error_function = [ - ( a - b ) for a, b in zip(expected, self.output_layer) ]
        # last layer is a bit special and it has to exist, so i do it explicitly here
        layer_errors = []
        err_last_layer = [ a * b.derivative()  for a, b in zip(der_error_function, self.network[-1]) ]
        layer_errors.insert(0, err_last_layer)

        err_next_layer = err_last_layer
        curr_layer = len(self.layout) - 2 # second to last layer
        while curr_layer >= 0:
            err_current_layer = []
            for i in range(self.layout[curr_layer]):
                weights_next_layer = self.weights_to_next_layer(position=i, next_layer=curr_layer + 1)
                dot_prod = np.dot(err_next_layer, weights_next_layer)
                neuron = self.network[curr_layer][i]
                err_current_layer.append(dot_prod * neuron.derivative())
            layer_errors.insert(0, err_current_layer)
            err_next_layer = err_current_layer
            curr_layer -= 1
        # this rounding is a weird use of the book, i would guess i take it out later
        # round(err_last_layer[0], 2) * self.weights[1][0][1] * self.layers[0][0].derivative()
        self.error_terms = layer_errors
        

    
    def weights_to_next_layer(self, position, next_layer):
        """
        Determines all weights that connect neuron at 'position' to the next layer
        """
        weights = []
        for neuron in self.network[next_layer]:
            weights.append(neuron.weights[position + 1]) # account for bias weight
        return weights
    

    def update_weights(self):
        num_layers = len(self.layout)
        for layer in range(num_layers):
            num_neurons = self.layout[layer]
            for neuron in range(num_neurons):
                self.network[layer][neuron].update_weights(lr=self.learning_rate, error_term=self.error_terms[layer][neuron])


    def loss_function(self, expected): # MSE
        total_loss = 0
        output_length = self.layout[-1] # length last layer
        for i in range(output_length):
            total_loss += ( self.network[-1][i].output - expected[i] ) ** 2
        error = total_loss / (output_length + 1) # Not entirely sure about the + 1 but the book say it will simplify things
        return error

    
    def training_loop(self, epochs):
        """
        This training loop is for one hot encoded output, the highest output should match the highest expected
        """
        index_list = list(range(len(self.x_train)))
        results = []
        for epoch in range(epochs):
            np.random.shuffle(index_list)
            correct_training_results = 0
            i = 0
            for j in index_list:
                self.forward_pass(self.x_train[j])
                if self.output_layer.argmax() == self.y_train[j].argmax():
                    correct_training_results += 1
                self.backward_pass(self.y_train[j])
                self.update_weights()
                i += 1
                if i % 1000 == 0:
                    # figure out how many digits is in len(self.x_train)
                    num_digits = len(str(len(self.x_train)))
                    print(f"Epoch {epoch + 1:>{num_digits}} - training loop: {i} / {len(index_list)}")

            correct_test_results = 0
            i = 0
            for j in range(len(self.x_test)):
                self.forward_pass(self.x_test[j])
                if self.output_layer.argmax() == self.y_test[j].argmax():
                    correct_test_results += 1
                i += 1
                if i % 1000 == 0:
                    print(f"Epoch {epoch + 1} - testing loop: {i} / {len(index_list)}")

            results.append((correct_training_results, correct_test_results))
        for i, result in enumerate(results):
            train, test = result
            print(f"Epoch {i + 1}: correct training = {train} / 60,000 - {(train / 60000) * 100}%")
            print(f"Epoch {i + 1}: correct test = {test} / 10,000 - {(test / 10000) * 100}%")
            
            
            

def main():
    x_train = np.array([
        [1,1,1],
        [0,0,0],
        [0.8,0.8,0.8],
        [0.1,0.1,0.1]
    ])
    y_train = np.array([
        [0,1],
        [1,0],
        [0,1],
        [1,0]
    ])
    x_test = np.array([
        [1,1,1],
        [0,0,0],
        [0.5,0.5,0.5],
    ])
    y_test = np.array([
        [0,0],
        [1,1],
        [0.5,0.5],
    ])
    a = NeuralNetwork (
        [4,2],
        x_train,
        y_train,
        x_test,
        y_test,
        learning_rate=0.1
    )
    a.training_loop(epochs=3)

if __name__ == "__main__":
    main()
