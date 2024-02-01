import numpy as np


class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()


class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]


class Parameters:
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 1000


class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_layer = Layer(hidden_size, input_size)
        self.hidden_layer = Layer(output_size, hidden_size)
        self.activation = Activation()


class LossFunction:
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


class ForwardProp:
    def forward(self, model, input_data):
        hidden_layer_output = [
            model.activation.sigmoid(np.dot(neuron.weights, input_data) + neuron.bias)
            for neuron in model.input_layer.neurons
        ]
        output_layer_output = [
            model.activation.sigmoid(
                np.dot(neuron.weights, hidden_layer_output) + neuron.bias
            )
            for neuron in model.hidden_layer.neurons
        ]
        return hidden_layer_output, output_layer_output


# Inside the BackProp class
class BackProp:
    def backward(self, model, input_data, target, hidden_output, output_output):
        loss_func = LossFunction()
        output_error = target - output_output
        output_delta = output_error * np.array(
            [model.activation.sigmoid_derivative(output) for output in output_output]
        )

        hidden_error = np.dot(
            output_delta,
            np.array([neuron.weights for neuron in model.hidden_layer.neurons]).T,
        )
        hidden_delta = hidden_error * np.array(
            [model.activation.sigmoid_derivative(output) for output in hidden_output]
        )

        for i, neuron in enumerate(model.hidden_layer.neurons):
            neuron.weights += (
                np.dot(hidden_output[i], output_delta[i])
                * model.parameters.learning_rate
            )
            neuron.bias += output_delta[i] * model.parameters.learning_rate

        for i, neuron in enumerate(model.input_layer.neurons):
            neuron.weights += (
                np.dot(input_data, hidden_delta[i]) * model.parameters.learning_rate
            )
            neuron.bias += hidden_delta[i] * model.parameters.learning_rate

        return loss_func.mean_squared_error(target, output_output)


class GradDescent:
    def __init__(self):
        self.parameters = Parameters()


class Training:
    def train(self, model, input_data, target):
        forward_prop = ForwardProp()
        back_prop = BackProp()
        grad_descent = GradDescent()

        for epoch in range(grad_descent.parameters.epochs):
            hidden_output, output_output = forward_prop.forward(model, input_data)
            loss = back_prop.backward(
                model, input_data, target, hidden_output, output_output
            )

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


# Example usage:
input_size = 3
hidden_size = 4
output_size = 1

model = Model(input_size, hidden_size, output_size)
training_data = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1]])
target_data = np.array([[1], [1], [0], [0]])

trainer = Training()
trainer.train(model, input_data=training_data, target=target_data)
