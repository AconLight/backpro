import random

const_delta = 0.1
random_delta = False


class Neuron:
    def __init__(self, func, is_bias):
        self.func = func
        self.is_bias = is_bias
        self.bias = 0
        self.sum_output = 0
        self.sum_input = 0

        # to be connected
        self.weights = []

    def calc_forward(self, input):
        self.last_input = input
        if self.is_bias:
            self.sum_input = self.bias
        else:
            self.sum_input = 0

        for i in range(len(input)):
            self.sum_input += input[i]*self.weights[i]

        self.sum_output = self.func.calc(self.sum_input)

    def calc_backward(self, sum_gamma_weights):
        if random_delta:
            delta = random.uniform(0.01, 0.1)
        else:
            delta = const_delta
        self.gamma = self.func.calc_der(self.sum_input) * sum_gamma_weights
        for i in range(len(self.weights)):
            self.weights[i] += delta * self.gamma * self.last_input[i]

        if self.is_bias:
            self.bias += delta * self.gamma

    def calc_backward_last(self, correct_value):
        if random_delta:
            delta = random.uniform(0.01, 0.1)
        else:
            delta = const_delta
        self.gamma = self.func.calc_der(self.sum_input) * (correct_value - self.sum_output)
        for i in range(len(self.weights)):
            self.weights[i] += delta * self.gamma * self.last_input[i]

        if self.is_bias:
            self.bias += delta * self.gamma


