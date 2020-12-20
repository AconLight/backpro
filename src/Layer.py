import random

from src.Neuron import Neuron


class Layer:
    def __init__(self, neurons_numb, func, is_bias):
        self.neurons = []
        for i in range(neurons_numb):
            self.neurons.append(Neuron(func, is_bias))

    def calc_forward(self):
        for neuron in self.neurons:
            neuron.calc_forward(self.input)
        return self.get_output()

    def calc_forward_first(self):
        for i in range(len(self.neurons)):
            self.neurons[i].calc_forward([self.input[i]])
        return self.get_output()

    def calc_backward_last(self, correct_values):
        for i in range(len(self.neurons)):
            self.neurons[i].calc_backward_last(correct_values[i])

        my_sum_gamma_weights_list = []
        for i in range(len(self.neurons[0].weights)):
            my_sum_gamma_weights = 0
            for j in range(len(self.neurons)):
                val = self.neurons[j].gamma * self.neurons[j].weights[i]
                my_sum_gamma_weights += val

            my_sum_gamma_weights_list.append(my_sum_gamma_weights)
        return my_sum_gamma_weights_list

    def calc_backward(self, sum_gamma_wieghts_list):
        for i in range(len(self.neurons)):
            self.neurons[i].calc_backward(sum_gamma_wieghts_list[i])

        my_sum_gamma_weights_list = []
        for i in range(len(self.neurons[0].weights)):
            my_sum_gamma_weights = 0
            for j in range(len(self.neurons)):
                my_sum_gamma_weights += self.neurons[j].gamma * self.neurons[j].weights[i]

            my_sum_gamma_weights_list.append(my_sum_gamma_weights)
        return my_sum_gamma_weights_list



    def connect_to(self, layer):
        self.next = layer
        layer.prev = self
        for i in range(len(layer.neurons)):
            layer.neurons[i].weights = []
            for j in range(len(self.neurons)):
                layer.neurons[i].weights.append(random.uniform(-1, 1)) # here would come rand if we wanted


    def get_output(self):
        output = []
        for neuron in self.neurons:
            output.append(neuron.sum_output)
        return output



