import random


class NeuralNet:
    def __init__(self):
        self.layers = []

    def connect_layers(self):
        for neuron in self.layers[0].neurons:
            neuron.weights = [random.uniform(-1, 1)]
        if len(self.layers) > 1:
            for i in range(len(self.layers)-1):
                self.layers[i].connect_to(self.layers[i+1])

    def train_one(self, single_data_sample):
        self.calc_forward(single_data_sample[0])
        self.calc_backward(single_data_sample[1])

    def calc_forward(self, single_data_sample_x):
        self.layers[0].input = single_data_sample_x
        input = self.layers[0].calc_forward_first()
        for idx in range(len(self.layers) - 1):
            self.layers[idx + 1].input = input
            input = self.layers[idx+1].calc_forward()
        return input

    def calc_backward(self, single_data_sample_y):
        my_sum_gamma_weights_list = self.layers[len(self.layers)-1].calc_backward_last(single_data_sample_y)
        for idx in range(len(self.layers) - 1):
            my_sum_gamma_weights_list = self.layers[len(self.layers)-2-idx].calc_backward(my_sum_gamma_weights_list)

    def train_by_epoches(self, data, epoches_numb):
        for i in range(epoches_numb):
            for single_data_sample in data:
                self.train_one(single_data_sample)


