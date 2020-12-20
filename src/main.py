import math

from src import data_gen
from src.Func import Func
from src.Layer import Layer
from src.NeuralNet import NeuralNet


# program settings
epoches_numb = 10000


# functions
f_pass = lambda x: x
f_pass_der = lambda x: 1

f_sig = lambda x: 1 / (1 + math.exp(-x))
f_sig_der = lambda x: f_sig(x) * (1 - f_sig(x))


# layers
layer1 = Layer(4, Func(f_pass, f_pass_der), False)
layer2 = Layer(2, Func(f_sig, f_sig_der), True)
layer3 = Layer(4, Func(f_sig, f_sig_der), True)


# neural network
neural_net = NeuralNet()

neural_net.layers.append(layer1)
neural_net.layers.append(layer2)
neural_net.layers.append(layer3)

neural_net.connect_layers()


# training
data = data_gen.get_data1()
neural_net.train_by_epoches(data, epoches_numb)


# results
print("input: [1,0,0,0]")
print("neural net output:")
print(list(map(lambda x: "{:.2f}".format(x), neural_net.calc_forward([1,0,0,0]))))
print("hidden layer output:")
print(layer2.get_output())
print()
print("input: [0,1,0,0]")
print("neural net output:")
print(list(map(lambda x: "{:.2f}".format(x), neural_net.calc_forward([0,1,0,0]))))
print("hidden layer output:")
print(layer2.get_output())
print()
print("input: [0,0,1,0]")
print("neural net output:")
print(list(map(lambda x: "{:.2f}".format(x), neural_net.calc_forward([0,0,1,0]))))
print("hidden layer output:")
print(layer2.get_output())
print()
print("input: [0,0,0,1]")
print("neural net output:")
print(list(map(lambda x: "{:.2f}".format(x), neural_net.calc_forward([0,0,0,1]))))
print("hidden layer output:")
print(layer2.get_output())
