import numpy as np
from utils import get_layer_outs_new, percent_str
from collections import defaultdict
from utils import load_MNIST
from keras.models import model_from_json, load_model, save_model
from utils import *
from keras.models import load_model

def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin

    return X_scaled


class NeuronCoverage:
    """
    Implements Neuron Coverage metric from "DeepXplore: Automated Whitebox Testing of Deep Learning Systems" by Pei
    et al.

    Supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """
    #  nc = NeuronCoverage(model, threshold=.75, skip_layers=skip_layers)
    def __init__(self, model, scaler=default_scale, threshold=0.75, skip_layers = None):
        self.activation_table = defaultdict(bool)

        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)

    def get_measure_state(self):
        return [self.activation_table]

    def set_measure_state(self, state):
        self.activation_table = state[0]

    #nc.test(X_test) (10000, 28, 28, 1)
    def test(self, test_inputs):
        """
        :param test_inputs: Inputs
        :return: Tuple containing the coverage and the measurements used to compute the coverage. 0th element is the
        percentage neuron coverage value.
        """

        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        used_inps = []
        nc_cnt = 0
        # print('outs shape: ', np.array(outs).shape)

        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            inp_cnt = 0

            for out_for_input in layer_out:  # out_for_input is output of layer for single input

                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    # print('neuron_index: ', neuron_index)

                    if not self.activation_table[(layer_index, neuron_index)] and np.mean(out_for_input[..., neuron_index]) > self.threshold and inp_cnt not in used_inps:
                        used_inps.append(inp_cnt)
                        nc_cnt += 1
                    #print('out_for_input[..., neuron_index： ', out_for_input[..., neuron_index].shape,out_for_input[..., neuron_index]) (24,24)
                    self.activation_table[(layer_index, neuron_index)] = self.activation_table[(layer_index, neuron_index)] or np.mean(out_for_input[..., neuron_index]) > self.threshold

                inp_cnt += 1


            if inp_cnt==1:
                break
        
        # print("self.activation_table.keys(): ", self.activation_table.keys())

        covered = len([1 for c in self.activation_table.values() if c])
        total = len(self.activation_table.keys())

        return percent_str(covered, total), covered, total, outs, nc_cnt

def mnist_model():
    # load the data:
    dataset = 'mnist'
    X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)  # 在utils中修改
    img_rows, img_cols = 28, 28

    # set the model
    model_path = r''

    model_name = (model_path).split('/')[-1]
    print(model_name)

    try:
        json_file = open(model_path + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(model_path + '.h5')

        # Compile the model before using,loss和optimizer不同，结果也会不同
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        model = load_model(model_path + '.h5')

    model.summary()

    skip_layers = [0, 8, 9]  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

def fashion_model():

    # load the data:
    dataset = 'fashion-mnist'
    X_train, Y_train, X_test, Y_test = load_fashion_MNIST()
    print('X_train: ', X_train.shape)
    print('Y_train: ', Y_train.shape)

    # set the model
    model_path = ''
    model = load_model(model_path + '.h5')
    model.summary()
    model_name = (model_path).split('/')[-1]
    print(model_name)

    skip_layers = [0, 2, 4, 8, 9]  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)


def cifar_model():

    # load the data==============:
    dataset = 'cifar10'
    X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    # set the model====================
    model_path = r''
    model_name = (model_path).split('/')[-1]
    print(model_name)
    model = load_model(model_path + '.h5')
    model.summary()

    skip_layers = [ ]  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)


if __name__ == "__main__":
    # load the data==============:
    dataset = 'cifar10'
    X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    # set the model====================
    model_path = r''
    model_name = (model_path).split('/')[-1]
    print(model_name)
    model = load_model(model_path + '.h5')
    model.summary()

    skip_layers = []  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)



    # load the data
    path = ''

    s_index = [1,2,3,4,5]

    for s in s_index:
        subset_coverage = []
        for i in range(0, 7):
            rs_path = path + 's' + str(s) + '/cifar_adversarial_' + str(s) + '_s' + str(i) + '_.npy'
            rs = np.load(rs_path)
            print('rs.shape: ', rs.shape)
            nc = NeuronCoverage(model, threshold=.75, skip_layers=skip_layers)  # SKIP ONLY INPUT AND FLATTEN LAYERS
            coverage, _, _, _, _ = nc.test(rs)
            coverage = float(coverage.strip('%'))

            subset_coverage.append(coverage)



