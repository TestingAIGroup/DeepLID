from utils import load_MNIST
from keras.models import model_from_json, load_model
import numpy as np
from utils import get_layer_outs, get_layer_outs_new, percent_str
from utils import *
from keras.models import load_model

class DeepGaugeLayerLevelCoverage:
    """
    Implements TKN and TKN-with-pattern coverage metrics from "DeepGauge: Multi-Granularity Testing Criteria for Deep
    Learning Systems" by Ma et al.

    Supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """

    def __init__(self, model, k, skip_layers=None):
        """
        :param model: Model
        :param k: k parameter (see the paper)
        :param skip_layers: Layers to be skipped (e.g. flatten layers)
        """
        self.activation_table = {}
        self.pattern_set = set()

        self.model = model
        self.k = k
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)

    def get_measure_state(self):
        return [self.activation_table, self.pattern_set]

    def set_measure_state(self, state):
        self.activation_table = state[0]
        self.pattern_set = state[1]

    #orig_coverage, _, _, _, _, orig_incrs = dg.test(X_test)
    def test(self, test_inputs):
        """
        :param test_inputs: Inputs
        :return: Tuple consisting of coverage results along with the measurements that are used to compute the
        coverages. 0th element is the TKN value and 3th element is the pattern count for TKN-with-pattern.
        """
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        neuron_count_by_layer = {}

        layer_count = len(outs)

        inc_cnt_tkn = 0
        for input_index in range(len(test_inputs)):  # out_for_input is output of layer for single input
            pattern = []

            inc_flag = False
            for layer_index in range(layer_count):  # layer_out is output of layer for all inputs
                out_for_input = outs[layer_index][input_index]

                neuron_outs = np.zeros((out_for_input.shape[-1],))
                neuron_count_by_layer[layer_index] = len(neuron_outs)
                for i in range(out_for_input.shape[-1]):
                    neuron_outs[i] = np.mean(out_for_input[..., i])

                top_k_neuron_indexes = (np.argsort(neuron_outs, axis=None)[-self.k:len(neuron_outs)])
                pattern.append(tuple(top_k_neuron_indexes))

                for neuron_index in top_k_neuron_indexes:
                    if not (layer_index, neuron_index) in self.activation_table: inc_flag = True
                    self.activation_table[(layer_index, neuron_index)] = True

                if layer_index + 1 == layer_count:
                    self.pattern_set.add(tuple(pattern))

            if inc_flag:
                inc_cnt_tkn += 1

        neuron_count = sum(neuron_count_by_layer.values())
        covered = len(self.activation_table.keys())

        # print(percent_str(covered, neuron_count))
        # TKNC                                                         #TKNP
        return percent_str(covered, neuron_count), covered, neuron_count, len(self.pattern_set), outs, inc_cnt_tkn

def mnist_model():
    # load the data:
    dataset = 'mnist'
    X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
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

    # set the model
    model_path = ''
    model = load_model(model_path + '.h5')
    model.summary()
    model_name = (model_path).split('/')[-1]
    print(model_name)

    skip_layers = [0, 2, 4, 8, 9]  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)
    print("Skipping layers:", skip_layers)

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

    # load the data:
    dataset = 'mnist'
    X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
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

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        model = load_model(model_path + '.h5')

    model.summary()

    skip_layers = [0, 8, 9]  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    # load the data
    path = ''
    s_index = [1,2,3,4,5]
    top_k = 3

    for s in s_index:
        subset_coverage = []
        for i in range(0, 7):
            rs_path = path + 's' + str(s) + '/mnist_adversarial_' + str(s) + '_s' + str(i) + '_.npy'
            rs = np.load(rs_path)


            dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=skip_layers)  # print(percent_str(covered, neuron_count))
            orig_coverage,_, _, _, _, orig_incrs = dg.test(rs)

            subset_coverage.append(float(orig_coverage.strip('%')))





