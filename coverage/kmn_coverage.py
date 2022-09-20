import numpy as np
from utils import get_layer_outs_new, percent_str
from collections import defaultdict
from utils import load_MNIST
from keras.models import model_from_json, load_model
import numpy as np
from utils import get_layer_outs_new, calc_major_func_regions, percent_str
from math import floor
from utils import *
from keras.models import load_model


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class DeepGaugePercentCoverage:
    """
    Implements KMN, NBC and SNAC coverage metrics from "DeepGauge: Multi-Granularity Testing Criteria for Deep Learning
    Systems" by Ma et al.

    Supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """
    #dg = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
    def __init__(self, model, k, train_inputs=None, major_func_regions=None, skip_layers=None):
        """
        :param model: Model
        :param k: k parameter for KNC metric
        :param train_inputs: Training inputs which used to compute the major function regions. Omitted if
        major_func_regions is provided.
        :param major_func_regions: Major function regions as defined in the paper. If not supplied, model will be run
        on the whole train_inputs once. This is intended to speed up the process when the class is used in incremental
        manner.
        :param skip_layers: Layers to be skipped (e.g. flatten layers)
        """

        self.activation_table_by_section, self.upper_activation_table, self.lower_activation_table = {}, {}, {}
        self.neuron_set = set()

        self.model = model
        self.k = k
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)

        if major_func_regions is None:
            if train_inputs is None:
                raise ValueError("Training inputs must be provided when major function regions are not given")

            self.major_func_regions = calc_major_func_regions(model, train_inputs, skip_layers)
        else:
            self.major_func_regions = major_func_regions

    def get_measure_state(self):
        """
        Returns a state object that can be used to incrementally measure the coverage
        :return: The current measurement state
        """
        return [self.activation_table_by_section, self.upper_activation_table, self.lower_activation_table,
                self.neuron_set]

    def set_measure_state(self, state):
        """
        Restores the measure state to continue the measurement from that state
        :param state: Measurement state as returned from a call to get_measure_state
        :return: None
        """
        self.activation_table_by_section = state[0]
        self.upper_activation_table = state[1]
        self.lower_activation_table = state[2]
        self.neuron_set = state[3]

    def test(self, test_inputs):
        """
        Measures KMN, NBC and SNAC coverages
        :param test_inputs: Inputs
        :return: A tuple containing the coverage results along with the values that are used to compute them as
        described in the paper
        """
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        kmnc_cnt = 0
        nbc_cnt = 0
        snac_cnt = 0
        kmnc_used_inps = []
        low_used_inps = []
        strong_used_inps = []

        #print('outs shape: ',np.array(outs).shape) (10000,24,24,4)
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of a particular layer for all inputs
            cntr = 0

            for out_for_input in layer_out:  # out_for_input is output of a particular layer for single input
                for neuron_index in range(out_for_input.shape[-1]):
                    # print('neuron_index: ', neuron_index)
                    neuron_out = np.mean(out_for_input[..., neuron_index])

                    global_neuron_index = (layer_index, neuron_index)

                    self.neuron_set.add(global_neuron_index)

                    neuron_low = self.major_func_regions[layer_index][0][neuron_index]
                    neuron_high = self.major_func_regions[layer_index][1][neuron_index]
                    section_length = (neuron_high - neuron_low) / self.k
                    section_index = floor((neuron_out - neuron_low) / section_length) if section_length > 0 else 0

                    if not (global_neuron_index,
                            section_index) in self.activation_table_by_section and cntr not in kmnc_used_inps:
                        kmnc_cnt += 1
                        kmnc_used_inps.append(cntr)

                    self.activation_table_by_section[(global_neuron_index, section_index)] = True

                    f1 = False
                    f2 = False
                    if neuron_out < neuron_low:
                        if global_neuron_index not in self.lower_activation_table and cntr not in low_used_inps:
                            f1 = True
                            low_used_inps.append(cntr)
                        self.lower_activation_table[global_neuron_index] = True
                    elif neuron_out > neuron_high:
                        if global_neuron_index not in self.upper_activation_table and cntr not in strong_used_inps:
                            f2 = True
                            snac_cnt += 1
                            strong_used_inps.append(cntr)
                        self.upper_activation_table[global_neuron_index] = True

                    if f1 or f2:
                        nbc_cnt += 1

                cntr += 1

        multisection_activated = len(self.activation_table_by_section.keys())
        lower_activated = len(self.lower_activation_table.keys())
        upper_activated = len(self.upper_activation_table.keys())

        total = len(self.neuron_set)

        return (percent_str(multisection_activated, self.k * total),  #  kmn
                multisection_activated,
                percent_str(upper_activated + lower_activated, 2 * total),  #  nbc
                percent_str(upper_activated, total),  # snac
                lower_activated, upper_activated, total,
                multisection_activated, upper_activated, lower_activated, total,
                outs, kmnc_cnt, nbc_cnt, snac_cnt)

def mnist_model():
    # load the data:
    dataset = 'mnist'
    X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
    img_rows, img_cols = 28, 28


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

    skip_layers = []  # SKIP LAYERS FOR NC, KMNC, NBC etc.
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

    skip_layers = []
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)


def cifar_model():

    X_train, Y_train, X_test, Y_test = load_CIFAR()
    model_path = r''
    model_name = (model_path).split('/')[-1]
    print(model_name)
    model = load_model(model_path + '.h5')
    model.summary()

    skip_layers = []  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)


if __name__ == "__main__":

    # load the data==============:
    dataset = 'cifar10'
    X_train, Y_train, X_test, Y_test = load_CIFAR()

    # set the model====================
    model_path = r''
    model_name = (model_path).split('/')[-1]
    print(model_name)
    model = load_model(model_path + '.h5')
    model.summary()

    skip_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 16, 17, 18, 19, 20, 21, 22, 23,
                   24]  # SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):  # index,layer
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)
    print("Skipping layers:", skip_layers)


    k_sect=1000
    res = {
        'model_name': model_name,
        'num_section': k_sect,
    }

    # load the data
    path = ''
    s_index = [1,2,3,4,5]

    for s in s_index:
        kmnc_coverage = []; nbc_coverage = []
        for i in range(0, 7):
            rs_path = path + 's' + str(s) + '/cifar_adversarial_' + str(s) + '_s' + str(i) + '_.npy'
            rs = np.load(rs_path)

            dg = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
            score = dg.test(rs)  # tuple类型


            kmnc_coverage.append(float(score[0].strip('%')))
            nbc_coverage.append(float(score[2].strip('%')))
        print('KMNC: ', kmnc_coverage)
        print('nbc: ', nbc_coverage)




