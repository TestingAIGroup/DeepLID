import innvestigate.utils
import numpy as np
from keras.models import load_model
import os
from utils import load_CIFAR,  filter_correct_classifications

def get_total_relevance(model, X_train, Y_train):
    analyzer = innvestigate.create_analyzer("lrp.epsilon", model, reverse_keep_tensors=True)

    X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model, X_train, Y_train)
    analysis = analyzer.analyze(X_train_corr[0:1])
    all_total_R_dense = np.zeros(analyzer._reversed_tensors[21][1].shape)

    for inp in range(len(X_train_corr)):

        analysis = analyzer.analyze(X_train_corr[inp*1:inp*1+1])
        relevance_dense = analyzer._reversed_tensors[21][1]
        all_total_R_dense += relevance_dense

    np.save('', all_total_R_dense)


def get_topk_neurons(model, num_pro):
    all_total_R_dense = np.load('')

    # activation8
    neuron_outs_dense = np.zeros((all_total_R_dense.shape[-1],))
    for i in range(all_total_R_dense.shape[-1]):
        neuron_outs_dense[i] = np.mean(all_total_R_dense[..., i])
    num_relevant_neurons_dense = round(num_pro * len(neuron_outs_dense))


    top_k_neuron_indexes_dense = (np.argsort(neuron_outs_dense, axis=None)[-num_relevant_neurons_dense:len(neuron_outs_dense)])
    total_topk_neuron_idx = {}

    total_topk_neuron_idx[model.layers[11].name] = top_k_neuron_indexes_dense


def get_total_relevance_composition(model, X_train, Y_train):

    analyzer = innvestigate.create_analyzer("lrp.epsilon", model,  reverse_keep_tensors=True)
    X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model, X_train, Y_train)

    analysis = analyzer.analyze(X_train_corr[0:1])

    all_total_R_act5 = np.zeros(analyzer._reversed_tensors[11][1].shape)
    all_total_R_act6 = np.zeros(analyzer._reversed_tensors[12][1].shape)


    for inp in range(len(X_train_corr)):
        print('inp: ', inp)
        analysis = analyzer.analyze(X_train_corr[inp * 1:inp * 1 + 1])
        relevance_act5 = analyzer._reversed_tensors[11][1]
        relevance_act6 = analyzer._reversed_tensors[12][1]

        all_total_R_act5 += relevance_act5
        all_total_R_act6 += relevance_act6


    np.save('', all_total_R_act5)
    np.save('', all_total_R_act6)


def get_topk_neurons_composition(model, num_pro):
    all_total_R_act5 = np.load('')
    all_total_R_act6 = np.load('')
    total_topk_neuron_idx= {}

    #activation5
    neuron_outs_act5 = np.zeros((all_total_R_act5.shape[-1],))
    for i in range(all_total_R_act5.shape[-1]):
        neuron_outs_act5[i] = np.mean(all_total_R_act5[..., i])
    num_relevant_neurons_act5=round(num_pro*len(neuron_outs_act5))

    # activation6
    neuron_outs_act6 = np.zeros((all_total_R_act6.shape[-1],))
    for i in range(all_total_R_act6.shape[-1]):
        neuron_outs_act6[i] = np.mean(all_total_R_act6[..., i])
    num_relevant_neurons_act6 = round(num_pro * len(neuron_outs_act6))


    top_k_neuron_indexes_act5 = (np.argsort(neuron_outs_act5, axis=None)[-num_relevant_neurons_act5:len(neuron_outs_act5)])
    total_topk_neuron_idx[model.layers[11].name]=top_k_neuron_indexes_act5

    top_k_neuron_indexes_act6 = (np.argsort(neuron_outs_act6, axis=None)[-num_relevant_neurons_act6:len(neuron_outs_act6)])
    total_topk_neuron_idx[model.layers[12].name]=top_k_neuron_indexes_act6

    return total_topk_neuron_idx


if __name__ == "__main__":

    model_path = ''
    model_name = (model_path).split('/')[-1]
    print(model_name)
    model = load_model(model_path + '.h5')

    X_train, Y_train, X_test, Y_test = load_CIFAR()


    model = innvestigate.utils.model_wo_softmax(model)
    model.summary()

