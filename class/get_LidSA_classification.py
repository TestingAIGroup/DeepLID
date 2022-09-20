from DeepLID.utils import *
from DeepLID.dsa_coverage import SurpriseAdequacyDSA
from DeepLID.lid_coverage import get_lids_other, get_class_matrix
from DeepLID.innvestigate_code import get_topk_neurons
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
experiment_folder = ''
model_folder      = r''


if __name__ == "__main__":

    experiment_folder = ''

    # load the data==============:
    dataset='cifar10'
    X_train, Y_train, X_test, Y_test = load_CIFAR()

    # set the model====================
    model_path = r''
    model_name = (model_path).split('/')[-1]
    print(model_name)
    model = load_model(model_path + '.h5')
    model.summary()


    skip_layers = [0]
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    subject_layer=list(set(range(len(model.layers))) - set(skip_layers))[:-1]

    X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model, X_train, Y_train)

    num_pros =0.20
    total_topk_neuron_idx = get_topk_neurons(model, num_pros)

    upper_bound=2
    layer_names=[]
    for key in total_topk_neuron_idx.keys():
        layer_names.append(key)

    sa = SurpriseAdequacyDSA(model, X_train_corr, layer_names, upper_bound, dataset, total_topk_neuron_idx)
    target_ats, train_ats, target_pred, train_pred = sa.test(X_test, dataset)


    class_matrix, all_idx = get_class_matrix (train_pred)

    batch_size_list = [14000]
    for b_s in batch_size_list:
        for k in range(2600, 2620, 220):
            inputs_lids = get_lids_other(target_ats, train_ats, b_s, len(layer_names), k, class_matrix, target_pred, all_idx)
            np.save('' + str(b_s) + '_' + str(k) + '.npy', inputs_lids)


