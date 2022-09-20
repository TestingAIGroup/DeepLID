from multiprocessing import Pool
from keras.models import Model
from utils import *

class SurpriseAdequacyDSA:
    def __init__(self,  model, train_inputs, layer_names, upper_bound, dataset, topk_neuron_idx):

        #self.surprise = surprise
        self.model = model
        self.train_inputs = train_inputs
        self.layer_names = layer_names
        self.upper_bound = upper_bound
        self.n_buckets = 1000
        self.dataset = dataset
        self.topk_neuron_idx = topk_neuron_idx
        self.save_path=''
        if dataset == 'drive': self.is_classification = False
        else: self.is_classification = True
        self.num_classes = 10
        self.var_threshold = 1e-5


    def test(self, test_inputs, dataset_name, instance='dsa'):

        if instance == 'dsa':

            target_ats, train_ats, target_pred, train_pred = fetch_dsa(self.model, self.train_inputs, test_inputs,
                                   dataset_name, self.layer_names,
                                   self.num_classes, self.is_classification,
                                   self.save_path, self.dataset, self.topk_neuron_idx)
        return target_ats, train_ats, target_pred, train_pred


def fetch_dsa(model, x_train, x_target, target_name, layer_names, num_classes, is_classification, save_path, dataset, topk_neuron_idx):

    prefix = "[" + target_name + "] "
    print("prefix: ", prefix)
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset, topk_neuron_idx)

    return target_ats, train_ats, target_pred, train_pred


def _get_train_target_ats(model, x_train, x_target, target_name, layer_names,
                          num_classes, is_classification, save_path, dataset, topk_neuron_idx):
    """Extract ats of train and target inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """
    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names)

    ###############ats of train set.##########
    if os.path.exists(saved_train_path[0]):
        print("Found saved {} ATs, skip serving".format("train"))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0]) #train_ats:  (60000, 12)
        train_pred = np.load(saved_train_path[1]) #train_pred:  (60000, 10)
        #print('train_ats: ',train_ats.shape)
        #print('train_pred: ', train_pred.shape)
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            topk_neuron_idx,
            num_classes=num_classes,
            is_classification=is_classification,
            save_path=saved_train_path,

        )
        print("train ATs is saved at " + saved_train_path[0])

    saved_target_path = _get_saved_path(save_path, dataset, target_name, layer_names)

    #################ats of target set.############
    #Team DEEPLRP
    if (os.path.exists(saved_target_path[0])):
        print("Found saved {} ATs, skip serving").format(target_name)
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])
    else:
        # target就是X_train
        target_ats, target_pred = get_ats(
            model,
            x_target, #X_test
            target_name,
            layer_names,
            topk_neuron_idx,
            num_classes=num_classes,
            is_classification=is_classification,
            save_path=saved_target_path,
        )
        print(target_name + " ATs is saved at " + saved_target_path[0])

    return train_ats, train_pred, target_ats, target_pred


def get_ats( model, dataset, name, layer_names, topk_neuron_idx, save_path=None, batch_size=128, is_classification=True, num_classes=10, num_proc=10,):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = Model(
        inputs=model.input, #Tensor("input_1:0", shape=(None, 28, 28, 1), dtype=float32)
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names], #layer_name 层的神经元输出的值
    )

    print("============================")
    prefix = "[" + name + "] "

    if is_classification:
        p = Pool(num_proc)
        print(prefix + "Model serving")

        pred = model.predict(dataset, batch_size=batch_size, verbose=1)

        if len(layer_names) == 1:  #计算coverage的只有一层
            layer_outputs = [temp_model.predict(dataset, batch_size=batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)


        print(prefix + "Processing ATs")
        ats = None

        for layer_name, layer_output in zip(layer_names, layer_outputs):

            if layer_output[0].ndim == 3:
                list_top_neuron_idx = topk_neuron_idx[layer_name]
                layer_matrix = np.array(p.map(_aggr_output, [layer_output[i][:,:,list_top_neuron_idx] for i in range(len(dataset))])
                )

            else:
                list_top_neuron_idx = topk_neuron_idx[layer_name]
                layer_matrix = np.array(layer_output[:, list_top_neuron_idx])

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

        print('ats.shape: ', ats.shape)

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred

def _get_saved_path(base_path, dataset, dtype, layer_names):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.

    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.

    print("at: ",at);print(at.shape)
    print("train_ats: ",train_ats);print((train_ats.shape))
    print(type(at))
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

