from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout
from DeepLID.utils import *
from scipy.spatial.distance import cdist
import os
from multiprocessing import Pool


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_path = '.h5'
train_path = ''
train_data_path = ''


class SurpriseAdequacy:
    # sa = SurpriseAdequacy(model, x_train, layer_names, dataset)
    def __init__(self,  model, train_inputs, layer_names, dataset, topk_neuron_idx):

        #self.surprise = surprise
        self.model = model
        self.train_inputs = train_inputs
        self.layer_names = layer_names
        self.n_buckets = 1000
        self.dataset = dataset
        self.topk_neuron_idx = topk_neuron_idx
        self.save_path=''
        self.var_threshold = 1e-5


    def test(self, test_inputs, dataset_name):
        train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(model, self.train_inputs, test_inputs, dataset_name,
                                                                               self.layer_names, self.save_path, self.dataset, self.topk_neuron_idx)

        return train_ats, train_pred, target_ats, target_pred


def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]

def data_generator(xs, ys, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x, target_size) for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x, target_size) for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)

def load_data():
    path_csv= 'Driving/testing/final_example.csv'
    temp = np.loadtxt(path_csv, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 0])
    test = []
    label = []
    for i in range(len(names)):
        n = names[i]
        path_image = 'Driving/testing/center/' + n + '.jpg'
        test.append(preprocess_image(path_image))
        label.append(float(temp[i, 1]))
    test = np.array(test)
    test = test.reshape(test.shape[0], 100, 100, 3)
    label = np.array(label)
    return test, label

def load_train_data(path, batch_size=64, shape=(100, 100), ):
    xs = []; ys = []
    idx_list = []; line_list = []  #csv文件中的值先存放在数组中

    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            idx_list.append(i)
            line_list.append(line)

    for i in range(0, 101396):
        xs.append(preprocess_image(path + line_list[i].split(',')[5]))
        ys.append(float(line_list[i].split(',')[6]))

    train_xs = np.array(xs)
    train_xs = train_xs.reshape(train_xs.shape[0], 100, 100, 3)
    train_ys = np.array(ys)


    return train_xs, train_ys


def filter_error_predictions(model, x_train, y_train):
    X_corr = []
    Y_corr = []

    preds = model.predict(x_train)  # 模型输出的预测值
    print('pred:', preds)
    print('y_train:', y_train)

    for i in range(len(y_train)):
        diff = preds[i][0] - y_train[i]
        diff = abs(diff * 25)
        if diff > 10:
            print('diff: ', diff)
        else:
            X_corr.append(x_train[i])
            Y_corr.append(y_train[i])

    return np.array(X_corr), np.array(Y_corr)

def estimate(i_batch, batch_size, layer_num, at, train_ats, k):

    start = i_batch * batch_size
    end = np.minimum(len(train_ats), (i_batch + 1) * batch_size)
    n_feed = end - start
    lid_batch_adv = np.zeros(shape=(n_feed, layer_num))  # LID(adv):[j,1], j表示某个batch中的第几个样本, 1 表示只有1层

    #print('train_ats  shape: ', train_ats.shape)
    X_act = train_ats[start:end]
    X_adv_act = at.reshape(1, at.shape[0])
    lid_batch_adv  = (mle_single(X_act, X_adv_act, k))

    return lid_batch_adv

# lid of a single query point x
def mle_single(data, x, k):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]
    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))

    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)

    return a[0]


def get_lids_all(target_ats, train_ats, batch_size, layer_num, k):

    all_lids=[]

    for i, at in enumerate(target_ats): # for one test input
        #same label
        n_batches = int(np.ceil(train_ats.shape[0] / float(batch_size)))
        lids_adv = []
        for i_batch in range(n_batches): # for one batch

            lid_batch_adv = estimate(i_batch, batch_size, layer_num, at, train_ats, k)  #一个batch中的lids
            lids_adv.append(lid_batch_adv)

        lids_adv_input = np.average(lids_adv)    #lids_adv_input=np.var(lids_adv)
        all_lids.append(lids_adv_input)

    return all_lids


def _get_train_target_ats(model, X_train, x_target, target_name, layer_names, save_path, dataset, topk_neuron_idx):

    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names)

    if os.path.exists(saved_train_path[0]):
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])

    else:
        train_ats, train_pred = get_ats( model, X_train, "train", layer_names,  topk_neuron_idx, save_path=saved_train_path)

    saved_target_path = _get_saved_path(
        save_path, dataset, target_name, layer_names
    )

    if os.path.exists(saved_target_path[0]):
        print("Found saved {} ATs, skip serving".format(target_name))
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])

    else:

        target_ats, target_pred = get_ats( model, x_target, target_name, layer_names, topk_neuron_idx, save_path=saved_target_path)


    return train_ats, train_pred, target_ats, target_pred


def get_ats( model, dataset, name, layer_names, topk_neuron_idx, save_path=None, batch_size=128,  num_proc=1):

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names], )

    p = Pool(num_proc)
    pred = model.predict(dataset, batch_size=batch_size, verbose=1)

    if len(layer_names) == 1:
        layer_outputs = [temp_model.predict(dataset, batch_size=batch_size, verbose=1)]
    else:
        layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)

    ats = None
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        print("Layer: " + layer_name)

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

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def _get_saved_path(base_path, dataset, dtype, layer_names):
    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_correct10_ats_V1" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_correct10_pred_V1" + ".npy"),
    )


def find_closest_at(at, train_ats):
    #The closest distance between subject AT and training ATs.
    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), np.argmin(dist))

def training_idx(train_ats):
    all_idx = [i for i, at in enumerate(train_ats)]
    return all_idx

def Dave_dropout(input_tensor=None, load_weights=False):  # simplified dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(16, (3, 3), padding='valid', activation='relu', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = Convolution2D(32, (3, 3), padding='valid', activation='relu', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool2')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', name='block1_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool3')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(.5)(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dropout(.25)(x)
    x = Dense(20, activation='relu', name='fc3')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name="prediction")(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('/Model3.h5')

    # compiling
    m.compile(loss='mse', optimizer='adadelta')

    return m

    # load the model


def get_topk_neurons(model, num_pro):

    all_total_R_act8 = np.load(
        '.npy')
    total_topk_neuron_idx= {}

    neuron_outs_act8 = np.zeros((all_total_R_act8.shape[-1],))
    for i in range(all_total_R_act8.shape[-1]):
        neuron_outs_act8[i] = np.mean(all_total_R_act8[..., i])
    num_relevant_neurons_act8 = round(num_pro * len(neuron_outs_act8))

    print('model.layers: ', model.layers[8].name)

    top_k_neuron_indexes_act8 = (np.argsort(neuron_outs_act8, axis=None)[-num_relevant_neurons_act8:len(neuron_outs_act8)])
    total_topk_neuron_idx[model.layers[8].name]=top_k_neuron_indexes_act8

    return total_topk_neuron_idx

if __name__ == "__main__":

    # get the data
    dataset = 'drive'

    X_test, Y_test = load_data()
    print('test_data: ', X_test.shape)  # test_label:  (5614, 100, 100, 3)
    print('test_label: ', Y_test.shape)  # label:  (5614,)

    # load the model
    model = Dave_dropout()
    model.load_weights(model_path)
    model.summary()

    # 过滤到模型预测结果超过十度的训练数据, LSA原方法计算时，不需要过滤
    X_train = np.load(train_data_path + 'train_all_data.npy')
    Y_train = np.load(train_data_path + 'train_all_pred.npy')
    X_train_corr, Y_train_corr = filter_error_predictions(model, X_train, Y_train)


    num_pro = 0.40
    total_topk_neuron_idx = get_topk_neurons(model, num_pro)


    # 针对所有的层
    layer_names = []
    for key in total_topk_neuron_idx.keys():
        layer_names.append(key)


    sa = SurpriseAdequacy(model, X_train_corr, layer_names,  dataset, total_topk_neuron_idx)
    train_ats, train_pred, target_ats, target_pred = sa.test(X_test, dataset)


    all_idx = training_idx(train_ats)

    batch_size_list = [40000, 60000]
    for b_s in batch_size_list:
        for k in range(30000, 50000, 200):
            print('k: ', k)

            inputs_lids = get_lids_all(target_ats, train_ats, b_s, len(layer_names), k)
            np.save('' + str(b_s) + '_' + str(k) + '.npy', inputs_lids)


