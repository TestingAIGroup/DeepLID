from multiprocessing import Pool
from keras.models import load_model, Model
from sklearn.metrics.pairwise import rbf_kernel
from utils import *
from scipy.spatial.distance import cdist
from scipy import stats
from sadl_variant.lof import LOF
import operator

def get_class_matrix(train_pred):

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label.argmax(axis=-1) not in class_matrix:
            class_matrix[label.argmax(axis=-1)] = []
        class_matrix[label.argmax(axis=-1)].append(i)
        all_idx.append(i)

    return class_matrix, all_idx

def get_lids_other(target_ats, train_ats, batch_size, layer_num, k, class_matrix, target_pred, all_idx):

    all_lids=[]; all_lids_normal=[]

    for i, at in enumerate(target_ats): # for one test input
        label = target_pred[i].argmax(axis=-1)  # get the label for one test input
        label_train_ats = train_ats[class_matrix[label]]  # training inputs has the same label with the test input
        label_other_ats=  train_ats[list(set(all_idx) - set(class_matrix[label]))] # training inputs has the different labels with the test input

        #same label
        n_batches = int(np.ceil(label_train_ats.shape[0] / float(batch_size)))
        lids_adv = []
        for i_batch in range(n_batches): # for one batch
            lid_batch_adv = estimate(i_batch, batch_size, layer_num, at, label_train_ats, k)  #一个batch中的lids
            lids_adv.append(lid_batch_adv)
        lids_adv_input=np.average(lids_adv)    #lids_adv_input=np.var(lids_adv)
        all_lids.append(lids_adv_input)

        #  different label
        normal_batches = int(np.ceil(label_other_ats.shape[0] / float(batch_size)))
        lids_normal = []
        for i_batch in range(normal_batches):  # for one batch
            lid_batch_normal = estimate(i_batch, batch_size, layer_num, at, label_other_ats, k)  # 一个batch中的lids
            lids_normal.append(lid_batch_normal)
        lids_normal_input = np.average(lids_normal)
        all_lids_normal.append(lids_normal_input)

    all_results =  np.true_divide( all_lids, all_lids_normal)

    return all_results

def find_closest_at(at, train_ats):
    #The closest distance between subject AT and training ATs.
    dist = np.linalg.norm(at - train_ats, axis=1)

    return (min(dist), train_ats[np.argmin(dist)])


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

def estimate_kernel(i_batch, batch_size, layer_num, at, train_ats, k):

    start = i_batch * batch_size
    end = np.minimum(len(train_ats), (i_batch + 1) * batch_size)
    n_feed = end - start
    lid_batch_adv = np.zeros(shape=(n_feed, layer_num))  # LID(adv):[j,1], j表示某个batch中的第几个样本, 1 表示只有1层

    #print('train_ats  shape: ', train_ats.shape)
    X_act = train_ats[start:end]
    X_adv_act = at.reshape(1, at.shape[0])
    lid_batch_adv  = (mle_batch_kernel(X_act, X_adv_act, k))

    return lid_batch_adv

def mle_batch_kernel(data, x, k):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))

    K = rbf_kernel(x, Y=data, gamma = 0.002)
    K = np.reciprocal(K) - 1
    # get the closest k neighbours
    a = np.apply_along_axis(np.sort, axis=1, arr=K)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)

    # remove inf values
    y = np.isinf(a)
    a[y] = 1000
    return a[0]

# lid of a single query point x
def mle_single(data, x, k):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))

    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)

    return a[0]


