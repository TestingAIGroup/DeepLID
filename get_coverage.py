import numpy as np
import random
def get_coverage(lower, upper, k, sa):
    buckets  = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100

def get_coverage_tn(lower, upper, k, sa):

    buckets  = np.digitize(sa, np.linspace(lower, upper, k))
    tn_buckets = list(set(buckets))
    print('tn_buckets',tn_buckets)
    count = 0
    for i in tn_buckets:
        if i >= 400:
            count = count+1
    return count/ 500 *100


def get_LDSC_coverage(original_lid):

    upper_bound = 20; n_buckets = 500
    original_coverage = get_coverage(np.amin(original_lid), upper_bound, n_buckets, original_lid)

    return original_coverage

def get_TNSC_coverage(original_lid):

    upper_bound = 20; n_buckets = 500
    original_coverage = get_coverage_tn(np.amin(original_lid), upper_bound, n_buckets, original_lid)

    return original_coverage

def get_LSC_coverage(original_lsa):


    upper_bound = 150; n_buckets = 1000
    original_coverage = get_coverage(np.amin(original_lsa), upper_bound, n_buckets, original_lsa)

    return original_coverage

def get_DSC_coverage(original_dsa):

    upper_bound = 2; n_buckets = 1000
    original_coverage = get_coverage(np.amin(original_dsa), upper_bound, n_buckets, original_dsa)

    return original_coverage

if __name__ == "__main__":
    rs_dsc =[];   rs_ldsc=[]
    path = ''

    s_index = ['s1' ]
    for s in s_index:
        rs_lsc = []
        for i in range(0, 3):
            rs_lsa = np.load(path + s +'/LSA/dave_dropout_lsa_natural_' + str(i) + '.npy')
            rs_lsc.append(get_LSC_coverage(rs_lsa))

