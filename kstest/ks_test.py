# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
from typing import List
import numpy as np


# -------------------------------------------------
# SHIFT DETECTOR
# -------------------------------------------------

def do_ks_test(reference_list: List, test_list: List, sign_level: float):
    R, T = 0, 1
    x = [(i, T) for idx, i in enumerate(test_list)] + [(i, R) for idx, i in enumerate(reference_list)]
    x = sorted(x, key=lambda i: i[0])
    cdf_te = {}
    cdf_tr = {}
    te_count = 0
    tr_count = 0
    for value, instance_type in x:
        if instance_type == T:
            te_count += 1
        else:
            tr_count += 1
        cdf_te[value] = te_count / len(test_list)
        cdf_tr[value] = tr_count / len(reference_list)
    ks_statistic = -1
    for value in cdf_te.keys():
        ks_statistic = max(ks_statistic, abs(cdf_te[value] - cdf_tr[value]))
    m, n = len(reference_list), len(test_list)
    z = np.sqrt(m * n / (m + n)) * ks_statistic

    critical_value = np.sqrt(- np.log(sign_level / 2) * 0.5)
    return z > critical_value
