import copy
import random
import time
from collections import defaultdict
import tqdm
import numpy as np
from kstest.log import getLogger

logging = getLogger(__name__)

RED = 1
BLUE = 2

VALUE_INDX = 0
TYPE_INDX = 1
INDICE_INDX = 2

POSITION_INDX = 1
RM_BALLS_INDX = 0

UPPER_BOUND_INDX = 0
LOWER_BOUND_INDX = 1


def compute_ks(x_1, x_2):
    x = sorted([(i, RED, idx) for idx, i in enumerate(x_1)] + [(i, BLUE, idx) for idx, i in enumerate(x_2)],
               key=lambda i: i[VALUE_INDX])
    cdf_1 = {}
    cdf_2 = {}
    red_count = 0
    blue_count = 0
    for value, ball_type, _ in x:
        if ball_type == RED:
            red_count += 1
        else:
            blue_count += 1
        cdf_1[value] = red_count
        cdf_2[value] = blue_count
    n = len(x_1)
    m = len(x_2)
    max_d = -1000000
    for (i, ball_type, _) in x:
        d = abs(cdf_1[i] / n - cdf_2[i] / m)
        max_d = max(max_d, d)
    weight = np.sqrt(n * m / (n + m))
    return weight * max_d


class Interpreter:
    def __init__(self, x_te: list, x_tr: list, pref_rank: list, level):
        self.c = np.sqrt(- np.log(level / 2) * 0.5)
        logging.info(f"Threshold is {self.c}")
        self.n = len(x_te)
        self.m = len(x_tr)
        self.pref_rank = pref_rank
        x = [(i, RED, idx) for idx, i in enumerate(x_te)] + [(i, BLUE, idx) for idx, i in enumerate(x_tr)]
        self.x = sorted(x, key=lambda i: i[VALUE_INDX])
        self.x_te = x_te
        self.x_tr = x_tr

        self.x_1_positions = [0] * self.n
        for idx, (_, ball_type, x_1_idx) in enumerate(self.x):
            if ball_type == RED:
                self.x_1_positions[x_1_idx] = idx

        cdf_1 = {}
        cdf_2 = {}
        cdf = {}
        red_count = 0
        blue_count = 0
        self.red_count = defaultdict(int)
        for value, ball_type, _ in self.x:
            if ball_type == RED:
                red_count += 1
                self.red_count[value] += 1
            else:
                blue_count += 1
            cdf_1[value] = red_count
            cdf_2[value] = blue_count
            cdf[value] = red_count + blue_count

        self.red_before_for_cdf = []
        self.blue_before_for_cdf = []
        self.cdf_red = cdf_1
        self.cdf_blue = cdf_2
        for idx, (value, ball_type, _) in enumerate(self.x):
            self.red_before_for_cdf.append(cdf_1[value])
            self.blue_before_for_cdf.append(cdf_2[value])

    def _cmpt_threshold_related_component(self, k, critical_value):
        # ===================================================
        # Compute sqrt{n-k + \frac{(n-k)^2}{m}} * c
        # ===================================================
        return np.sqrt(self.n - k + (self.n - k) ** 2 / self.m) * critical_value

    def weighted_cdf_diff(self, idx, r, k):
        # ===================================================
        # Compute the weighted CDF difference at a position
        # ===================================================
        red_x_i = self.red_before_for_cdf[idx]
        blue_x_i = self.blue_before_for_cdf[idx]
        x_1_cdf = (red_x_i - r) / (self.n - k)
        x_2_cdf = blue_x_i / self.m
        cdf_diff = abs(x_1_cdf - x_2_cdf)
        weight = np.sqrt((self.n - k) * self.m / (self.n - k + self.m))
        # logging.info(f"Red_x_i {red_x_i} Blue_x_i {blue_x_i} r {r} k {k} n {self.n} m {self.m}")
        return weight * cdf_diff

    def compute_upper_lower_bounds_uniq_value(self, k):
        critical_value = self.c
        threshold_related = self._cmpt_threshold_related_component(k, critical_value)
        M_i = -np.infty

        range_r = {}

        for value, _, _ in self.x:
            r_b = self.cdf_red[value] - (self.n - k) / self.m * self.cdf_blue[value]
            M_i = max(M_i, r_b)
            # 1e-10 is used to compensate the computation error of float number
            upper_bound = int(min(k, self.cdf_red[value], np.floor(r_b + threshold_related + 1e-10)))
            lower_bound = int(max(0, k - self.n + self.cdf_red[value], np.ceil(M_i - threshold_related - 1e-10)))
            # assert upper_bound >= lower_bound, (k, self.cdf_red[value], r_b + threshold_related, k - self.n + self.cdf_red[value], M_i - threshold_related)
            assert upper_bound >= lower_bound
            range_r[value] = (upper_bound, lower_bound)
        range_r = [kv for kv in sorted(range_r.items(), key=lambda i: i[0])]
        return range_r

    def compute_weight_ks_statistic(self, path, k):
        weighted_ks_statistic = -np.infty
        for idx in range(len(self.x)):
            weighted_cdf_diff = self.weighted_cdf_diff(idx, path[idx], k)
            weighted_ks_statistic = max(weighted_ks_statistic, weighted_cdf_diff)
        removed_positions = set()
        prev = 0

        for idx in range(len(self.x)):
            assert prev <= path[idx] <= prev + 1
            if path[idx] > prev:
                removed_positions.add(idx)
            prev = path[idx]
        return weighted_ks_statistic, tuple([self.x[i][INDICE_INDX] for i in removed_positions])

    def do_interpretation_pref_rank(self):
        k = self.find_k()[0]
        logging.info(f"Finish finding K {k}")
        if k == 0:
            return None, None
        return self.do_interpretation_given_k(k)

    def do_interpretation_given_k_no_prune(self, k):
        """
        Construction Algorithm without Pruning Data Items
        """
        range_r_origin = self.compute_upper_lower_bounds_uniq_value(k)
        value_index = {}
        values = []
        for idx, i in enumerate(range_r_origin):
            value_index[i[0]] = idx
            values.append(i[0])
        m = len(range_r_origin)
        range_r_origin = [(i[1][UPPER_BOUND_INDX], i[1][LOWER_BOUND_INDX]) for i in range_r_origin]

        removed_x_1 = set()
        removed_values = defaultdict(int)

        while len(removed_x_1) != k:
            qualified_instances = []
            for instance_id in self.pref_rank:
                if instance_id in removed_x_1:
                    continue

                # print(range_r)
                idx_in_X = self.x_1_positions[instance_id]
                x_value = self.x[idx_in_X][VALUE_INDX]
                removed_values[x_value] += 1
                find = True

                range_r_prime = [range_r_origin[-1], ]

                for i in range(m - 2, -2, -1):
                    # for i in range(start_idx, -2, -1):
                    u_i_plus_1_prime = range_r_prime[-1][UPPER_BOUND_INDX]
                    u_i_plus_1_value = values[i + 1]
                    if i != -1:
                        u_i = range_r_origin[i][UPPER_BOUND_INDX]
                    else:
                        u_i = 0
                    u_i_prime = min(u_i, u_i_plus_1_prime - removed_values[u_i_plus_1_value])
                    if i != -1:
                        l_i = range_r_origin[i][LOWER_BOUND_INDX]
                    else:
                        l_i = 0
                    if i != -1:
                        range_r_prime.append((u_i_prime, l_i))

                    if l_i > u_i_prime:
                        find = False
                        break

                if find:
                    qualified_instances.append(instance_id)
                removed_values[x_value] -= 1
            instance_id = qualified_instances[0]
            removed_x_1.add(instance_id)
            idx_in_X = self.x_1_positions[instance_id]
            x_value = self.x[idx_in_X][VALUE_INDX]
            removed_values[x_value] += 1


        _x_1 = [i for idx, i in enumerate(self.x_te) if idx not in set(removed_x_1)]
        assert len(_x_1) == self.n - k, (len(removed_x_1), k)
        # =====================================
        # Verify the interpretation
        # =====================================
        _ks = compute_ks(_x_1, self.x_tr)

        assert _ks < self.c, (_ks, "Removed X1", len(removed_x_1), "k", k)
        # print("==========My Time==========", time.time() - st)
        return removed_x_1, _ks

    def do_interpretation_given_k(self, k):
        """
        Efficient Construction Algorithm
        """
        range_r_origin = self.compute_upper_lower_bounds_uniq_value(k)
        value_index = {}
        values = []
        for idx, i in enumerate(range_r_origin):
            value_index[i[0]] = idx
            values.append(i[0])
        m = len(range_r_origin)
        range_r_origin = [(i[1][UPPER_BOUND_INDX], i[1][LOWER_BOUND_INDX]) for i in range_r_origin]

        removed_x_1 = set()
        removed_values = defaultdict(int)

        st = time.time()
        range_r_prime_prev = None
        for instance_id in self.pref_rank:
            # print(range_r)
            idx_in_X = self.x_1_positions[instance_id]
            x_value = self.x[idx_in_X][VALUE_INDX]
            removed_values[x_value] += 1
            find = True

            range_r_prime = [range_r_origin[-1], ]
            start_idx = value_index[x_value] - 1

            for i in range(m - 2, -2, -1):
                # for i in range(start_idx, -2, -1):
                u_i_plus_1_prime = range_r_prime[-1][UPPER_BOUND_INDX]
                u_i_plus_1_value = values[i + 1]
                if i != -1:
                    u_i = range_r_origin[i][UPPER_BOUND_INDX]
                else:
                    u_i = 0
                u_i_prime = min(u_i, u_i_plus_1_prime - removed_values[u_i_plus_1_value])
                if i != -1:
                    l_i = range_r_origin[i][LOWER_BOUND_INDX]
                else:
                    l_i = 0
                if i != -1:
                    range_r_prime.append((u_i_prime, l_i))

                if l_i > u_i_prime:
                    find = False
                    break

            if find:
                removed_x_1.add(instance_id)
            else:
                removed_values[x_value] -= 1
            if len(removed_x_1) == k:
                break

        _x_1 = [i for idx, i in enumerate(self.x_te) if idx not in set(removed_x_1)]
        assert len(_x_1) == self.n - k, (len(removed_x_1), k)
        # =====================================
        # Verify the interpretation
        # =====================================
        _ks = compute_ks(_x_1, self.x_tr)

        assert _ks < self.c, (_ks, "Removed X1", len(removed_x_1), "k", k)
        # print("==========My Time==========", time.time() - st)
        return removed_x_1, _ks

    def find_k(self):
        k_estimate = self.binary_search_k()
        k = k_estimate
        # ====================
        # Debug
        if k_estimate > 0:
            assert self.check_k(k_estimate - 1, if_necessary_sufficient=True) == False
        # ====================
        while not self.check_k(k, if_necessary_sufficient=True):
            k = k + 1
        return k, k_estimate

    def find_k_brute_force(self):
        k = 0
        while not self.check_k(k, if_necessary_sufficient=True):
            k = k + 1
        return k

    def binary_search_k(self):
        upper = self.n - 1
        lower = 0
        k = None
        while lower <= upper:
            k = (lower + upper) // 2
            if self.check_k(k, if_necessary_sufficient=False):
                upper = k - 1
            else:
                lower = k + 1
        return k

    def check_k(self, k, if_necessary_sufficient):
        """
        Check if removing k points can keep the null-hypothesis
        """
        threshold_related = self._cmpt_threshold_related_component(k, self.c)
        M_i = -np.infty

        for idx, (_, i_type, _) in enumerate(self.x):

            r_b = self.red_before_for_cdf[idx] - (self.n - k) / self.m * self.blue_before_for_cdf[idx]
            M_i = max(M_i, r_b)

            if np.floor(r_b + threshold_related) < 0:
                return False

            if k < np.ceil(M_i - threshold_related):
                return False

            if not if_necessary_sufficient:
                if r_b + threshold_related < M_i - threshold_related:
                    return False
            else:
                if np.floor(r_b + threshold_related) < np.ceil(M_i - threshold_related):
                    return False

        return True
