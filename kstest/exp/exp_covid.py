from collections import defaultdict
from kstest.baselines import Covid
import numpy as np
from kstest.log import getLogger
from kstest.shift_detector import ShiftDetectorMSP
import json

logger = getLogger(__name__)

hsda2population = {
    "East Kootenay": 79856,
    "Kootenay Boundary": 78463,
    "Okanagan": 362258,
    "Thompson Cariboo Shuswap": 219467,
    "Fraser East": 295763,
    "Fraser North": 639245,
    "Fraser South": 784977,
    "Richmond": 198309,
    "Vancouver": 649028,
    "North Shore/Coast Garibaldi": 284389,
    "South Vancouver Island": 383360,
    "Central Vancouver Island": 270817,
    "North Vancouver Island": 122233,
    "Northwest": 72848,
    "Northern Interior": 141765,
    "Northeast": 68758,
}

region2code = {
    "Fraser": 2,
    "Vancouver Coastal": 3,
    "Northern": 5,
    "Vancouver Island": 4,
    "Interior": 1
}

region2population = {
    "Fraser": hsda2population["Fraser East"] + hsda2population["Fraser North"] + hsda2population["Fraser South"],
    "Vancouver Coastal": hsda2population["Richmond"] + hsda2population["Vancouver"] + hsda2population[
        "North Shore/Coast Garibaldi"],
    "Northern": hsda2population["South Vancouver Island"] + hsda2population["Central Vancouver Island"] +
                hsda2population["North Vancouver Island"],
    "Vancouver Island": hsda2population["Northwest"] + hsda2population["Northern Interior"] + hsda2population[
        "Northeast"],
    "Interior": hsda2population["East Kootenay"] + hsda2population["Okanagan"] + hsda2population["Kootenay Boundary"] +
                hsda2population["Thompson Cariboo Shuswap"]
}


def load_data():
    month2ages = defaultdict(list)
    month2population = defaultdict(list)
    month2ha = defaultdict(list)
    with open("data/covid/BCCDC_COVID19_Dashboard_Case_Details.csv") as f:
        for idx, ln in enumerate(f):
            if idx == 0:
                continue
            lnsegs = [i.strip('"') for i in ln.strip().split(",")]
            month = lnsegs[0][:7]
            age = lnsegs[3].split("-")[0]
            if age == "90+":
                age = 10
            elif age == "<10":
                age = 1
            elif age == "Unknown":
                continue
            else:
                # For example `11` -> 2
                age = int(age[0]) + 1
            if lnsegs[1] in region2code:
                month2ages[month].append(int(age))
                month2population[month].append(region2population[lnsegs[1]])
                month2ha[month].append(region2code[lnsegs[1]])
    return month2ages, month2population, month2ha


MOCHI = "MOCHI"
CORNER_SEARCH = "CORNER_SEARCH"
GRACE = "GRACE"
GREEDY = "GREEDY"
if __name__ == '__main__':
    month2ages, month2population, month2ha = load_data()
    with open("result/result_covid.json", "w") as w, open("result/data_covid.json", "w") as w_data:
        m_1 = "2020-08"
        m_2 = "2020-09"
        shift_detector = ShiftDetectorMSP(0.05, None, None)

        ref_cases = month2ages[m_1]
        test_cases = month2ages[m_2]

        X_tr_red = np.expand_dims(np.array(ref_cases), axis=1)
        X_te_red = np.expand_dims(np.array(test_cases), axis=1)
        obj = Covid(X_tr_red, X_te_red, shift_detector, level=0.05, score=month2population[m_2])
        logger.info(f"{m_1} - {m_2} {len(ref_cases)} {len(test_cases)}")

        output = {
            "M": obj.interpret_by_mci({0}),
            "DEN": obj.interpret_by_density_categorical({0}),
            GREEDY: obj.interpret_by_greedy_metric(ground_truth={0}),
            CORNER_SEARCH: obj.interpret_by_corner_search(ground_truth={0}, k_max=100),
            GRACE: obj.interpret_by_grace(ground_truth={0}, k_max=100),
        }
        ha_dist = defaultdict(int)
        for i in output["M"]["Interpretation"]:
            ha_dist[month2ha[m_2][i]] += 1

        w.write(f"{json.dumps(output)}\n")
        w_data.write(json.dumps({
            "REFERENCE": ref_cases,
            "TEST": test_cases,
            "M1": m_1,
            "M2": m_2,
        }) + "\n")

        logger.info('=' * 50)
