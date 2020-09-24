import json
import os
import pandas as pd
from enum import Enum
import numpy as np


class TSData(Enum):
    Tweet = "realTweets"
    Traffic = "realTraffic"
    KnownCause = "realKnownCause"
    AwsCloud = "realAWSCloudwatch"
    AdExchange = "realAdExchange"
    ArtificialAnomaly = "artificialWithAnomaly"



def load_ts_dataset(dataset: str):
    for e in TSData:
        if dataset == e.name:
            return ReadNABDatasetWithTime(e)
    raise Exception(f"{dataset} is not supported")


def ReadNABDatasetWithTime(dataset):
    with open('./data/NAB/labels/combined_windows.json') as data_file:
        json_label = json.load(data_file)

    folder = f"./data/NAB/data/{dataset.value}"
    file2ts = {}
    for _, _, files in os.walk(folder):
        for file in files:
            file_name = os.path.join(folder, file)
            abnormal = pd.read_csv(file_name, header=0, index_col=0)
            abnormal['label'] = 0
            list_windows = json_label[f"{dataset.value}/{file}"]
            for window in list_windows:
                start = window[0]
                end = window[1]
                abnormal.loc[start:end, 'label'] = 1
                print(f"{file} {file_name} Abnormal Segment Length {len(abnormal.loc[start:end, 'label'])}")

            abnormal_data = abnormal['value'].to_numpy()
            abnormal_label = abnormal['label'].to_numpy()

            abnormal_data = np.expand_dims(abnormal_data, axis=1)
            abnormal_label = np.expand_dims(abnormal_label, axis=1)

            file2ts[file] = (abnormal_data, abnormal_label, abnormal.index.tolist())
            print(abnormal_data.shape)
    return file2ts

