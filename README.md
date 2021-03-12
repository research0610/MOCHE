# MOCHI
Comprehensible Counterfactual Interpretation on Kolmogorov-Smirnov Test

## Setup Experiment Environment

Use the following command to setup the experiment environment.

The command will install the dependencies required by MOCHI.

```
python setup.py install
```

## Interpretation Methods

`grace.py`: The baseline method GRACE [1].

`corner_search.py`: The basline method CornerSearch [2].

`mochi.py`: Our proposed method, MOCHE.

`baselines.py`: Abstract class of interpretation methods. The baseline methods STOMP [5], Series2Graph [6], and D3 [7] are implemented in the file as well.

## Run Experiments

Produce interpretations with the preference list generated by Bitmap [3]



```
sh bin/run_exp_bitmap.sh
```

Produce interpretations with the preference list generated by Spectral Residual [4]


```
sh bin/run_exp_spectral.sh
```

Produce interpretations on the failed KS test conducted on the COVID-19 dataset


```
python exp/exp_covid.py
```

## Dataset

We use the dataset in the NAB repository and the COVID-19 cases in British Columbia, Canada.


## Links

Bitmap Library: https://github.com/linkedin/luminol

Spectral Residual: https://docs.seldon.io/projects/alibi-detect/en/latest/api/modules.html

NAB Dataset: https://github.com/numenta/NAB/tree/master/data

COVID-19 Dataset: http://www.bccdc.ca/health-info/diseases-conditions/covid-19/data

## Reference

[1] Generating Concise and Informative Contrastive Sample to Explain Neural Network Model’s Prediction

[2] Sparse and Imperceivable Adversarial Attacks

[3] Assumption-free anomaly detection in timeseries

[4] Time-Series Anomaly De-tection Service at Microsoft

[5] Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View that Includes Motifs, Discords and Shapelets

[6] Series2Graph: Graph-based Subsequence Anomaly Detection for Time Series

[7] Online Outlier Detection in Sensor Data Using Non-Parametric Models
