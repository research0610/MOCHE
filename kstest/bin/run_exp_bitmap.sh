echo "AdExchange"
echo "log/adexchange_L.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset AdExchange --explainer L --gpu cuda:1 &> log/adexchange_L.log &

echo "ArtificialAnomaly"
echo "log/artificial_anomaly_L.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset ArtificialAnomaly --explainer L --gpu cuda:1 &> log/artificial_anomaly_L.log &

echo "AwsCloud"
echo "log/aws_L.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset AwsCloud --explainer L --gpu cuda:1 &> log/aws_L.log &

echo "KnownCause"
echo "log/knowncause_L.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset KnownCause --explainer L --gpu cuda:1 &> log/knowncause_L.log &

echo "Traffic"
echo "log/traffic_L.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset Traffic --explainer L --gpu cuda:1 &> log/traffic_L.log &

echo "Tweet"
echo "log/tweet_L.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset Tweet --explainer L --gpu cuda:1 &> log/tweet_L.log &

# nohup sh bin/run_exp_bitmap.sh &