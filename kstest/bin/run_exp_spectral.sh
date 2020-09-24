echo "AdExchange"
echo "log/adexchange_SR.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset AdExchange --explainer SR --gpu cuda:2 &> log/adexchange_SR.log &

echo "ArtificialAnomaly"
echo "log/artificial_anomaly_SR.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset ArtificialAnomaly --explainer SR --gpu cuda:2 &> log/artificial_anomaly_SR.log &

echo "AwsCloud"
echo "log/aws_SR.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset AwsCloud --explainer SR --gpu cuda:2 &> log/aws_SR.log &

echo "KnownCause"
echo "log/knowncause_SR.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset KnownCause --explainer SR --gpu cuda:2 &> log/knowncause_SR.log &

echo "Traffic"
echo "log/traffic_SR.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset Traffic --explainer SR --gpu cuda:2 &> log/traffic_SR.log &

echo "Tweet"
echo "log/tweet_SR.log"
nohup python exp/exp_concept_drift_keep_null_ts_slide_window.py --dataset Tweet --explainer SR --gpu cuda:2 &> log/tweet_SR.log &