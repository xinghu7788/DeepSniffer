cd ../training_deepsniffer
ckpt_name="deepsniffer" 
python 2_train_seg_dd.py --model $ckpt_name
log_dir="logs_$ckpt_name"
cp $log_dir/train.log ../../Results/Figure6/logs/sequence_predictor.log
