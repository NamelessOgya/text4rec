
# ./sandbox/run_and_log.sh --config_name params/b4r.yaml --lr 0.001 --recreate_data --num_epochs 2 --debug


# ./sandbox/run_and_log.sh --config_name params/t4r.yaml --lr 0.00001 --num_epochs 2 --debug
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml --lr 0.00001 --num_epochs 2 --debug --use_hard_negative_mining  --infonce_temperature 10

./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 0.0001 --loss_type "bce" --gbce_q 0.1 --bert_num_blocks 2 --train_negative_sample_size 256  --num_epochs 2 --debug
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml --lr 0.00001 --num_epochs 2 --debug --use_prefix_augmentation
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml --lr 0.00001 --num_epochs 2 --debug --use_hard_negative_mining --use_prefix_augmentation


./make_dashboard.sh


