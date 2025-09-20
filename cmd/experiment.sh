
# # ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 0.001 --recreate_data --num_epochs 100 --bert_num_blocks 3
# # ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 0.001 --num_epochs 100 --bert_num_blocks 3
# # ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 0.001 --num_epochs 100 --bert_num_blocks 3

# # b4r emb/z
./sandbox/run_and_log.sh --config_name params/t4r.yaml --loss_type "infonce" --lr 1e-4 --use_hard_negative_mining --use_curriculum_learning --hard_negative_curriculum_k_initial 1 --hard_negative_curriculum_k_final 10 --hard_negative_curriculum_total_epochs 50 --enable_lr_schedule --gamma 0.95 --train_negative_sample_size 256
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "infonce" --gbce_q 1.0 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 3 --train_negative_sample_size 1 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
# ./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
