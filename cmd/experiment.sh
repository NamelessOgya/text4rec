
# # ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 0.001 --recreate_data --num_epochs 100 --bert_num_blocks 3
# # ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 0.001 --num_epochs 100 --bert_num_blocks 3
# # ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 0.001 --num_epochs 100 --bert_num_blocks 3

# # b4r emb
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 3 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.7 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.5 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-2 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-3 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-4 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 5e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
./sandbox/run_and_log.sh --config_name params/t4r.yaml  --lr 1e-5 --loss_type "gbce" --gbce_q 0.3 --bert_num_blocks 2 --train_negative_sample_size 256 --gamma 0.8
