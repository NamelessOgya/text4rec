
# ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 1e-3 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 --debug #--recreate_data
# ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 1e-3 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
# ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 1e-3 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
# ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 1e-2 --bert_num_blocks 2 --model_init_seed 0820 --train_negative_sampling_seed 0820 
# ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 1e-2 --bert_num_blocks 2 --model_init_seed 0831 --train_negative_sampling_seed 0831  
# ./sandbox/run_and_log.sh --config_name params/b4r.yaml  --lr 1e-2 --bert_num_blocks 2 --model_init_seed 0924 --train_negative_sampling_seed 0924 

# gSASRec experiment
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 32 --model_init_seed 0820 --train_negative_sampling_seed 0820  --generate_item_embeddings
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 32 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 32 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 32 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 32 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 32 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 32 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 32 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 32 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 64 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 64 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 64 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 64 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 64 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 64 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 64 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 64 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 64 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 128 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 128 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 128 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 128 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 128 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 128 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 128 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 128 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 128 --model_init_seed 0924 --train_negative_sampling_seed 0924 

# gSASRec experiment: block3
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 32 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 64 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-2 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-3 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0820 --train_negative_sampling_seed 0820 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0831 --train_negative_sampling_seed 0831 
./sandbox/run_and_log.sh --config_name params/sasrec.yaml --lr 1e-4 --train_negative_sample_size 128 --bert_num_blocks 3 --model_init_seed 0924 --train_negative_sampling_seed 0924 
