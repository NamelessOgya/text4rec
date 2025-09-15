
import subprocess
import os
import json
import pandas as pd
from datetime import datetime

def run_experiment(params):
    """Runs a single experiment with the given parameters."""
    args = [
        'poetry', 'run', 'python', 'main.py',
        '--mode', 'train',
        '--dataset_code', 'amazon',
        '--model_code', 'bert_embedding',
        '--dataloader_code', 'bert_embedding',
        '--trainer_code', 'bert',
        '--experiment_description', 'bertemb_automated_test',
        '--device', 'cuda',
        '--num_gpu', '1',
        '--min_rating', '0',
        '--min_uc', '5',
        '--min_sc', '0',
        '--split', 'leave_one_out',
        '--train_batch_size', '128',
        '--val_batch_size', '128',
        '--test_batch_size', '128',
        '--train_negative_sampler_code', 'random',
        '--train_negative_sample_size', '0',
        '--train_negative_sampling_seed', '0',
        '--test_negative_sampler_code', 'random',
        '--test_negative_sample_size', '100',
        '--test_negative_sampling_seed', '98765',
        '--optimizer', 'Adam',
        '--enable_lr_schedule',
        '--decay_step', '25',
        '--gamma', '1.0',
        '--num_epochs', '2', # Using a small number of epochs for quick testing
        '--metric_ks', '1', '5', '10', '20', '50', '100',
        '--best_metric', 'NDCG@10',
        '--model_init_seed', '0',
        '--generate_item_embeddings',
        '--item_embedding_path', 'Data/preprocessed/amazon_min_rating0-min_uc5-min_sc0-splitleave_one_out/item_embeddings.npy',
        '--bert_dropout', '0.1',
        '--bert_hidden_units', '1024',
        '--projection_mlp_dims', '512', '256',
        '--projection_dropout', '0.1',
        '--bert_mask_prob', '0.15',
        '--bert_max_len', '100',
        '--bert_num_blocks', '2',
        '--bert_num_heads', '4'
    ]

    for key, value in params.items():
        args.extend([f'--{key}', str(value)])

    print(f"Running experiment with params: {params}")
    subprocess.run(args, check=True)

def collect_results(experiment_description):
    """Collects results from all experiments with the given description."""
    results = []
    experiment_dir = 'experiments'
    for dirname in os.listdir(experiment_dir):
        if dirname.startswith(experiment_description):
            config_path = os.path.join(experiment_dir, dirname, 'config.json')
            metrics_path = os.path.join(experiment_dir, dirname, 'logs', 'test_metrics.json')

            if os.path.exists(config_path) and os.path.exists(metrics_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                result = {**config, **metrics}
                results.append(result)

    if results:
        df = pd.DataFrame(results)
        # sort by best_metric
        df = df.sort_values(by=df.columns[df.columns.str.startswith('NDCG@10')].tolist()[0], ascending=False)
        results_path = os.path.join(experiment_dir, f'{experiment_description}_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        print(df)

if __name__ == '__main__':
    # Define the parameter grid
    param_grid = {
        'lr': [0.0001, 0.00005, 0.00001],
    }

    # Run experiments
    for lr in param_grid['lr']:
        params = {'lr': lr}
        run_experiment(params)

    # Collect and summarize results
    collect_results('bertemb_automated_test')
