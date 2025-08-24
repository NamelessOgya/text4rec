import torch
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from models import model_factory
from dataloaders import dataloader_factory
from utils import * # Assuming utils contains necessary helper functions

def recommend():
    parser = argparse.ArgumentParser(description='Recommend items using a trained model.')
    parser.add_argument('--dataset_code', type=str, required=True, help='Dataset code (e.g., ml-20m)')
    parser.add_argument('--model_code', type=str, default='bert', help='Model code (e.g., bert)')
    parser.add_argument('--epoch_num', type=int, required=True, help='Epoch number of the model to load')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Directory where experiments are saved')
    parser.add_argument('--experiment_description', type=str, required=True, help='Description of the experiment (e.g., test_2025-08-24_0)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top items to recommend')

    # BERT specific arguments (must match training args)
    parser.add_argument('--bert_max_len', type=int, required=True, help='Length of sequence for bert')
    parser.add_argument('--bert_hidden_units', type=int, required=True, help='Size of hidden vectors (d_model)')
    parser.add_argument('--bert_num_blocks', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, required=True, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--model_init_seed', type=int, default=None, help='Random seed for model initialization')

    args = parser.parse_args()

    # Dynamically get the number of items from the dataset
    from datasets import dataset_factory
    temp_dataset_args = SimpleNamespace(dataset_code=args.dataset_code, min_rating=4, min_uc=5, min_sc=0, split='leave_one_out', generate_item_embeddings=False)
    dataset_instance = dataset_factory(temp_dataset_args)
    preprocessed_data = dataset_instance.load_dataset()
    num_items = len(preprocessed_data['smap'])
    args.num_items = num_items

    # Reconstruct export_root to find the model
    export_root = Path(args.experiment_dir).joinpath(args.experiment_description)
    model_path = export_root.joinpath('models', f'model_epoch_{args.epoch_num}.pth')

    if not model_path.exists():
        print(f'Error: Model not found at {model_path}')
        return

    # Create a SimpleNamespace object for model and dataloader factories
    model_args = SimpleNamespace(
        **vars(args),
        # Add missing dataset args required by AbstractDataset's __init__
        min_rating=4,
        min_uc=5,
        min_sc=0,
        split='leave_one_out',
        generate_item_embeddings=False,

        # Dataloader specific args
        dataloader_code='bert',
        dataloader_random_seed=0.0,
        bert_mask_prob=0.15,
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        train_negative_sampler_code='random',
        train_negative_sample_size=1,
        train_negative_sampling_seed=0,
        test_negative_sampler_code='random',
        test_negative_sample_size=1,
        test_negative_sampling_seed=0,
    )

    # Initialize dataloaders to get the dataset object for item mapping
    _, _, test_loader = dataloader_factory(model_args)
    dataset = test_loader.dataset

    # Initialize model
    model = model_factory(model_args)
    model.to(args.device)

    # Load model state
    model_state = torch.load(model_path, map_location=args.device)
    if 'model_state_dict' in model_state:
        model.load_state_dict(model_state['model_state_dict'])
    else:
        model.load_state_dict(model_state)
    model.eval()

    print(f'Model from epoch {args.epoch_num} loaded successfully from {model_path}')

    # --- Recommendation Logic ---
    def get_recommendations(input_ids, model, dataset, top_k, device, max_len, num_items):
        if not input_ids:
            print("Input sequence is empty. Cannot recommend.")
            return []

        # Add MASK token at the end
        # The MASK token ID is num_items + 1, consistent with training
        masked_sequence = input_ids + [num_items + 1]
        
        # Pad the sequence
        padded_sequence = masked_sequence + [0] * (max_len - len(masked_sequence))
        padded_sequence = padded_sequence[:max_len]
        
        input_tensor = torch.tensor([padded_sequence], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor)
            
            # Get logits for the MASK token's position
            mask_position = len(masked_sequence) - 1
            mask_logits = logits[0, mask_position, :]

            # Exclude items already in the input sequence and padding/special tokens
            # Item IDs are 1-based, so we can ignore index 0 (padding)
            mask_logits[0] = -1e9  # Ignore padding
            for item_id in input_ids:
                if 1 <= item_id <= num_items:
                    mask_logits[item_id] = -1e9

            # Get top_k recommendations
            top_k_scores, top_k_indices = torch.topk(mask_logits, top_k)
            recommended_item_ids = top_k_indices.tolist()

            # Map item IDs back to names if available
            recommended_items = []
            if hasattr(dataset, 'idx2item'):
                for item_id in recommended_item_ids:
                    if item_id in dataset.idx2item:
                        recommended_items.append(dataset.idx2item[item_id])
                    else:
                        recommended_items.append(f'Item ID {item_id} (Unknown)')
            else:
                recommended_items = recommended_item_ids
        
        return recommended_items

    # --- Interactive Recommendation Loop ---
    print('\n--- Interactive Recommendation ---')
    print('Enter a sequence of item IDs separated by spaces (e.g., "101 205 312").')
    print('Type "exit" to quit.')

    while True:
        user_input = input("Enter item sequence: ").strip()
        if user_input.lower() == 'exit':
            break
        
        if not user_input:
            continue

        try:
            input_sequence = [int(x) for x in user_input.split()]
            if not all(1 <= i <= args.num_items for i in input_sequence):
                print(f"Error: All item IDs must be between 1 and {args.num_items}.")
                continue
            
            print(f'Input sequence (item IDs): {input_sequence}')
            
            recommendations = get_recommendations(
                input_sequence, model, dataset, args.top_k, args.device, 
                args.bert_max_len, args.num_items
            )
            
            print(f'Top {args.top_k} recommended items: {recommendations}\n')

        except ValueError:
            print("Invalid input. Please enter space-separated integer IDs.\n")

if __name__ == '__main__':
    recommend()
