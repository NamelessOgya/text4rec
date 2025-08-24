import torch
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModel

from models import model_factory
from dataloaders import dataloader_factory
from datasets.utils import get_e5_embedding
from utils import *

def recommend():
    parser = argparse.ArgumentParser(description='Recommend items from text using a trained BERTEmbeddingModel.')
    
    # --- Arguments for model loading ---
    parser.add_argument('--dataset_code', type=str, required=True, help='Dataset code (e.g., amazon)')
    parser.add_argument('--model_code', type=str, default='bert_embedding', help='Model code, should be bert_embedding')
    parser.add_argument('--dataloader_code', type=str, default='bert_embedding', help='Dataloader code, should be bert_embedding')
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Directory where experiments are saved')
    parser.add_argument('--experiment_description', type=str, required=True, help='Description of the experiment to load')
    parser.add_argument('--epoch_num', type=int, required=True, help='Epoch number of the model to load')
    parser.add_argument('--item_embedding_path', type=str, required=True, help='Path to item_embeddings.npy')

    # --- Arguments for recommendation ---
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top items to recommend')
    parser.add_argument('--embedding_model_name', type=str, default='intfloat/multilingual-e5-large', help='Name of the E5 model')
    parser.add_argument('--embedding_batch_size', type=int, default=32, help='Batch size for E5 embedding generation')

    # --- Arguments needed for model/dataloader initialization (must match training) ---
    parser.add_argument('--bert_hidden_units', type=int, required=True, help='Size of hidden vectors (d_model)')
    parser.add_argument('--bert_num_blocks', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, required=True, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, required=True, help='Dropout probability')
    parser.add_argument('--bert_max_len', type=int, required=True, help='Length of sequence for bert')
    parser.add_argument('--model_init_seed', type=int, default=0)
    parser.add_argument('--min_rating', type=int, default=4)
    parser.add_argument('--min_uc', type=int, default=5)
    parser.add_argument('--min_sc', type=int, default=0)
    parser.add_argument('--split', type=str, default='leave_one_out')
    parser.add_argument('--bert_mask_prob', type=float, default=0.15)
    parser.add_argument('--enable_lr_schedule', action='store_true', help='Whether to enable learning rate scheduler.')

    args = parser.parse_args()

    # --- Load Dataloader to get item mappings ---
    from datasets import dataset_factory
    dataset_instance = dataset_factory(SimpleNamespace(dataset_code=args.dataset_code, min_rating=args.min_rating, min_uc=args.min_uc, min_sc=args.min_sc, split=args.split, generate_item_embeddings=False))
    preprocessed_data = dataset_instance.load_dataset()
    item_text_dict = preprocessed_data['item_text']
    args.num_items = len(preprocessed_data['smap'])
    # Create a dummy dataloader to get the dataset object with idx2item mapping
    dummy_dataloader_args = SimpleNamespace(**vars(args), generate_item_embeddings=False, train_negative_sampler_code='random', train_negative_sample_size=0, train_negative_sampling_seed=0, test_negative_sampler_code='random', test_negative_sample_size=0, test_negative_sampling_seed=0, dataloader_random_seed=0.0, train_batch_size=1, val_batch_size=1, test_batch_size=1)
    _, _, test_loader = dataloader_factory(dummy_dataloader_args)
    dataset = test_loader.dataset

    # --- Load Trained Model ---
    export_root = Path(args.experiment_dir).joinpath(args.experiment_description)
    model_path = export_root.joinpath('models', f'model_epoch_{args.epoch_num}.pth')
    if not model_path.exists():
        print(f'Error: Model not found at {model_path}')
        return

    model = model_factory(args)
    model.to(args.device)
    model_state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()
    print(f'Model from epoch {args.epoch_num} loaded successfully from {model_path}')

    # --- Sequential Text-based Recommendation Logic ---
    print('\n--- Sequential Text-based Recommendation ---')
    
    # 1. Load E5 model for text embedding
    print(f"Loading E5 model: {args.embedding_model_name}...")
    e5_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    e5_model = AutoModel.from_pretrained(args.embedding_model_name).to(args.device)
    print("E5 model loaded.")

    # 2. Read text from stdin
    input_text = sys.stdin.read().strip()
    if not input_text:
        print("Input text is empty. Cannot recommend.")
        return
    
    input_lines = input_text.splitlines()
    print(f"Received {len(input_lines)} lines of text for sequence recommendation.")

    # 3. Generate embeddings for input text sequence using E5
    text_embeds = get_e5_embedding(input_lines, e5_model, e5_tokenizer, args.device, args.embedding_batch_size)
    text_embeds_tensor = torch.from_numpy(text_embeds).float().to(args.device)

    # 4. Prepare model input by adding a MASK embedding at the end
    mask_embedding = torch.zeros(1, text_embeds_tensor.size(1), device=args.device)
    sequence_with_mask = torch.cat([text_embeds_tensor, mask_embedding], dim=0)

    # 5. Pad or truncate the sequence to match model's max_len
    padded_sequence = sequence_with_mask
    mask_position = len(padded_sequence) - 1 # Position of the mask token before padding

    if len(padded_sequence) < args.bert_max_len:
        padding_len = args.bert_max_len - len(padded_sequence)
        padding = torch.zeros(padding_len, text_embeds_tensor.size(1), device=args.device)
        padded_sequence = torch.cat([padding, padded_sequence], dim=0)
    else:
        padded_sequence = padded_sequence[-args.bert_max_len:]
        mask_position = args.bert_max_len - 1 # Update mask position if truncated
    
    padded_sequence = padded_sequence.unsqueeze(0) # Add batch dimension

    # 6. Get model output (logits)
    with torch.no_grad():
        logits = model(padded_sequence)  # B x T x V
        
        # Get logits for the MASK token's position
        mask_logits = logits[0, mask_position, :]

        # Get top_k recommendations
        top_k_scores, top_k_indices = torch.topk(mask_logits, args.top_k)
        recommended_item_ids = top_k_indices.tolist()

    # 7. Print results
    print(f'\n--- Recommendations for the sequence: ---')
    for line in input_lines:
        print(f'  - "{line}"')

    recommended_items = []
    for item_id in recommended_item_ids:
        if item_id in item_text_dict:
            recommended_items.append(item_text_dict[item_id])
        else:
            recommended_items.append(f'Item ID {item_id} (Unknown)')

    print(f'\nTop {args.top_k} recommended items:')
    for j, item_text in enumerate(recommended_items):
        print(f'  {j+1}. {item_text}')

if __name__ == '__main__':
    recommend()
