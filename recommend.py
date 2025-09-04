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
    idx2item = preprocessed_data['smap_r']
    args.num_items = len(preprocessed_data['smap'])

    # --- Load Trained Model ---
    export_root = Path(args.experiment_dir).joinpath(args.experiment_description)
    model_path = export_root.joinpath('models', 'best_acc_model.pth')
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

    # 6. Get predicted embedding and calculate similarity
    with torch.no_grad():
        sequence_output = model(padded_sequence)  # B x T x E
        predicted_embedding = sequence_output[0, mask_position, :] # 1 x E
        all_item_embeddings = model.item_embeddings # V x E
        scores = torch.matmul(predicted_embedding, all_item_embeddings.transpose(0, 1)) # 1 x V
        top_k_scores, top_k_indices = torch.topk(scores, args.top_k)
        recommended_item_indices = top_k_indices.squeeze().tolist()

    # 7. Print results
    print(f'\n--- Recommendations for the sequence: ---')
    for line in input_lines:
        print(f'  - "{line}"')

    print(f'\nTop {args.top_k} recommended items:')
    for j, item_idx in enumerate(recommended_item_indices):
        if item_idx == 0:
            # This is a padding item, skip it.
            continue
        if item_idx not in idx2item:
            print(f'  {j+1}. Item index {item_idx} not found in mapping. Skipping.')
            continue

        original_item_id = idx2item[item_idx]
        item_text = item_text_dict.get(item_idx, '(Unknown)')
        print(f'  {j+1}. Item ID {original_item_id} ({item_text})')

if __name__ == '__main__':
    recommend()