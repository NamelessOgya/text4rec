"""
poetry run python sandbox/generate_embeddings.py \
--dataset_path ./Data/preprocessed/amazon_min_rating4-min_uc5-min_sc0-splitleave_one_out/dataset.pkl \
--output_path  ./Data/preprocessed/amazon_min_rating4-min_uc5-min_sc0-splitleave_one_out/item_embeddings.npy

"""

import pickle
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_e5_embedding(texts, model, tokenizer, device, batch_size=32, max_length=512):
    """
    Generate E5 embeddings for a list of texts.
    Prepends "passage: " to each text for document embedding.
    """
    all_embeddings = []
    model.eval()
    
    # Add prefix for passage embedding
    prefixed_texts = ["passage: " + str(text) for text in texts]

    with torch.no_grad():
        for i in tqdm(range(0, len(prefixed_texts), batch_size), desc="Generating Embeddings"):
            batch_texts = prefixed_texts[i:i+batch_size]
            
            inputs = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
            
            outputs = model(**inputs)
            
            # Perform average pooling
            embeddings = outputs.last_hidden_state.masked_fill(~inputs['attention_mask'][..., None].bool(), 0.0).sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]
            
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Generate E5 embeddings for item descriptions.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the preprocessed dataset.pkl file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated embeddings .npy file.")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large", help="Name of the E5 model from Hugging Face.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation.")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    item_text_dict = dataset['item_text']
    smap = dataset['smap']
    
    # The number of items is the max mapped id + 1
    num_items = max(smap.values()) + 1
    print(f"Found {len(item_text_dict)} item texts for {num_items} total items.")

    # Load E5 model and tokenizer
    print(f"Loading model and tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    # Prepare texts and corresponding indices
    # Ensure the order of texts matches the new item IDs (0 to num_items-1)
    texts_to_embed = []
    # Use a placeholder for items without text
    placeholder_text = "" 
    for i in range(num_items):
        texts_to_embed.append(item_text_dict.get(i, placeholder_text))

    # Generate embeddings
    embeddings = get_e5_embedding(texts_to_embed, model, tokenizer, device, args.batch_size)

    # Save embeddings
    print(f"Saving embeddings to {args.output_path}...")
    np.save(args.output_path, embeddings)
    print("Done.")

if __name__ == "__main__":
    main()
