import wget
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
import zipfile
import gzip
import shutil
import sys
import ssl
import urllib.request
import torch
from transformers import AutoTokenizer, AutoModel


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


def download(url, savepath):
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response, open(savepath, 'wb') as out_file:
        out_file.write(response.read())


def unzip(zippath, savepath):
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()

# for gz files
def ungzip(gzpath, savepath):
    with gzip.open(gzpath, 'rb') as f_in, open(savepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def get_count(tp, id):
    groups = tp[[id]].groupby(id, as_index=False)
    count = groups.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount