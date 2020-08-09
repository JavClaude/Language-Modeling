import random
from typing import List

import torch
import numpy as np

def next_word_processing(texts_to_ids: List, batch_size: int, seq_length: int):
    """
    Preprocess the texts_to_ids for next word prediction

    Args:
        texts_to_ids (List): List of tokens
        batch_size (int): Size of the batch for GD
        seq_length (int): Max seq length to process
    """
    n_batches = int(len(texts_to_ids)/(seq_length*batch_size))
    texts_to_ids = texts_to_ids[:n_batches*batch_size*seq_length] # Get the exact number of batches wrt to batch size and seq length

    target_texts = np.zeros_like(texts_to_ids)
    target_texts[:-1] = texts_to_ids[1:] #Shift data to the right
    target_texts[-1] = texts_to_ids[0]

    target_texts = target_texts.reshape(batch_size, -1)
    texts_to_ids = np.reshape(texts_to_ids, (batch_size, -1))

    return texts_to_ids, target_texts

def get_batches(train_texts: np.ndarray, target_texts: np.ndarray, batch_size: int, seq_length: int):
    """Batch Generator

    Args:
        train_texts (np.ndarray)
        target_texts (np.ndarray)
        batch_size (int): Size of the batch for GD
        seq_length (int): Max seq length to process
    """
    n_batches = np.prod(train_texts.shape) // (seq_length * batch_size)
    for i in range(0, n_batches*seq_length, seq_length):
        yield train_texts[:, i:i+seq_length], target_texts[:, i:i+seq_length]

def words_to_tokens(words: str, tokenizer, device):
    """ Encode and wraps tokens into torch.tensor

    Args:
        words (str): 
        tokenizer ([type]): 
    """
    return torch.tensor(tokenizer.encode(words).ids, dtype=torch.long, device=device).view(1, -1)

def predict_sentence(model, tokenizer, seed_str, n_length, topk, device):
    """ Predict sentence based on a pre trained model/tokenizer

    Args:
        model (nn.Module): Pre-trained nn.Model
        tokenizer (tokenizer): Pre-trained tokenizer from tokenizers
        seed_str (str): Starting point for text generation
        n_length (int): Max length for the generated
        topk ([type]): [description]
    """
    model.eval()
    
    model.to(device)

    state_h, state_c = model.reset_state(1, device=device)

    for _ in range(n_length):
        s2t = words_to_tokens(seed_str, tokenizer, device)

        preds, state_h, state_c = model(s2t, state_h, state_c)

        state_h = state_h.detach()
        state_c = state_c.detach()

        _, top_idx = torch.topk(torch.softmax(preds[0][-1], 0), k=topk)

        random_token = random.choice(top_idx.tolist())

        seed_str += tokenizer.decode([random_token])
    
    return seed_str
