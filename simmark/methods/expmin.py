import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sentence_transformers import SentenceTransformer
from .seeding import simhash_seed, normal_seed


def top_p_sampling(probs, xi_or_stack, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens where cumulative probability exceeds top_p
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False).item()

    # Create new probs and xi
    top_p_sorted_indices = sorted_indices[:cutoff_index+1]
    top_p_probs = probs[top_p_sorted_indices]

    if isinstance(xi_or_stack, np.ndarray) and xi_or_stack.ndim == 2:
        # xi_or_stack.shape == [k, vocab]
        xi_min = xi_or_stack.min(axis=0)
    else:
        xi_min = xi_or_stack
    top_p_xi = torch.tensor(
        xi_min[top_p_sorted_indices.cpu().numpy()],
        device=top_p_probs.device,
        dtype=top_p_probs.dtype,
    )

    next_token = torch.argmin(-top_p_xi.log() / top_p_probs)

    # Map back to original indices
    return sorted_indices[next_token].item()

class ExpMinProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.model = generation_config['model']
        self.embedding_dimension = self.model.config.hidden_size
        self.vocab_size = generation_config['vocab_size']
        self.k = generation_config['k']
        self.b = generation_config['b']
        self.seed = generation_config['seed']
        self.transformer_model = generation_config['transformer_model']
        self.tokenizer = generation_config['tokenizer']
        self.prior_tokens = generation_config['prior_tokens']
        self.seed_function = generation_config['seed_function']

    def forward(self, input_ids, logits):
        batch_size = input_ids.shape[0]
        for batch in range(batch_size):
            # Sample hash_idx
            rng = np.random.default_rng()
            hash_idx = rng.integers(self.k)
            # print(f"input_ids[batch]: {input_ids[batch]}")

            gen_seed = self.seed_function(input_ids[batch], self.prior_tokens, self.tokenizer, self.transformer_model, hash_idx, self.seed, self.k, self.b)
            # print(f"hash_idx: {hash_idx}, seed for generation: {gen_seed}")

            # Compute xi using input_vector, hash_idx, and seed
            # Use simhash_seed to sample xi ~ Unif[(0,1)^vocab size]
            rng = np.random.default_rng(gen_seed)
            xi = rng.random(self.vocab_size)
    
            probs = logits[batch].softmax(dim=-1)         
            next_token = top_p_sampling(probs, xi, 0.9)
            
            logits[batch, next_token] = logits[batch, next_token] + 50.0
        
        return logits  

from scipy.stats import expon, gamma

def expmin_detect(text, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    transformer_model = config['transformer_model']
    prior_tokens = config['prior_tokens']

    avg_cost = 0
    # p_values = []

    for i in range(1, len(ids)):
        min_cost = float('inf')
        # print(f"i: {i}, tokens: {ids[:i]}")
        for hash_idx in range(config['k']):
            det_seed = config['seed_function'](ids[:i], prior_tokens, config['tokenizer'], transformer_model, hash_idx, config['seed'], config['k'], config['b'])
            # print(f"hash_idx: {hash_idx}, seed for detection: {det_seed}")
            rng = np.random.default_rng(det_seed)
            xi = rng.random(config['vocab_size'])
            cost = -np.log(max(xi[ids[i]], 1e-12))
            min_cost = min(min_cost, cost)
        avg_cost += min_cost
    avg_cost /= (len(ids) - 1)

    shape = len(ids) - 1
    rate = (len(ids) - 1) * config['k']
    p_value = gamma.cdf(avg_cost, shape, scale=1/rate)

    print(f"Detection cost: {avg_cost}, p-value: {p_value}")

    return p_value
    #     p_value = expon.cdf(min_cost, scale=1/config['k'])
    #     p_values.append(p_value)
    # median_pvalue = np.median(p_values)

    # print(f"p-value: {median_pvalue}")

    # return median_pvalue