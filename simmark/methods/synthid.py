import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from .seeding import simhash_seed, normal_seed

def get_gvalues(input_ids, hash_idx, prior_tokens, vocab_size, seed, k, b, depth, device, seed_function, tokenizer, transformer_model):
    g_values = np.empty((depth, vocab_size))
    for i in range(depth):
        seed = seed_function(input_ids, prior_tokens, tokenizer, transformer_model, hash_idx+i, seed, k, b)
        # Use simhash_seed to sample xi ~ Unif[(0,1)^vocab size]
        rng = np.random.default_rng(seed)
        g_values[i,:] = rng.integers(low=0, high=2, size=vocab_size)
    g_tensor = torch.from_numpy(g_values).to(device)

    return g_tensor

def top_p_sampling(probs, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens where cumulative probability exceeds top_p
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False).item()

    # Set the logits of the tokens beyond top_p to zero
    sorted_probs[cutoff_index+1:] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    sorted_probs = torch.where(
        torch.isfinite(sorted_probs), sorted_probs, torch.tensor(0.0)
    )
    # Sample from the filtered distribution
    next_token = torch.multinomial(sorted_probs, 1)

    # Map back to original indices
    return sorted_indices[next_token].item()

class SynthIDProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.prior_tokens = generation_config['prior_tokens']
        self.depth = generation_config['depth']
        self.model = generation_config['model']
        self.embedding_dimension = self.model.config.hidden_size
        self.k = generation_config['k']
        self.b = generation_config['b']
        self.transformer_model = generation_config['transformer_model']
        self.tokenizer = generation_config['tokenizer']
        self.seed_function = generation_config['seed_function']

    def forward(self, input_ids, logits):
        batch_size = input_ids.shape[0]
        g_values = torch.zeros(batch_size, self.depth, self.vocab_size, device=logits.device)
        for batch in range(batch_size):
            # Sample hash_idx
            rng = np.random.default_rng()
            hash_idx = rng.integers(self.k)

            # Compute xi using input_vector, hash_idx, and seed
            g_values[batch,:,:] = get_gvalues(input_ids[batch], hash_idx, self.prior_tokens, self.vocab_size, self.seed, self.k, self.b, self.depth, logits.device, self.seed_function, self.tokenizer, self.transformer_model)
    
        probs = logits.softmax(dim=-1)

        for i in range(self.depth):
            g_values_at_depth = g_values[:,i,:]
            g_mass_at_depth = (g_values_at_depth * probs).sum(dim=-1, keepdims=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        for batch in range(batch_size):
            next_token = top_p_sampling(probs[batch], 0.9)
            logits[batch,:] = 1e-5
            logits[batch,next_token] = 1e5

        return logits

from scipy.stats import rv_discrete, binom
from math import comb

def custom_distribution(n, k, m):
    """
    Custom distribution for Z = sum of m iid Y's, where Y = max(X_1,...,X_k)
    and X_i ~ Binom(n, 0.5)"""
    # CDF of X
    x = np.arange(n+1)
    F = binom.cdf(x, n, 0.5)

    # pmf of Y = max(X_1,...,X_k)
    pY = np.empty(n+1)
    pY[0] = F[0]**k
    pY[1:] = F[1:]**k - F[:-1]**k
    pY = np.clip(pY, 0.0, None)
    pY /= pY.sum()

    # support = np.arange(len(pY))
    # return rv_discrete(name="custom_max_dist", values=(support, pY))
    # pmf of Z = sum of m iid Y's (convolution)
    pZ = pY.copy()
    for _ in range(m-1):
        pZ = np.convolve(pZ, pY)
        pZ = np.clip(pZ, 0.0, None)
        pZ /= pZ.sum()

    support = np.arange(len(pZ))
    return rv_discrete(name="custom_max_dist", values=(support, pZ))

def synthid_detect(text, config):
    device = config['model'].device
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze().to(device)
    transformer_model = config['transformer_model']

    # p_values = []
    # custom_dist = custom_distribution(config['depth'], config['k'])
    total_max_cost = 0

    for i in range(1, len(ids)):
        max_cost = float('-inf')
        for hash_idx in range(config['k']):
            g_values = get_gvalues(ids[:i], hash_idx, config['prior_tokens'], config['vocab_size'], config['seed'], config['k'], config['b'], config['depth'], device, config['seed_function'], config['tokenizer'], transformer_model)
            cost = g_values[:,ids[i]].sum().item()
            max_cost = max(max_cost, cost)

        total_max_cost += max_cost
    custom_dist = custom_distribution(config['depth'], config['k'], len(ids)-1)
    p_value = 1 - custom_dist.cdf(total_max_cost)

    print(f"cost: {total_max_cost}, p-value: {p_value}")

    return p_value
    #     p_value = 1 - custom_dist.cdf(max_cost)

    #     p_values.append(p_value)
    # median_pvalue = np.median(p_values)

    # print(f"p-value: {median_pvalue}")

    # return median_pvalue