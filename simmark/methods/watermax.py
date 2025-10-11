import torch
import numpy as np
from scipy.stats import norm
# NOTE:
import hashlib, hmac

from .seeding import simhash_seed, normal_seed

def _sample_top_p_candidates(probs, top_p, num_samples):
    # return up to num_samples token indices sampled from the top_p distribution
    # probs: 1d tensor of size vocab_size
    # top_p: float in (0,1) (nucleus threshold)
    # num_samples: int, number of samples to draw (with replacement)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # now we find cuffoff index where cumulative_probs exceeds top_p
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False).item()
    # clamp to avoid an out-of-range on top_p about 1.0
    cutoff_index = min(cutoff_index, probs.numel() - 1)

    # keep tokens up to and including cutoff_index
    top_p_sorted_indices = sorted_indices[:cutoff_index+1]
    top_p_probs = probs[top_p_sorted_indices]
    # renormalize
    top_p_probs = top_p_probs / top_p_probs.sum()  

    # lets add a numerical stability check
    kept_probs = torch.where(torch.isfinite(top_p_probs), 
                             top_p_probs, 
                             torch.tensor(0.0, device=top_p_probs.device, dtype=top_p_probs.dtype))
    if kept_probs.sum() <= 0:
        # pick single most probable token
        return top_p_sorted_indices[:1]
    
    draws = torch.multinomial(kept_probs, num_samples=num_samples, replacement=True)
    return top_p_sorted_indices[draws]

def _hmac64(salt_int, prefix_key_int, token_int):
    # NOTE: co-pilot seems to think this is correctly implremented, but we should check :)
    salt = salt_int.to_bytes(8, 'big', signed=False)
    msg = prefix_key_int.to_bytes(8, 'big', signed=False) + int(token_int).to_bytes(8, 'big', signed=False)
    mac = hmac.new(salt, msg, hashlib.sha256).digest()
    return int.from_bytes(mac[:8], 'big', signed=False)

class WaterMaxProcessor(torch.nn.Module):
    # token level low latenxy (m=1) watermax implementation, allows simhas seeding
    # at a high level, this is how it works
    # 1. at each gen step, propose num_drafts token candidates from top_p nucleus
    # 2. for each candidate, compute deterministic seed from the causal prefix including
    #    the candidate token using the seed function (which can be simhash or normal or whatever)
    # 3. seed the prng and sample a random variable u from N(0,1)
    # 4. select the candidate with the largest random variable u (i.e. smallest one step p value) 
    #    and force that token
    # some comments: per position random variable is iid under H0 (i.e. the first prefix)
    # and is seeded by a function of the current tokens and preceding h-1 tokens (hash window
    # size is h). so the random variable is deterministic function of the causal prefix
    # also, to preserve token score independence, we discard repeated n grams. so we keep a 
    # "seen signature" set of all ngrams in the current sequence for candidate token, 
    # to downweight its r.v. if the candidate token creates a repeated ngram
    
    def __init__(self, generation_config):
        super().__init__()
        self.model = generation_config['model']
        self.vocab_size = generation_config['vocab_size']

        # base seed for simhash
        self.seed = generation_config['seed']  

        self.transformer_model = generation_config['transformer_model']
        self.tokenizer = generation_config['tokenizer']
        self.prior_tokens = generation_config['prior_tokens']

        # simhash hash size
        self.k = generation_config['k'] 
        # simhash band size 
        self.b = generation_config['b']  

        self.seed_function = generation_config.get('seed_function', simhash_seed)
        # nucleus sampling threshold
        self.top_p = generation_config.get('top_p', 0.9)  
        # number of candidate tokens to propose
        self.num_drafts = generation_config.get('num_drafts', 8)  

        self.seen_signatures = set()
        
    def forward(self, input_ids, logits):
        # ok, so we will choose next token by maximizing the watermax score increment
        # for each batch row, we will
        # 1. build small token candidate set by sampling from top_p nucleus
        # 2. for each candidate, compute seed from causal prefix + candidate
        # 3. seed prng and sample u ~ N(0,1) (penalize if necessary)
        # 4. select candidate with largest u (smallest p value) and force that token

        batch_size = input_ids.shape[0]
        device = logits.device
        for batch in range(batch_size):
            # sample candidate set from top_p nucleus
            probs = logits[batch].softmax(dim=-1)

            # 1. propose candidates
            candidate_ids = _sample_top_p_candidates(probs, self.top_p, self.num_drafts)
            candidate_ids = candidate_ids.to(device)

            # NOTE: NOW, we are going tocompute a prefix key K_t 
            # from the prefix ONLY (no candidate)
            # prefix_ids = input_ids[batch]  
            # K_t = self.seed_function(prefix_ids,
            #                          self.prior_tokens,
            #                          self.tokenizer,
            #                          self.transformer_model,
            #                          hash_idx=0,
            #                          seed=self.seed,
            #                          k=self.k,
            #                          b=self.b)
            # # dedup prefix key
            # signature = K_t 

            # best_u = -np.inf
            # best_token = candidate_ids[0].item()
            # # record signature for dedup
            # best_signature = signature  

            # 2. now we will build k band keys from prefix only 
            #  and exclude current token
            prefix_ids = input_ids[batch]
            # K_list = []
            r_idx = np.random.randint(0, self.k)
            # for j in range(self.k):
            K_t = self.seed_function(prefix_ids,
                                        self.prior_tokens,
                                        self.tokenizer,
                                        self.transformer_model,
                                        hash_idx=r_idx,
                                        seed=self.seed,
                                        k=self.k,
                                        b=self.b)
                # K_list.append(K_tj)

            best_u = -np.inf
            best_token = candidate_ids[0].item()
            # we can keep prefix sig if we want to keep the dedup
            # best_signature = K_list[0]  

            for c in candidate_ids.tolist():
                # 2. compute seed from causal prefix + candidate
                #    NOTE: NOW, we derive candidate-specific seed via HMAC(K_t, c))
                cand_seed = _hmac64(self.seed, K_t, int(c))
                rng = np.random.default_rng(cand_seed)
                # sample u ~ N(0,1)
                u = rng.standard_normal() 

                # mirror watermax's dedup: repeated windows contribute zero to S
                # if signature in self.seen_signatures:
                #     # NOTE: we could also subtract a small penalty here instead of zeroing out
                #     u = 0.0

                if u > best_u:
                    best_u = u
                    best_token = c
                    # best_signature remains the prefix signature (used for dedup)

            # # 3. now, per candidate, we'll take max over k bands
            # for c in candidate_ids.tolist():
            #     u_max = -np.inf
            #     for K_tj in K_list:
            #         cand_seed = _hmac64(self.seed, K_tj, int(c))
            #         rng = np.random.default_rng(cand_seed)
            #         u = rng.standard_normal()
            #         if u > u_max:
            #             u_max = u
            #     if u_max > best_u:
            #         best_u = u_max
            #         best_token = c
            #         # best_signature = K_list[0]

            # # 4) force best token
            # logits[batch, :] = 1e-5
            # logits[batch, best_token] = 1e5

            # record the chosen signature to enforce per-text independence
            # self.seen_signatures.add(best_signature)

            # finally, force best token
            logits[batch,:] = 1e-5
            logits[batch, best_token] = 1e5

        return logits
    
from scipy.stats import norm
from scipy.signal import fftconvolve

def p_value_sum_of_max_normals(z_obs, k, m, grid_points=1000, tail=1e-12, return_debug=False):
    """
    Compute p = P(Z >= z_obs) for Z = sum_{i=1}^m Y_i where Y = max(X_1,...,X_k),
    X_j ~ N(0,1) iid.

    Parameters
    ----------
    z_obs : float
        Observed value of Z for the upper-tail probability.
    k : int
        Number of Normals in the max (Y = max of k).
    m : int
        Number of iid copies Y_i being summed.
    grid_points : int, default 1000
        Number of points in the base grid used to tabulate the density of Y.
        Increase for more accuracy in extreme tails.
    tail : float, default 1e-12
        Tail mass used to truncate the base grid for Y via quantiles.
        Decrease (e.g., 1e-15 or 1e-18) for more extreme tails.
    return_debug : bool, default False
        If True, also return a dict with the z-grid, pdf, and cdf arrays.

    Returns
    -------
    p : float
        Upper-tail probability P(Z >= z_obs).
    debug : dict (optional)
        {'z': grid, 'pdf': pZ, 'cdf': cdf} if return_debug=True.
    """
    # --- checks ---
    if not (isinstance(k, int) and k >= 1 and isinstance(m, int) and m >= 1):
        raise ValueError("Require integers k>=1 and m>=1.")
    if grid_points < 100:
        raise ValueError("grid_points should be reasonably large (>=100).")
    if not np.isfinite(z_obs):
        raise ValueError("z_obs must be finite.")

    # --- Grid for Y via quantiles of F_Y(y) = Phi(y)^k ---
    # Choose [a,b] so that F_Y(a)=tail and F_Y(b)=1-tail
    q_low = tail ** (1.0 / k)
    q_high = (1.0 - tail) ** (1.0 / k)
    a = norm.ppf(q_low)
    b = norm.ppf(q_high)
    y = np.linspace(a, b, grid_points)
    dy = y[1] - y[0]

    # --- pdf of Y: f_Y(y) = k * phi(y) * Phi(y)^(k-1) ---
    phi = norm.pdf(y)
    Phi = norm.cdf(y)
    pY = k * phi * np.power(Phi, k - 1)
    pY = np.maximum(pY, 0.0)
    # normalize (small numerical drift)
    pY /= np.trapz(pY, y)

    # --- m-fold convolution for Z (continuous conv via FFT with *dy) ---
    pZ = pY.copy()
    for _ in range(m - 1):
        pZ = fftconvolve(pZ, pY, mode="full") * dy
        pZ = np.maximum(pZ, 0.0)

    # Support for Z is [m*a, m*b] with uniform spacing dy
    z0 = m * a
    z1 = m * b
    z = np.linspace(z0, z1, pZ.size)

    # Final normalization guard
    area = np.trapz(pZ, z)
    if area <= 0 or not np.isfinite(area):
        raise RuntimeError("Failed to normalize resulting density.")
    pZ /= area

    # --- CDF via cumulative trapezoid (here simple Riemann sum is fine) ---
    cdf = np.cumsum(pZ) * (z[1] - z[0])
    # clip/anchor endpoints
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[0] = 0.0
    cdf[-1] = 1.0

    # --- p-value P(Z >= z_obs) ---
    if z_obs <= z[0]:
        p_val = 1.0
    elif z_obs >= z[-1]:
        p_val = 0.0
    else:
        F_at = np.interp(z_obs, z, cdf)
        p_val = max(0.0, 1.0 - F_at)

    if return_debug:
        return p_val, {"z": z, "pdf": pZ, "cdf": cdf}
    return p_val

def watermax_detect(text, config):
    # we will detect watermax simhash watermark by computing the simhash
    # 1. for each token i, define a seeded variable u_i ~ N(0,1) 
    #   where the seed is a function of thecurrent token and preceding h-1 tokens
    # 2. the global score is S = sum_i u_i, over subset I that excludes dups
    # 3. let M = |I|, then under H0, s ~ N(0,M) so we can compute p value as
    #   p = Phi( -S / sqrt(M) )
    # we'll obvs reject for some small p value threshold
    tokenizer = config['tokenizer']
    transformer_model = config['transformer_model']
    prior_tokens = config['prior_tokens']
    seed = config['seed']
    k = config['k']
    b = config['b']
    seed_function = config.get('seed_function', simhash_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ids = tokenizer.encode(text, return_tensors="pt").squeeze().to(device)

    S = 0.0
    M = 0
    # seen_signatures = set()

    # so now we iterate over tokens, for each position i, we build causal
    # prefix that incluses the current token and pass the seed function, 
    # (internally this truncates the last prior token) and we dedup on the
    # sig
    for i in range(len(ids)):
        # prefix ONLY for the key (exclude current token)
        prefix_ids = ids[:i]
        # handle i==0 (empty prefix) by falling back to just current token context
        if prefix_ids.numel() == 0:
            prefix_ids = ids[:1]

        # K_t = seed_function(prefix_ids, 
        #                     prior_tokens, 
        #                     tokenizer, 
        #                     transformer_model, 
        #                     hash_idx=0, 
        #                     seed=seed, 
        #                     k=k, 
        #                     b=b)

        K_list = []
        for j in range(k):
            K_tj = seed_function(prefix_ids,
                                 prior_tokens,
                                 tokenizer,
                                 transformer_model,
                                 hash_idx=j,
                                 seed=seed,
                                 k=k,
                                 b=b)
            K_list.append(K_tj)

        # if K_t in seen_signatures:
        #     continue
        # seen_signatures.add(K_t)

        # derive candidate-specific seed with the *observed* token id ids[i]
        # cand_seed = _hmac64(seed, K_t, int(ids[i].item()))
        # rng = np.random.default_rng(cand_seed)
        # u = rng.standard_normal()
        u_max = -np.inf
        tok_i = int(ids[i].item())
        for K_tj in K_list:
            cand_seed = _hmac64(seed, K_tj, tok_i)
            rng = np.random.default_rng(cand_seed)
            u = rng.standard_normal()
            if u > u_max:
                u_max = u

        S += float(u_max)
        M += 1

    if M <= 0:
        print("warning, M <= 0 in watermax detection, no independent tokens found, returning p=0.5")
        return 0.5
    
    z = S / M # NOTE: watermax actually uses S / sqrt(M), but we want per token average
    # p value is left tail
    # p_value = float(norm.cdf(-z))
    p_value = p_value_sum_of_max_normals(S, k, M)
    print(f"watermax simhash detect: S = {S}, M = {M}, z = {z}, p = {p_value}")
    return p_value