import torch
import numpy as np
import hashlib

def to_bytes(x):
    if isinstance(x, int):
        return x.to_bytes(8, byteorder="big", signed=False)
    elif isinstance(x, np.integer):
        return int(x).to_bytes(8, byteorder="big", signed=False)
    elif isinstance(x, torch.Tensor) and x.dim() == 0:
        return int(x.item()).to_bytes(8, byteorder="big", signed=False)
    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return b"".join(to_bytes(int(i)) for i in x)
    else:
        raise TypeError(f"Unsupported type for to_bytes: {type(x)}")

def simhash_seed(input_ids, prior_tokens, tokenizer, transformer_model, hash_idx, seed, k, b):
    # print(prior_tokens, hash_idx, seed, k, b)
    with torch.no_grad():  
        input_text = tokenizer.decode(input_ids[-prior_tokens:], skip_special_tokens=True)
    vec = transformer_model.encode(input_text)
    input_vector = vec / (np.linalg.norm(vec) + 1e-12)
    # print(f"input_text: {input_text}")
    # print(f"input_vector: {input_vector[:4]}")
    # print("vector checksum:", np.sum(input_vector), np.mean(input_vector))
    # Gaussian hyperplanes
    rng = np.random.default_rng(hash_idx + k * seed)
    embed_dim = input_vector.shape[0] 
    random_vectors = rng.standard_normal((b, embed_dim))
    # print(f"random_vectors: {random_vectors[0][:4]}")

    # Projections
    projections = random_vectors @ input_vector
    # Small margin so near-zero projections don't flip bits easily
    tau = 0.02  # start here; 0.01–0.05 are typical
    # Ternary code in {-1, 0, +1}
    ternary = (projections >  tau).astype(np.int8) - (projections < -tau).astype(np.int8)
    # Map {-1,0,+1} -> {0,1,2} so it has a stable byte representation
    ternary_bytes = (ternary + 1).astype(np.uint8).tobytes()

    # Stable, explicit payload for hashing (keeps determinism)
    payload = f"{hash_idx}|{k}|{seed}|tau={tau}|".encode("utf-8") + ternary_bytes
    simhash_seed = int(hashlib.sha256(payload).hexdigest(), 16)

    return simhash_seed & 0xFFFFFFFF

def normal_seed(input_ids, prior_tokens, tokenizer, transformer_model, hash_idx, seed, k, b):
    prior_ids = input_ids[-prior_tokens:].sum().item()
    normal_seed = int(hashlib.sha256(to_bytes(hash_idx) + to_bytes(seed) + to_bytes(prior_ids)).hexdigest(), 16)
    return normal_seed & 0xFFFFFFFF

def no_hash_seed(input_ids, prior_tokens, tokenizer, transformer_model, hash_idx, seed, k, b):
    no_hash_seed = int(hashlib.sha256(to_bytes(seed)).hexdigest(), 16)
    return no_hash_seed & 0xFFFFFFFF