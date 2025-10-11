import torch

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

class NoMarkProcessor(torch.nn.Module):
    def __init__(self, gen_config):
        super().__init__()
    
    def forward(self, input_ids, logits):
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            probs = logits[b].softmax(dim=-1)
            next_token = top_p_sampling(probs, 0.9)
            logits[b,:] = 1e-5
            logits[b,next_token] = 1e5
        return logits