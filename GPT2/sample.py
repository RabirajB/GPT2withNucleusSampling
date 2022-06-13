'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT

    Changes made by Rabiraj Bandyopadhyay (RabirajB)
    1) Added the Nucleus Sampling to check the results compared to top_k sampling
    The original code can be found at "https://github.com/ari-holtzman/degen/blob/master/gen.py" based on the paper Holtzman et.al "The Curious Case of Neural Text Degenaration"
'''
import torch
import torch.nn.functional as F
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def top_p_logits(logits, top_p, filter_value = -float('Inf')):
    if top_p == 0:
        return logits
    
    #samp_probs = F.softmax(logits, dim = -1)
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cumulative_probs = torch.cumsum(sorted_logits, dim = -1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove) # Need to research this
    logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    #print(start_token)
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            #logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, top_p = 0.95)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output