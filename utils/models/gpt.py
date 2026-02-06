import torch
from utils.model_loader import load_gpt
from utils.model_config import HF_MODELS

def gpt_next_word(text, model_name=None, top_k=5):
    """
    Returns top-k next-token predictions (true next token, not free generation)
    """
    if model_name is None:
        model_name = HF_MODELS["gpt"]["model"]

    tokenizer, model = load_gpt(model_name)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits for the last token
    next_token_logits = outputs.logits[:, -1, :]

    # Top-k token ids
    top_tokens = torch.topk(next_token_logits, top_k, dim=-1).indices[0]

    return [
        tokenizer.decode([token]).strip()
        for token in top_tokens
    ]
