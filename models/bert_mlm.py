import torch
from utils.model_loader import load_bert_mlm
from utils.model_config import HF_MODELS

def bert_masked_prediction(text, model_name=None, top_k=5):
    if model_name is None:
        model_name = HF_MODELS["bert_mlm"]["model"]
        
    tokenizer, model = load_bert_mlm(model_name)
    inputs = tokenizer(text, return_tensors="pt")

    mask_positions = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)

    if len(mask_positions[0]) == 0:
        return ["⚠️ No [MASK] token found"]

    with torch.no_grad():
        outputs = model(**inputs)

    mask_index = mask_positions[1]
    logits = outputs.logits[0, mask_index, :]
    top_tokens = torch.topk(logits, top_k, dim=1).indices[0]

    return [tokenizer.decode([token]).strip() for token in top_tokens]
