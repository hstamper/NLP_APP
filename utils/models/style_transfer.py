from utils.model_loader import load_style_transfer
from utils.model_config import HF_MODELS

def style_transfer(text, style="formal", model_name=None):
    if model_name is None:
        model_name = HF_MODELS["style"]["model"]
        
    pipe = load_style_transfer(model_name=model_name)
    prompt = f"Transfer this text to {style} style: {text}"
    return pipe(prompt)[0]["generated_text"]
