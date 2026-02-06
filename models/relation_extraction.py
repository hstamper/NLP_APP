from utils.model_loader import load_relation_extraction
from utils.model_config import HF_MODELS

def relation_extraction(text, model_name=None):
    if model_name is None:
        model_name = HF_MODELS["relation"]["model"]
        
    pipe = load_relation_extraction(model_name=model_name)
    return pipe(text)
