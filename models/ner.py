from utils.model_loader import load_ner
from utils.model_config import HF_MODELS

def ner_extraction(text, model_name=None):
    if model_name is None:
        model_name = HF_MODELS["ner"]["model"]

    ner_pipe = load_ner(model_name=model_name)
    return ner_pipe(text)
