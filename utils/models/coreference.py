from utils.model_loader import load_coref
from utils.model_config import HF_MODELS

def coreference_resolution(text, model_name=None):
    if model_name is None:
        model_name = HF_MODELS["coref"]["model"]

    model = load_coref(model_name=model_name)
    preds = model.predict(texts=[text])
    return preds[0].clusters
