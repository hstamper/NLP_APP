from utils.model_loader import load_summarization
from utils.model_config import HF_MODELS

def summarize_text(text, model_name=None):
    if model_name is None:
        model_name = HF_MODELS["summarization"]["model"]

    summarizer = load_summarization(model_name=model_name)
    return summarizer(text)[0]["summary_text"]

