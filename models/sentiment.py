from utils.model_loader import load_sentiment
from utils.model_config import HF_MODELS

def sentiment_analysis(text, model_name=None):
    if model_name is None:
        model_name = HF_MODELS["sentiment"]["model"]
        
    sentiment_pipe = load_sentiment(model_name=model_name)
    return sentiment_pipe(text)

