from utils.model_loader import load_translation
from utils.model_config import HF_MODELS

def translate_text(text, src_lang="eng_Latn", tgt_lang="fra_Latn", model_name=None):
    if model_name is None:
        model_name = HF_MODELS["translation"]["model"]

    translator = load_translation(model_name=model_name, src_lang=src_lang, tgt_lang=tgt_lang)
    return translator(text)[0]["translation_text"]
