from utils.model_loader import load_qa
from utils.model_config import HF_MODELS

def question_answering(question, context, model_name=None):
    if model_name is None:
        model_name = HF_MODELS["qa"]["model"]

    qa_pipe = load_qa(model_name=model_name)

    result = qa_pipe(question=question, context=context)
    return result("answer")
