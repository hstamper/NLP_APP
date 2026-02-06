# ----------------------------------------------
# Configuration for all NLP tasks in the playground
# ----------------------------------------------

HF_MODELS = {
    "bert_mlm": {"model": "bert-base-uncased"},
    "gpt": {"model": "gpt2"},
    "sentiment": {"model": "distilbert-base-uncased-finetuned-sst-2-english"},
    "summarization": {"model": "facebook/bart-large-cnn"},
    "ner": {"model": "dbmdz/bert-large-cased-finetuned-conll03-english"},
    "qa": {"model": "distilbert-base-cased-distilled-squad"},
    "coref": {"model": "biu-nlp/f-coref"}, 
    "relation": {"model": "babelscape/rebel-large"},
    "style": {"model": "google/flan-t5-base"}, # Example model
    "translation": {"model": "t5-small"} }
