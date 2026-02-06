# Registry for all NLP tasks to drive the UI dynamically
from models import (
    bert_mlm, gpt, ner, qa, sentiment, coreference, relation, style, summarization, translation, masking_demo
)

TASK_REGISTRY = {
    "bert_mlm": {
        "name": "Masked Word Prediction (BERT)",
        "module": bert_mlm,
        "function": bert_mlm.bert_masked_prediction,
        "input_type": "TEXT",
        "default_input": "I love playing [MASK] on weekends.",
        "default_model": "bert-base-uncased",
        "button_label": "Predict",
        "doc": "Top Predictions:"
    },
    "gpt": {
        "name": "True Next Word Prediction (GPT)",
        "module": gpt,
        "function": gpt.gpt_next_word,
        "input_type": "TEXT",
        "default_input": "I love playing",
        "default_model": "gpt2",
        "button_label": "Generate",
        "doc": "Top Next Word Predictions:"
    },
    "sentiment": {
        "name": "Sentiment Analysis",
        "module": sentiment,
        "function": sentiment.sentiment_analysis,
        "input_type": "TEXT",
        "default_input": "I really love this product!",
        "default_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "button_label": "Analyze",
        "doc": "Sentiment Analysis Reference:"
    },
    "ner": {
        "name": "Named Entity Recognition",
        "module": ner,
        "function": ner.ner_extraction,
        "input_type": "TEXT",
        "default_input": "Barack Obama was born in Hawaii.",
        "default_model": "dslim/bert-base-NER",
        "button_label": "Extract",
        "doc": "Extracted Entities:"
    },
    "qa": {
        "name": "Question Answering",
        "module": qa,
        "function": qa.question_answering,
        "input_type": "QA",
        "default_input": {
            "context": "BERT was introduced by Google in 2018.",
            "question": "Who introduced BERT?"
        },
        "default_model": "deepset/roberta-base-squad2",
        "button_label": "Answer",
        "doc": "Answer:"
    },
    "masking_demo": {
        "name": "Masking Demo (Training Concept)",
        "module": masking_demo,
        "function": masking_demo.masking_demo,
        "input_type": "TEXT_TO_TUPLE",
        "default_input": "Transformers are very powerful models",
        "default_model": "bert-base-uncased",
        "button_label": "Generate Masking Demo",
        "doc": "Masking Output"
    },
    "translation": {
        "name": "Multilingual Translation",
        "module": translation,
        "function": translation.translate_text,
        "input_type": "TRANSLATION",
        "default_input": "Hello world!",
        "default_model": "facebook/nllb-200-distilled-600M",
        "params": {
            "src_lang": "eng_Latn",
            "tgt_lang": "fra_Latn"
        },
        "button_label": "Translate",
        "doc": "Translation:"
    },
    "style": {
        "name": "Text Style Transfer",
        "module": style,
        "function": style.style_transfer,
        "input_type": "STYLE",
        "default_input": "I am happy with this service.",
        "default_model": "google/flan-t5-base",
        "params": {
            "style": "formal"
        },
        "button_label": "Transfer",
        "doc": "Styled Text:"
    },
    "coref": {
        "name": "Coreference Resolution",
        "module": corefeference,
        "function": corefeference.coreference_resolution,
        "input_type": "TEXT",
        "default_input": "Angela told Mary that she would help her.",
        "default_model": "biu-nlp/f-coref",
        "button_label": "Resolve",
        "doc": "Coreference Clusters:"
    },
    "relation": {
        "name": "Relation Extraction",
        "module": relation,
        "function": relation.relation_extraction,
        "input_type": "TEXT",
        "default_input": "Barack Obama was born in Hawaii.",
        "default_model": "Babelscape/rebel-large",
        "button_label": "Extract",
        "doc": "Extracted Relations:"
    },
    "summarization": {
        "name": "Text Summarization",
        "module": summarization,
        "function": summarization.summarize_text,
        "input_type": "TEXT",
        "default_input": "Transformers are large models that have revolutionized NLP...",
        "default_model": "facebook/bart-large-cnn",
        "button_label": "Summarize",
        "doc": "Summary:"
    }
}

