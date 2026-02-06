import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
try:
    from fastcoref import FCoref
except ImportError:
    FCoref = None

from huggingface_hub import login

# ------------------ Hugging Face Authentication ------------------ #
def hf_authenticate():
    """
    Authenticate with Hugging Face automatically using:
    1. Environment variable HUGGINGFACE_HUB_TOKEN
    2. Streamlit secrets
    """
    # Environment variable
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Streamlit secrets (if running in Streamlit Cloud)
    if not token:
        try:
            token = st.secrets["HUGGINGFACE_HUB_TOKEN"]
        except Exception:
            pass

    if token:
        try:
            login(token=token, add_to_git_credential=False)
            st.info(" Hugging Face authenticated successfully.")
        except Exception as e:
            st.warning(f" Failed to authenticate with provided token. Proceeding anonymously. Error: {e}")
    else:
        st.warning(" No Hugging Face token found. Only public models will work.")

# Authenticate immediately on import
hf_authenticate()

# ---------------- GENERIC HF PIPELINE LOADER ---------------- #
@st.cache_resource(show_spinner="Loading Hugging Face pipeline...")
def load_pipeline(task: str, model_name: str, **kwargs):
    """
    Load any Hugging Face pipeline with caching.
    """
    return pipeline(task=task, model=model_name, **kwargs)

# ---------------- BERT MASKED LM LOADER ---------------- #
@st.cache_resource(show_spinner="Loading BERT Masked LM model...")
def load_bert_mlm(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

# ---------------- GPT / CAUSAL LM LOADER ---------------- #
@st.cache_resource(show_spinner="Loading GPT model...")
def load_gpt(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPT2 has no pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, model

# ---------------- COREFERENCE LOADER ---------------- #
@st.cache_resource(show_spinner="Loading Coreference model...")
def load_coref(model_name="biu-nlp/f-coref"):
    if FCoref is None:
        raise ImportError("fastcoref is not installed. Install it via `pip install fastcoref`")
    return FCoref(model_name)

# ---------------- RELATION EXTRACTION LOADER ---------------- #
@st.cache_resource(show_spinner="Loading Relation Extraction model...")
def load_relation_extraction(model_name="Babelscape/rebel-large"):
    return load_pipeline(task="text-classification", model_name=model_name)

# ---------------- STYLE TRANSFER LOADER ---------------- #
@st.cache_resource(show_spinner="Loading Style Transfer model...")
def load_style_transfer(model_name="google/flan-t5-base", max_length=128):
    return load_pipeline(task="text2text-generation", model_name=model_name, max_length=max_length)

# ---------------- SUMMARIZATION LOADER ---------------- #
@st.cache_resource(show_spinner="Loading Summarization model...")
def load_summarization(model_name="facebook/bart-large-cnn", max_length=120, min_length=40):
    return load_pipeline(
        task="summarization",
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

# ---------------- TRANSLATION LOADER ---------------- #
@st.cache_resource(show_spinner="Loading Translation model...")
def load_translation(model_name="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"):
    return load_pipeline(
        task="translation",
        model_name=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

# ---------------- NER LOADER ---------------- #
@st.cache_resource(show_spinner="Loading NER model...")
def load_ner(model_name="dslim/bert-base-NER"):
    return load_pipeline(task="ner", model_name=model_name, aggregation_strategy="simple")

# ---------------- QUESTION ANSWERING LOADER ---------------- #
@st.cache_resource(show_spinner="Loading QA model...")
def load_qa(model_name="deepset/roberta-base-squad2"):
    return load_pipeline(task="question-answering", model_name=model_name)

# ---------------- SENTIMENT ANALYSIS LOADER ---------------- #
@st.cache_resource(show_spinner="Loading Sentiment Analysis model...")
def load_sentiment(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    return load_pipeline(task="sentiment-analysis", model_name=model_name)
