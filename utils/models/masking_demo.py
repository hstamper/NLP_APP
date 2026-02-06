import streamlit as st
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from utils.model_loader import hf_authenticate

# Ensure Hugging Face is authenticated
hf_authenticate()

# ---------------- TOKENIZER LOADER ---------------- #
@st.cache_resource(show_spinner="Loading tokenizer for masking demo...")
def load_tokenizer(model_name="bert-base-uncased"):
    """
    Load tokenizer for Masked LM demo.
    """
    return AutoTokenizer.from_pretrained(model_name)

# ---------------- MASKING DEMO FUNCTION ---------------- #
def masking_demo(text, model_name=None, max_length=12, mlm_probability=0.15):
    """
    Demonstrates masked language modeling preprocessing.
    Returns:
        input_ids: tokenized input with [MASK] applied
        labels: target labels for masked positions
    """
    if model_name is None:
        model_name = "bert-base-uncased" # Default behavior

    tokenizer = load_tokenizer(model_name=model_name)

    # Tokenize input
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Data collator for MLM
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    batch = collator([inputs])

    # Convert tensors to lists for Streamlit display
    return batch["input_ids"].tolist(), batch["labels"].tolist()
