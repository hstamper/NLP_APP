import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch
from fastcoref import FCoref
import streamlit as st
st.set_page_config(page_title="🧠 NLP Playground", layout="centered")
from utils.model_loader import hf_authenticate
from utils.task_registry import TASK_REGISTRY

# Ensure Hugging Face authentication
hf_authenticate()
# Set torch threads
torch.set_num_threads(1)

# Streamlit page config
# st.set_page_config(page_title="🧠 NLP Playground", layout="centered")
st.title("🧠 NLP Playground")
st.write("Select an NLP task, enter text, and view the output.")

# ---------------- Sidebar ---------------- #
# Create a mapping from display name back to task key
display_name_to_key = {cfg["name"]: key for key, cfg in TASK_REGISTRY.items()}
selected_task_name = st.sidebar.selectbox(
    "Select NLP Task",
    list(display_name_to_key.keys())
)

task_key = display_name_to_key[selected_task_name]
task_config = TASK_REGISTRY[task_key]

# ---------------- Model Customization ---------------- #
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Settings")
use_custom_model = st.sidebar.checkbox("Use Custom Model")
if use_custom_model:
    model_name = st.sidebar.text_input("Hugging Face Model ID", task_config["default_model"])
else:
    model_name = task_config["default_model"]

st.sidebar.info(f"Using model: `{model_name}`")

# ---------------- TASK EXECUTION ---------------- #

input_type = task_config["input_type"]
func = task_config["function"]
default_input = task_config["default_input"]
button_label = task_config.get("button_label", "Run")
doc_label = task_config.get("doc", "Output:")

if input_type == "TEXT":
    text = st.text_area("Input Text", default_input) if len(default_input) > 50 else st.text_input("Input Text", default_input)
    if st.button(button_label):
        result = func(text, model_name=model_name)
        st.write(doc_label)
        if isinstance(result, list):
            for item in result:
                st.write(item)
        elif isinstance(result, dict):
            st.write(result)
        else:
            st.write(result)

elif input_type == "QA":
    context_default = default_input["context"]
    question_default = default_input["question"]
    context = st.text_area("Context", context_default)
    question = st.text_input("Question", question_default)
    if st.button(button_label):
        st.write(func(question, context, model_name=model_name))

elif input_type == "TRANSLATION":
    text = st.text_area("Input Text", default_input)
    # Use params from registry as defaults, but allow user overrides if needed
    src_lang = st.text_input("Source Language", task_config["params"]["src_lang"])
    tgt_lang = st.text_input("Target Language", task_config["params"]["tgt_lang"])
    if st.button(button_label):
        st.write(func(text, src_lang=src_lang, tgt_lang=tgt_lang, model_name=model_name))

elif input_type == "STYLE":
    text = st.text_area("Input Text", default_input)
    style_default = task_config["params"]["style"]
    style_type = st.text_input("Style", style_default)
    if st.button(button_label):
        st.write(func(text, style=style_type, model_name=model_name))

elif input_type == "TEXT_TO_TUPLE":
    # Specifically for masking demo which returns (input_ids, labels)
    text = st.text_area("Input Text", default_input)
    if st.button(button_label):
        # Unpack tuple
        res1, res2 = func(text, model_name=model_name)
        st.subheader("Masked Input IDs")
        st.code(res1)
        st.subheader("Labels")
        st.code(res2)
