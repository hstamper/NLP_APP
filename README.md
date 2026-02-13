# 🔬 Hilliard's Professional NLP Playground

**Interactive Natural Language Processing with State-of-the-Art Models**

A Streamlit-powered application for exploring and running NLP tasks in real time using Hugging Face Transformers. Select a task, provide input, and see results instantly — no coding required.

---

## Overview

This application provides a unified interface for running multiple NLP tasks through pre-trained models from the Hugging Face ecosystem. Each task features smart result rendering tailored to its output type, from entity tags and sentiment bars to formatted text and code displays.

---

## Supported Tasks

| Task | Description | Input Type |
|---|---|---|
| **Named Entity Recognition (NER)** | Extracts people, locations, organizations, and other entities from text | Text |
| **Sentiment Analysis** | Classifies text as positive, negative, or neutral with confidence scores | Text |
| **Question Answering** | Finds answers to questions given a context passage | Context + Question |
| **Translation** | Translates text between languages with configurable source/target | Text + Language Pair |
| **Text Summarization** | Condenses long text into concise summaries | Text |
| **Style Transfer** | Rewrites text in a specified style | Text + Target Style |
| **Masked Language Modeling** | Demonstrates token masking and prediction for MLM training | Text |
| **Coreference Resolution** | Resolves pronoun and entity references using FastCoref | Text |

Tasks are defined in a modular registry (`utils/task_registry.py`), making it straightforward to add new tasks without modifying the main application.

---

## Project Structure

```
NLP_Project/
├── app.py                       # Main Streamlit application
├── utils/
│   ├── model_loader.py          # Hugging Face authentication & model loading
│   └── task_registry.py         # Task definitions, default models & inputs
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A Hugging Face account and API token (for gated models)

### Installation

```bash
# Clone or navigate to the project
cd NLP_Project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Hugging Face Authentication

The application authenticates with Hugging Face to access models. Set your token as an environment variable:

```bash
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
```

Or add it to a `.env` file in the project root. You can generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Features

### Task Selection

Use the sidebar to select any available NLP task. Each task comes with a sensible default model and sample input so you can immediately see results.

### Custom Models

Toggle "Use custom model" in the sidebar to specify any compatible Hugging Face model ID. This lets you test your own fine-tuned models or explore alternatives from the Hugging Face Hub.

### Smart Result Rendering

Results are automatically rendered based on task type:

- **NER** — Color-coded entity tags grouped by type (Person, Location, Organization, Misc)
- **Sentiment** — Labeled confidence bars with color indicators
- **Question Answering** — Formatted answer extraction
- **Translation & Summarization** — Clean text output
- **Masked Language Modeling** — Code-formatted token IDs and labels

### Responsive Design

The interface features a warm cream and red theme with custom typography (DM Sans for UI, JetBrains Mono for code/data), gradient hero section, styled cards, and smooth hover animations.

---

## Adding New Tasks

New NLP tasks can be added by updating the task registry in `utils/task_registry.py`:

1. Define the task function that accepts input text and a `model_name` parameter
2. Add an entry to `TASK_REGISTRY` with the task configuration:

```python
"your_task_key": {
    "name": "Display Name",
    "function": your_task_function,
    "default_model": "huggingface/model-id",
    "default_input": "Sample input text",
    "input_type": "TEXT",  # TEXT, QA, TRANSLATION, STYLE, or TEXT_TO_TUPLE
    "button_label": "Run",
    "doc": "Output Label",
}
```

The main `app.py` will automatically pick up the new task in the sidebar dropdown and handle input/output rendering.

---

## Technology Stack

- **Frontend**: Streamlit with custom CSS theming
- **NLP Models**: Hugging Face Transformers
- **Coreference Resolution**: FastCoref
- **Deep Learning**: PyTorch
- **Typography**: DM Sans, JetBrains Mono (Google Fonts)

---

## License

This project is intended for educational and personal use.
