
# Cross-Lingual Sentiment Analysis Benchmark: BERT vs. XLM-RoBERTa

A comparative NLP benchmark evaluating cross-lingual transfer and multilingual sentiment classification using `bert-base-uncased` and `xlm-roberta-base`. The project covers dataset ingestion, transformer fine-tuning, multilingual evaluation, ONNX model export, and an interactive Streamlit inference dashboard.

## Overview

This project investigates how monolingual and multilingual transformer architectures perform on three-class sentiment classification across English, Spanish, Hindi, and French.

The benchmark compares an English-trained BERT model against a multilingual XLM-RoBERTa model to analyze cross-lingual generalization, language-specific performance degradation, and inference efficiency.

### Key Features

- **Multilingual Data Ingestion:** Load and preprocess sentiment datasets using Hugging Face Datasets and MLCommons Croissant metadata.
- **Transformer Fine-Tuning:** Train BERT and XLM-RoBERTa for three-class sentiment classification.
- **Cross-Lingual Evaluation:** Measure zero-shot transfer performance across four languages.
- **Evaluation Metrics:** Track accuracy and macro-averaged F1-score for imbalanced multiclass classification.
- **Dynamic Batch Padding:** Use `DataCollatorWithPadding` to reduce unnecessary padding during training and inference.
- **ONNX Export:** Export trained models with dynamic batch and sequence dimensions for flexible inference.
- **Interactive Dashboard:** Deploy a Streamlit application for sentiment predictions and inference latency measurements.

## Models

| Model | Architecture | Training Setup | Objective |
|---|---|---|---|
| BERT | `bert-base-uncased` | Fine-tuned on English | Evaluate zero-shot cross-lingual transfer |
| XLM-RoBERTa | `xlm-roberta-base` | Fine-tuned on multilingual data | Evaluate multilingual sentiment classification |

### Why Compare BERT and XLM-RoBERTa?

**BERT** uses an English-centric vocabulary and pretrained representations. When fine-tuned only on English data, its performance can degrade substantially on other languages, especially those using different scripts.

**XLM-RoBERTa (XLM-R)** is pretrained on multilingual text using a shared subword vocabulary. Its multilingual representations enable stronger cross-lingual transfer, although performance can still vary by language and training-data coverage.

## Dataset

- **Dataset:** [`cardiffnlp/tweet_sentiment_multilingual`](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)
- **Data ingestion:** MLCommons Croissant metadata using `mlcroissant`
- **Languages:** English (`en`), Spanish (`es`), Hindi (`hi`), and French (`fr`)
- **Task:** Three-class sentiment classification
- **Labels:** Negative, Neutral, Positive

The dataset contains multilingual tweets, making it useful for evaluating sentiment classification under informal language, abbreviations, and language-specific expressions.

## Evaluation Results

The following table reports the empirical benchmark results obtained in the project.

| Model | Language | Accuracy | Macro F1 |
|---|---|---:|---:|
| BERT (English-trained) | English | 0.575 | 0.518 |
| BERT (English-trained) | Spanish | 0.370 | 0.308 |
| BERT (English-trained) | Hindi | 0.295 | 0.266 |
| BERT (English-trained) | French | 0.415 | 0.397 |
| XLM-R (Multilingual-trained) | English | 0.650 | 0.647 |
| XLM-R (Multilingual-trained) | Spanish | 0.615 | 0.605 |
| XLM-R (Multilingual-trained) | Hindi | 0.410 | 0.399 |
| XLM-R (Multilingual-trained) | French | 0.670 | 0.668 |

*Note: These are the reported results from the benchmark run, not independently reproduced measurements.*

### Key Findings

**1. Multilingual training improves cross-lingual performance**

XLM-RoBERTa achieves higher accuracy and macro F1 across all four evaluated languages, demonstrating the benefit of multilingual pretrained representations and multilingual fine-tuning.

**2. English-only BERT struggles with cross-lingual transfer**

BERT's macro F1 drops from `0.518` on English to `0.308` on Spanish, `0.266` on Hindi, and `0.397` on French.

Hindi exhibits the largest degradation, consistent with the challenges of transferring English-only representations to a language written in a different script.

**3. Performance varies by language**

XLM-R achieves its strongest reported result on French, with a macro F1 of `0.668`, while Hindi remains the most challenging language at `0.399`.

**4. Macro F1 complements accuracy**

Accuracy measures the overall proportion of correct predictions, whereas macro F1 gives equal weight to each sentiment class. Reporting both helps reveal class-specific weaknesses that overall accuracy alone may obscure.

### Macro F1 Comparison

| Language | BERT | XLM-R | Absolute Improvement |
|---|---:|---:|---:|
| English | 0.518 | 0.647 | +0.129 |
| Spanish | 0.308 | 0.605 | +0.297 |
| Hindi | 0.266 | 0.399 | +0.133 |
| French | 0.397 | 0.668 | +0.271 |

XLM-R improves macro F1 by **12.9 to 29.7 percentage points**, depending on the language.

## Technical Workflow

The project follows an end-to-end NLP pipeline:

1. **Data ingestion:** Load multilingual tweet sentiment data and identify the relevant language splits.
2. **Preprocessing:** Normalize the dataset structure, map sentiment labels, and prepare model inputs.
3. **Tokenization:** Apply the corresponding tokenizer for each pretrained model.
4. **Fine-tuning:** Train sequence-classification models using Hugging Face `Trainer`.
5. **Evaluation:** Compute accuracy and macro F1 for each target language.
6. **Model export:** Convert the fine-tuned models to ONNX with dynamic batch and sequence dimensions.
7. **Inference:** Run predictions through ONNX Runtime.
8. **Deployment:** Provide an interactive Streamlit interface for sentiment classification and latency measurement.

## ONNX Optimization

The fine-tuned models are exported to ONNX format to support inference outside the standard PyTorch execution pipeline.

### Export Configuration

- **Export framework:** PyTorch ONNX exporter
- **ONNX opset:** 12
- **Dynamic axes:** Batch size and sequence length
- **Runtime:** ONNX Runtime
- **Exported models:** `bert_mono.onnx` and `xlm_multi.onnx`

Dynamic axes allow the exported graphs to accept variable batch sizes and tokenized sequence lengths.

### Inference Performance

The Streamlit dashboard measures inference latency per request and per token, enabling interactive performance inspection.

The project targets sub-15 ms per-token inference latency. This should be treated as a target rather than a guaranteed benchmark: actual latency depends on the hardware, model, sequence length, batch size, and runtime configuration.

## Streamlit Application

The application provides an interactive interface for testing sentiment classification with the exported models.

### Capabilities

- Enter text for sentiment prediction.
- Select a model for inference.
- Inspect the predicted sentiment class.
- Measure inference latency per request.
- Estimate latency per processed token.

The application is launched locally using Streamlit.

## Project Structure

```text
.
├── nlp_advanced_bootcamp_project.ipynb
│   └── Dataset ingestion, preprocessing, fine-tuning,
│       evaluation, and ONNX export
├── app.py
│   └── Streamlit sentiment analysis dashboard
├── bert_mono.onnx
│   └── Exported English-trained BERT model
├── xlm_multi.onnx
│   └── Exported multilingual XLM-RoBERTa model
└── README.md
    └── Project documentation
```

The ONNX files are generated artifacts. If they are not committed to the repository, generate them by running the notebook.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your actual GitHub repository URL and directory name.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install "transformers[torch]" datasets evaluate scikit-learn mlcroissant onnx onnxruntime streamlit matplotlib pandas numpy
```

For Jupyter Notebook execution, install Jupyter if it is not already available:

```bash
pip install notebook
```

### 4. Run the Training and Export Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open `nlp_advanced_bootcamp_project.ipynb` and execute the cells to load the dataset, fine-tune the models, evaluate the results, and export the ONNX artifacts.

Training may require substantial compute and memory resources, depending on the chosen batch size, sequence length, and dataset size.

### 5. Launch the Streamlit Dashboard

Once the ONNX model files have been generated:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in a browser to interact with the sentiment analysis application.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming language | Python |
| NLP and transformers | Hugging Face Transformers |
| Dataset processing | Hugging Face Datasets, MLCommons Croissant, pandas, NumPy |
| Model training | PyTorch, Hugging Face Trainer |
| Evaluation | scikit-learn, Evaluate |
| Model optimization | ONNX, ONNX Runtime |
| Interactive deployment | Streamlit |
| Visualization | Matplotlib |

## Limitations and Future Improvements

- **Language coverage:** Expand evaluation to additional languages and scripts.
- **Reproducibility:** Record random seeds, model checkpoints, dataset splits, and training configurations.
- **Fair comparisons:** Ensure consistent evaluation splits, label mappings, and comparable fine-tuning conditions.
- **Class-level analysis:** Add confusion matrices, per-class precision and recall, and language-wise error analysis.
- **Robustness:** Evaluate slang, code-switching, transliteration, sarcasm, and domain shifts.
- **Inference optimization:** Benchmark PyTorch and ONNX Runtime under identical hardware and batch-size conditions.
- **Deployment reliability:** Add dependency version pinning, model-file validation, and automated tests.

## Conclusion

This project benchmarks monolingual and multilingual transformer models for cross-lingual sentiment analysis across English, Spanish, Hindi, and French.

The reported results show that multilingual XLM-RoBERTa outperforms English-trained BERT across all four languages. The ONNX export and Streamlit deployment extend the work beyond offline evaluation into an interactive inference workflow.

The project demonstrates practical experience with transformer fine-tuning, multilingual NLP evaluation, model export, inference optimization, and application deployment.

---

**Core takeaway:** Multilingual pretrained representations provide substantially stronger cross-lingual sentiment classification performance than an English-only model in this benchmark, particularly for Spanish and French.
