# Sarcasm Explainer

This repository contains the Course Project for **Theory of Programming Languages / Natural Language Processing**. The project focuses on detecting and classifying sarcasm in Amazon product reviews using advanced preprocessing pipelines and state-of-the-art Transformer-based language models.

---

## 📖 Overview

Sarcasm detection is one of the most challenging tasks in sentiment analysis and natural language processing (NLP). Sarcastic utterances often express a positive literal sentiment while implying a negative context (or vice versa), which confuses traditional sentiment classifiers.

This project implements a complete machine learning pipeline:

1. **Advanced Text Preprocessing**: Customized cleaning specifically designed to retain sarcasm-indicative features.
2. **Group-Aware Splitting**: Resolving data leakage during cross-validation by keeping paired product reviews together.
3. **Fine-Tuning Transformer Models**: Benchmarking and evaluating four transformer architectures—**DistilBERT**, **DistilBERT-SST2**, **RoBERTa-base**, and **RoBERTa-large**—specifically fine-tuned to classify sarcastic versus regular reviews.

---

## 📊 Dataset Description

The project utilizes the **Sarcasm Amazon Reviews Corpus** (curated by Elena Filatova, LREC 2012), which is structured as follows:

- **Ironic & Regular Reviews**: Text reviews scraped from Amazon.com.
- **Review Pairings (`file_pairing.txt`)**: Contains a total of **923 entries**:
  - **330 Pairs**: Matches of ironic reviews and their regular counterparts written for the same Amazon product.
  - **106 Ironic Reviews**: Unpaired ironic reviews.
  - **486 Regular Reviews**: Unpaired regular reviews.
- **Sarcasm Lines (`sarcasm_lines.txt`)**: **437 text utterances** extracted from ironic reviews that MTurk annotators highlighted as containing the explicit sarcasm/irony.
- **Excel Metadata (`five_labels_plus_stars.xlsx`)**: Initial star ratings assigned by review authors, along with step-2 annotations and labels.

The corpus files are stored in `datasets/SarcasmAmazonReviewsCorpus/`.

---

## 🛠️ Data Preprocessing Pipeline

Implemented in [PreProcessing.ipynb](PreProcessing.ipynb), the preprocessing pipeline prepares raw review texts for transformer training without losing sarcasm indicators:

1. **HTML Decoding & Scraping**: Converts HTML entities (e.g., `&amp;` to `&`) and removes raw HTML tags using `BeautifulSoup`.
2. **Text Normalization**: Converts all text to lowercase and strips leading/trailing whitespaces.
3. **Content Filtering**: Removes URLs, email addresses, and username mentions using regular expressions.
4. **Contraction Expansion**: Normalizes English contractions (e.g., `can't` $\rightarrow$ `cannot`, `won't` $\rightarrow$ `will not`, `'ve` $\rightarrow$ `have`).
5. **Punctuation Normalization**: Filters out special characters while explicitly keeping punctuation marks crucial for sarcasm detection (`?`, `!`, `.`, `,`, `'`). Repeating punctuation (e.g., `!!!`, `???`) is compressed to a single mark.
6. **Elongated Word Normalization**: Normalizes character repetitions (e.g., `soooo` $\rightarrow$ `soo`) to capture expressive emphasis without bloating the vocabulary.
7. **Empty Sample Handling**: Identifies empty strings and maps them to a placeholder `[empty_review]` rather than dropping samples.
8. **Group-Aware Data Splitting**: Sarcasm datasets containing paired reviews (ironic and regular reviews of the same product) suffer from data leakage if split randomly. To prevent this, the pipeline uses scikit-learn's `GroupShuffleSplit` grouping by product ID, ensuring that no product-level reviews bleed across the training, validation, and testing sets.
   - **Preprocessing Split Ratio**: 70% Train / 15% Validation / 15% Test.
   - **Notebook Fine-Tuning Split Ratio**: 80% Train / 10% Validation / 10% Test (re-split from the generated `full_dataset.csv` for optimal training sizes).

---

## 🚀 Model Architectures & Configurations

Four deep learning models from the Hugging Face `transformers` library are evaluated:

| Feature / Model             | DistilBERT Base                         | DistilBERT SST-2                                  | RoBERTa Base                            | RoBERTa Large                            |
| :-------------------------- | :-------------------------------------- | :------------------------------------------------ | :-------------------------------------- | :--------------------------------------- |
| **Base Model checkpoint**   | `distilbert-base-uncased`               | `distilbert-base-uncased-finetuned-sst-2-english` | `roberta-base`                          | `roberta-large`                          |
| **Pre-trained Domain**      | General English                         | Fine-tuned on Stanford Sentiment Treebank (SST-2) | General English (RoBERTa pre-training)  | General English (355M parameter variant) |
| **Parameters**              | ~66M                                    | ~66M                                              | ~125M                                   | ~355M                                    |
| **Epochs**                  | 2                                       | 2                                                 | 2                                       | 2                                        |
| **Learning Rate**           | `1.5e-5`                                | `1.5e-5`                                          | `1.5e-5`                                | `2e-5`                                   |
| **Batch Size (Train/Eval)** | 8 / 8                                   | 8 / 8                                             | 16 / 16                                 | 4 / 4                                    |
| **Gradient Accumulation**   | 2 steps (Effective BS = 16)             | 2 steps (Effective BS = 16)                       | 4 steps (Effective BS = 64)             | 2 steps (Effective BS = 8)               |
| **Warmup Configuration**    | 10% of total steps (Cosine)             | 10% of total steps (Cosine)                       | Dynamic 15% of total steps (Cosine)     | 10% of total steps (Cosine)              |
| **Weight Decay**            | 0.01                                    | 0.01                                              | 0.01                                    | 0.01                                     |
| **Optimizer & Precision**   | Mixed Precision (fp16) if GPU available | Mixed Precision (fp16) if GPU available           | Mixed Precision (fp16) if GPU available | Single Precision (fp32)                  |
| **Early Stopping**          | Patience = 1 epoch                      | Patience = 1 epoch                                | Patience = 1 epoch                      | Patience = 1 epoch                       |

_Note: The choice of **DistilBERT SST-2** represents a transfer learning experiment where sentiment polarity signals are transferred to help sarcasm detection, leveraging the intuition that sarcasm represents a conflict of sentiment expressions._

---

## 📊 Evaluation Metrics

All notebooks compute evaluation statistics on validation and test sets during training:

- **Accuracy**: Overall classification accuracy.
- **Macro Precision**: Average precision across sarcastic and regular classes.
- **Macro Recall**: Average recall across sarcastic and regular classes.
- **Macro F1-Score**: The primary optimization metric (`metric_for_best_model="macro_f1"`) representing the harmonic mean of precision and recall.

---

## 📂 Repository Structure

```filepath
sarcasm-explainer/
├── datasets/
│   └── SarcasmAmazonReviewsCorpus/
│       ├── Ironic.rar                      # Compressed ironic product reviews
│       ├── Regular.rar                     # Compressed regular product reviews
│       ├── ReadMe                          # Original dataset release notes
│       ├── file_pairing.txt                # Lists review pairings (pairs, ironic, regular)
│       ├── five_labels_plus_stars.xlsx     # Excel sheet with stars and final labels
│       └── sarcasm_lines.txt               # Text utterances highlighting sarcasms
├── output/                                 # Target directory for training artifacts
├── .gitignore                              # Standard git exclusions
├── PreProcessing.ipynb                     # Data cleaning, pair matching, and splitting notebook
├── DistilBERT-base-uncased.ipynb           # DistilBERT training and validation script
├── DistilBERT-SST2.ipynb                   # Sentiment-pre-trained DistilBERT training script
├── RoBERTa-base.ipynb                      # RoBERTa-base training and validation script
├── RoBERTa-large.ipynb                     # RoBERTa-large training and validation script
└── README.md                               # Project documentation and report (this file)
```

---

## 💻 How to Run

### Step 1: Preprocessing

1. Open [PreProcessing.ipynb](PreProcessing.ipynb) in Google Colab or a local Jupyter environment.
2. Upload the files from `datasets/SarcasmAmazonReviewsCorpus/` when prompted.
3. Run all cells to extract, clean, feature-engineer, split, and generate the preprocessed CSV datasets:
   - `train.csv`
   - `validation.csv`
   - `test.csv`
   - `full_dataset.csv`

### Step 2: Training the Models

1. Choose one of the model notebooks:
   - [DistilBERT-base-uncased.ipynb](DistilBERT-base-uncased.ipynb)
   - [DistilBERT-SST2.ipynb](DistilBERT-SST2.ipynb)
   - [RoBERTa-base.ipynb](RoBERTa-base.ipynb)
   - [RoBERTa-large.ipynb](RoBERTa-large.ipynb)
2. Load the notebook in your execution environment (with GPU runtime recommended, e.g., Google Colab T4).
3. Upload the generated `full_dataset.csv` when prompted.
4. Execute the cells to tokenize the text datasets, initialize the models, run the fine-tuning process, evaluate on validation/test sets, and save weights.
