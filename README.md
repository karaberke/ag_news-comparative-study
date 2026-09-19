# AG News Comparative Study

A comparison of five text classifiers on the [AG News](https://huggingface.co/datasets/ag_news) topic-classification dataset (World, Sports, Business, Sci/Tech), from classical baselines to a fine-tuned transformer and a zero-shot model.

Team project for CS 371N (Natural Language Processing) at UT Austin, by Berke Kara and Anh Nguyen.

## Models and results

All models use the same 30,000-article sample, split 70/15/15 into train, validation, and test.

| # | Model | Test accuracy | Test macro-F1 |
|---|---|---|---|
| 1 | TF-IDF + logistic regression | 90.36% | 0.9032 |
| 2 | Pretrained Word2Vec + MLP classifier | 88.53% | 0.8852 |
| 3 | Two-layer bidirectional LSTM, trained from scratch | 89.24% | 0.8921 |
| 4 | DistilBERT (`distilbert-base-uncased`), fine-tuned | **92.96%** | **0.9296** |
| 5 | Zero-shot BART (`facebook/bart-large-mnli`), no task training | 67.64% | 0.6433 |

## Analysis

- **Interpretability:** Integrated Gradients (Captum) attributions for the DistilBERT model show which words push each prediction toward or away from a class.
- **Error analysis:** the best model's remaining errors concentrate in Business vs. Sci/Tech, where keyword overlap drives confusion.

## Running it

Open `ag_news_comparative_study.ipynb` in Jupyter or Google Colab and run the cells in order. A GPU is recommended for models 3 to 5.

Stack: Python, PyTorch, Hugging Face Transformers, scikit-learn, Captum.
