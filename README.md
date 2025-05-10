
# Cross-Industry AI Text Classification Engine for Real-Time Sentiment & Intent Detection

## Project Overview

This project focuses on building a Universal AI-Powered Text Classification Engine that uses machine learning and NLP to classify unstructured text—such as customer support tickets, reviews, clinical notes, and financial headlines—into sentiment, urgency, or intent. The system is modular, enabling rapid customization for various industry use cases.

---

## Goal

To develop reusable ML tools that accurately classify text for real-time decision-making and automation across platforms using models like RoBERTa and Logistic Regression.

---

## Intended Audience

- AI/ML Developers and NLP Engineers  
- Strategy & Ops Teams (Healthcare, Finance, Retail)  
- SaaS Product Managers  
- Customer Support Automation Leads  

---

## Strategy & Pipeline

### I. Preprocessing
- Tokenization, Lemmatization, Stopword Removal using spaCy  
- Slang normalization for Twitter data  

### II. Embedding & Feature Extraction
- TF-IDF for baseline  
- Word2Vec for semantic context  
- RoBERTa for transformer embeddings  

### III. Modeling & Training
- Scikit-learn: Logistic Regression, Random Forest  
- HuggingFace: DistilBERT, RoBERTa  
- Evaluation: Accuracy, F1-score, ROC-AUC  

### IV. Deployment Options
- Flask API for integration  
- Streamlit dashboard for demo  
- Docker for containerized deployment  

---
ai-text-classification/
├── app/
│   ├── flask_app.py
│   └── streamlit_app.py
├── data/
│   └── sample.csv
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── model.onnx
├── notebooks/
│   └── model_training.ipynb
├── utils/
│   └── preprocessing.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore

##  Challenges

- Adapting to diverse domains  
- Label harmonization across industries  
- Class imbalance in real-world datasets  
- Transformer speed optimization  

---

## Problem Statement

Can a flexible ML system accurately classify unstructured text across sectors—enhancing triage, reducing manual workload, and accelerating insights?

---

##  Dataset

- **Source**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)  
- **Attributes**: `tweet_text`, `company`, `intent`, `timestamp`  
- **Use Cases**: Telecom complaint triage, airline support prioritization, sentiment and topic tagging

---

## Implementation Overview

- Text preprocessing with spaCy  
- Intent and sentiment labeling  
- Baseline model: TF-IDF + Random Forest  
- Advanced model: RoBERTa fine-tuning  
- Deployed via Flask API and Streamlit  
- Dockerized for scalable deployment  

---

##  Results

- Logistic Regression (TF-IDF): F1 ≈ 0.83, ROC-AUC ≈ 0.81  
- RoBERTa Transformer: F1 ≈ 0.89, ROC-AUC ≈ 0.92  
- Streamlit interface for real-time predictions  
- Confusion matrix + ROC curves provided for transparency  

---

##  Conceptual Enhancement

**LangChain + RAG**

Enable enterprise chatbots to ask:
> “What are the top complaints this month by airline customers?”

Use Retrieval-Augmented Generation to pull answers from vector databases + real-time model predictions.

---

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [CardiffNLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  
- [Streamlit Docs](https://docs.streamlit.io/)  
- [Docker Deployment](https://docs.docker.com/)  
- [Kaggle Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)  

