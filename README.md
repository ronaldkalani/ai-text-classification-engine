# Cross-Industry AI Text Classification Engine for Real-Time Sentiment & Intent Detection

This project implements a universal AI-powered NLP system that classifies unstructured text from diverse domains—such as customer support tweets, product reviews, clinical notes, and financial headlines—into actionable categories (sentiment, urgency, topic).

Designed to enhance user experiences across multiple devices and platforms, the system leverages Scikit-learn and Hugging Face Transformers (RoBERTa, DistilBERT) for high-accuracy classification. The solution includes preprocessing with spaCy, embedding techniques (TF-IDF, Word2Vec, BERT), multiple model options, and end-to-end deployment via Flask, Streamlit, and Docker.

##  Features

- Text preprocessing (tokenization, lemmatization, stopword & slang removal)
- Embedding using TF-IDF, Word2Vec, and RoBERTa
- ML models: Logistic Regression, Random Forest
- Transformer models: DistilBERT, RoBERTa
- Model evaluation: Accuracy, F1 Score, ROC-AUC
- Deployment via Flask API & Streamlit dashboard
- Dockerized application with ONNX & Pickle model serialization

## Folder Structure

```
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
```

##  Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ai-text-classification.git
cd ai-text-classification

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

##  Run the App

### Streamlit

```bash
streamlit run app/streamlit_app.py
```

### Flask

```bash
python app/flask_app.py
```

##  Model Training

Refer to `notebooks/model_training.ipynb` for:
- Data loading and preprocessing
- Embedding (TF-IDF, Word2Vec, RoBERTa)
- Model training and evaluation
- Saving models

##  Dataset

**Kaggle Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

Includes tweets labeled by company, sentiment, and topic.

##  Technologies Used

- Python, pandas, numpy, scikit-learn
- spaCy, gensim, transformers (Hugging Face)
- Flask, Streamlit, Docker
- Pickle, ONNX, Torch

##  Future Enhancements

- Add LangChain & RAG for question answering
- Enable model retraining on live data
- Integrate with Slack or Zendesk APIs for real-time classification

##  References

- [Transformers – Hugging Face](https://huggingface.co/transformers/)
- [CardiffNLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [Kaggle Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker](https://docs.docker.com/)
