import streamlit as st
import pickle
import re

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)            # remove URLs
    text = re.sub(r"@\w+", "", text)               # remove mentions
    text = re.sub(r"#\w+", "", text)               # remove hashtags
    text = re.sub(r"[^\w\s]", "", text)            # remove punctuation
    return text

# Streamlit UI
st.title("📊 Tweet Direction Classifier (Inbound vs Outbound)")
st.markdown("Classifies whether a tweet is **📥 inbound (from customer)** or **📤 outbound (from company)**")

user_input = st.text_area("✏️ Enter a tweet below:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])

        prediction = model.predict(features)[0]
        probas = model.predict_proba(features)[0]
        confidence = max(probas) * 100

        label = "📥 Inbound (from customer)" if prediction == 1 else "📤 Outbound (from company)"
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}%")
