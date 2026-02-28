import streamlit as st
import joblib

# Load trained model and label encoder
model = joblib.load("spam_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classification",
    layout="centered"
)

# Title and description
st.title("SMS Spam Classification System")
st.write(
    "This application uses an ensemble learning model to classify "
    "SMS messages as Spam or Ham."
)

# Input text box
message = st.text_area("Enter SMS message")

# Predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter an SMS message.")
    else:
        # Make prediction
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0][1]

        # Decode label
        label = label_encoder.inverse_transform([prediction])[0]

        # Display result
        st.subheader("Prediction Result")
        st.write(f"Prediction: **{label.upper()}**")
        st.write(f"Spam Probability: **{probability:.2f}**")

        if label == "spam":
            st.error("The message is classified as SPAM.")
        else:
            st.success("The message is classified as HAM.")