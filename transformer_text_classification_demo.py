"""
Beginner-friendly Transformer project: Real-world text classification.

Use case:
- A tiny "customer feedback monitor" that classifies feedback as
  POSITIVE or NEGATIVE using a pre-trained Transformer model.

Why this is a good starter project:
1. It uses a real business scenario (feedback monitoring).
2. It avoids training from scratch (faster and simpler).
3. It still demonstrates practical Transformer usage.
4. It includes a "real-time" mode where new messages are classified
   instantly as you type them.

Requirements:
    pip install transformers torch

Run:
    python transformer_text_classification_demo.py
"""

# Import the Hugging Face pipeline helper.
# `pipeline` gives us an easy interface to use Transformer models
# without writing complex low-level model code.
from transformers import pipeline


# -------------------------------
# 1) Build the Transformer pipeline
# -------------------------------
# We select sentiment-analysis because it is one of the simplest
# and most useful text classification tasks.
#
# Model used:
# distilbert-base-uncased-finetuned-sst-2-english
# - DistilBERT = a smaller/faster version of BERT (good for beginners)
# - fine-tuned on SST-2 = ready to predict POSITIVE/NEGATIVE sentiment
#
# This shows practical use of Transformer architecture in production-like
# settings where we use a pre-trained model for immediate results.
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)


# -------------------------------
# 2) Utility function for predictions
# -------------------------------
def predict_sentiment(text: str) -> dict:
    """
    Classify one text message and return the model's prediction.

    Parameters
    ----------
    text : str
        The input text to classify.

    Returns
    -------
    dict
        Example output:
        {
            'label': 'POSITIVE',
            'score': 0.9998
        }

    Notes for beginners
    -------------------
    - The Transformer internally tokenizes text (turns words into tokens),
      processes them with self-attention, and produces a final prediction.
    - We don't manually code attention layers here; the pre-trained model
      already contains all Transformer components.
    """
    # `classifier` expects a string (or list of strings).
    # It returns a list of results, one result per input.
    result = classifier(text)[0]
    return result


# -------------------------------
# 3) Demo dataset (real-world style customer feedback)
# -------------------------------
sample_feedback = [
    "The delivery was super fast and the packaging was excellent!",
    "I am disappointed. The product stopped working in two days.",
    "Customer support solved my issue quickly. Great service.",
    "The app crashes every time I open it. Very frustrating.",
    "Decent quality for the price, but shipping was a bit late.",
]


# -------------------------------
# 4) Batch classification demo
# -------------------------------
print("\n=== Batch Demo: Customer Feedback Classification ===")
for i, message in enumerate(sample_feedback, start=1):
    prediction = predict_sentiment(message)

    # Convert score to percentage for easier reading.
    confidence_percent = prediction["score"] * 100

    print(f"\nFeedback #{i}")
    print(f"Text       : {message}")
    print(f"Prediction : {prediction['label']}")
    print(f"Confidence : {confidence_percent:.2f}%")


# -------------------------------
# 5) Real-time mode
# -------------------------------
print("\n=== Real-time Demo ===")
print("Type any sentence and press Enter to classify it instantly.")
print("Type 'exit' to stop.\n")

while True:
    user_text = input("You: ").strip()

    # Stop condition so user can end the program gracefully.
    if user_text.lower() == "exit":
        print("Program ended. Goodbye!")
        break

    # Ignore empty input so the model is called only with meaningful text.
    if not user_text:
        print("Please type a non-empty sentence.")
        continue

    prediction = predict_sentiment(user_text)
    confidence_percent = prediction["score"] * 100

    print(f"Model: {prediction['label']} ({confidence_percent:.2f}% confidence)\n")
