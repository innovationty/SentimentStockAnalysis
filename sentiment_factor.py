import pandas as pd, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
df = pd.read_csv("stock_tweets.csv", usecols=["Date","Tweet","Stock Name"])

MODEL = "ProsusAI/finbert"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, torch_dtype=torch.float16 if device=="cuda" else torch.float32
        ).to(device).eval()

batch_size = 128          # Adjustable based on GPU memory
labels = {0: "positive", 1: "negative", 2: "neutral"}

def batch_tokenize(texts):
    return tokenizer(texts, truncation=True, max_length=128,
                     padding=True, return_tensors="pt")

sentiments = []
loader = DataLoader(df["Tweet"].tolist(), batch_size=batch_size, shuffle=False)

with torch.no_grad(): 
    for batch in tqdm(loader, desc="Predict", total=len(loader)):
        tok = batch_tokenize(batch).to(device)
        logits = model(**tok).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        sentiments.extend([labels[p] for p in preds])

df["sentiment"] = sentiments
df.to_csv("tweets_with_sentiment.csv", index=False)
# Added at the end of the script
sentiment_counts = df["sentiment"].value_counts()
print("\nSentiment distribution:\n", sentiment_counts)
print("\nSentiment proportions:\n", sentiment_counts / len(df))