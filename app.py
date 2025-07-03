import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

nltk.download('punkt')

# Load and preprocess text
@st.cache_data
def load_data():
    with open('sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as f:
        text = f.read().lower()
    tokens = word_tokenize(text)
    word_counts = Counter(tokens)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    sequence_length = 4
    data = []
    for i in range(len(tokens) - sequence_length):
        input_seq = tokens[i:i + sequence_length - 1]
        target = tokens[i + sequence_length - 1]
        data.append((input_seq, target))

    def encode(seq): return [word2idx[word] for word in seq]
    encoded_data = [(torch.tensor(encode(inp)), torch.tensor(word2idx[target]))
                    for inp, target in data]

    return tokens, word2idx, idx2word, encoded_data, len(vocab), sequence_length

tokens, word2idx, idx2word, encoded_data, vocab_size, sequence_length = load_data()

# Define the model
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(PredictiveKeyboard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

# Train model and cache
@st.cache_resource
def train_model():
    model = PredictiveKeyboard(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(5):  # Reduced for faster demo
        total_loss = 0
        random.shuffle(encoded_data)
        for input_seq, target in encoded_data[:10000]:
            input_seq = input_seq.unsqueeze(0)
            output = model(input_seq)
            loss = criterion(output, target.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

model = train_model()

# Suggest next word
def suggest_next_words(model, text_prompt, top_k=3):
    model.eval()
    tokens = word_tokenize(text_prompt.lower())
    if len(tokens) < sequence_length - 1:
        return ["Please enter at least 3 words."]
    input_seq = tokens[-(sequence_length - 1):]
    try:
        input_tensor = torch.tensor([word2idx[word] for word in input_seq]).unsqueeze(0)
    except KeyError:
        return ["Unknown word(s) in input."]
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, top_k).indices.tolist()
    return [idx2word[idx] for idx in top_indices]

# Streamlit UI
st.title("🔮 Sherlock Holmes Predictive Keyboard")
st.markdown("Enter at least 3 words to get a prediction for the next likely word.")

user_input = st.text_input("Type your sentence:")

if st.button("Suggest Next Word"):
    if user_input.strip():
        suggestions = suggest_next_words(model, user_input)
        st.write("**Top Suggestions:**")
        for i, word in enumerate(suggestions, 1):
            st.write(f"{i}. {word}")
    else:
        st.warning("Please enter a sentence with at least 3 words.")

