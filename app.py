import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import top_k_accuracy_score
import nltk
nltk.download('punkt')

st.set_page_config(page_title="WordWave - Next Word Prediction", page_icon="🧠") # First command

# Load model and tokenizer
model = load_model("word-wave.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1]
vocab_size = len(tokenizer.word_index) + 1

# =======================
# ⚙️ Evaluation Metrics
# =======================

@st.cache_data(show_spinner=True)
def evaluate_model_metrics(model, _tokenizer, sample_size=5000):
    X_samples = []
    y_true = []

    word_index = _tokenizer.word_index
    index_word = _tokenizer.index_word

    for word, idx in list(word_index.items())[:sample_size]:
        seed = word
        tokenized = _tokenizer.texts_to_sequences([seed])[0]
        if len(tokenized) < 2:
            continue
        for i in range(1, len(tokenized)):
            seq = tokenized[:i+1]
            padded = pad_sequences([seq[:-1]], maxlen=max_len, padding='pre')
            if padded.shape[1] == max_len:
                X_samples.append(padded[0])
                y_true.append(seq[-1])
            if len(X_samples) >= sample_size:
                break
        if len(X_samples) >= sample_size:
            break

    if not X_samples:
        return 0.0, float('inf')  # return dummy metrics if data is empty

    X_eval = np.array(X_samples)
    y_true = np.array(y_true)

    y_pred_probs = model.predict(X_eval, verbose=0)

    # Top-5 accuracy
    top_5 = top_k_accuracy_score(y_true, y_pred_probs, k=5)

    # Perplexity
    log_probs = np.log(np.take_along_axis(y_pred_probs, y_true[:, None], axis=1).flatten() + 1e-10)
    perplexity = np.exp(-np.mean(log_probs))

    return top_5, perplexity


# Precompute evaluation
with st.spinner("Evaluating model metrics..."):
    top_5_acc, perplexity_score = evaluate_model_metrics(model, tokenizer)

# =======================
# 🧠 Inference Functions
# =======================

def beam_search_decoder(seed_text, beam_width=5, next_words=10, max_len=50):
    sequences = [(seed_text, 0.0)]
    for _ in range(next_words):
        all_candidates = []
        for seq, score in sequences:
            tokenized = tokenizer.texts_to_sequences([seq])[0]
            tokenized = pad_sequences([tokenized], maxlen=max_len, padding='pre')
            preds = model.predict(tokenized, verbose=0)[0]
            top_indices = np.argsort(preds)[-beam_width:]
            for idx in top_indices:
                word = tokenizer.index_word.get(idx, '')
                if not word: continue
                candidate = (seq + ' ' + word, score - np.log(preds[idx] + 1e-10))
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
    return sequences[0][0]

def evaluate_bleu(reference_sentence, generated_sentence):
    reference = [reference_sentence.split()]
    candidate = generated_sentence.split()
    return sentence_bleu(reference, candidate, weights=(0.5, 0.5))

# =======================
# 🎛️ Streamlit UI
# =======================

st.title("🧠 WordWave: Next Word Prediction")
st.write("Generate coherent text using a trained BiLSTM + Attention model.")

# Sidebar Metrics
st.sidebar.header("📊 Model Evaluation")
st.sidebar.metric("🎯 Top-5 Accuracy", f"{top_5_acc * 100:.2f}%")
st.sidebar.metric("📉 Perplexity", f"{perplexity_score:.2f}")
st.sidebar.caption("Evaluated on a subset of 5,000 examples")

# Input
seed_text = st.text_input("Enter your seed text", value="Artificial intelligence is")
next_words = st.slider("How many words to generate?", min_value=1, max_value=20, value=10)

# Generate
if st.button("Generate"):
    generated = beam_search_decoder(seed_text, beam_width=5, next_words=next_words, max_len=max_len)
    st.markdown("### 📝 Generated Text")
    st.success(generated)

    reference = st.text_input("Optional: Enter reference sentence to compute BLEU score")
    if reference:
        bleu = evaluate_bleu(reference, generated)
        st.metric("🔵 BLEU Score", f"{bleu:.4f}")
    else:
        st.info("BLEU Score skipped (no reference provided)")
