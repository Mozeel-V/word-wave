
# WordWave – Next Word & Sequence Predictor

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3670A0?logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Model-TensorFlow%2FKeras-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Build-Active-blue)](https://github.com/Mozeel-V/word-wave)

WordWave is an intelligent next-word and short-sequence predictor built on a **Bidirectional LSTM** with **attention mechanism**, trained on a subset of the Wikipedia dataset. The app provides real-time word generation and metric-based evaluation, accessible via a user-friendly **Streamlit dashboard**.

---

## 🚀 Features

- Built using a deep **Embedding → BiLSTM → Attention → Dense** pipeline for next-word prediction
- Supports **beam search decoding** to improve generation quality over greedy search
- Evaluates with key metrics:  
  - ✅ **Top-5 Accuracy**
  - 📉 **Perplexity**
  - 🔵 **BLEU Score**
- User can input a **seed sentence and target length**, and the app generates fluent text
- Designed as a **Streamlit web app** for easy interaction and visualization

---

## 🛠️ Project Structure

```sh
wordwave/
├── app.py                 # Streamlit app UI and core functionalities
├── word-wave.ipynb        # Jupyter Notebook for training and saving model
├── word-wave.keras        # Trained Keras model (saved format)
├── tokenizer.pkl          # Fitted tokenizer object (Pickle)
├── requirements.txt       # All Python dependencies
├── README.md              # This file
└── .gitignore             # Standard gitignore file template
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Mozeel-V/wordwave.git
cd wordwave
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

You’ll be able to enter text, pick how many words to generate, and see live predictions along with evaluation metrics.

---

## 📈 Model Overview

- **Architecture**:  
  `Embedding → Bidirectional LSTM → Attention → Dense`
- **Loss**: Sparse Categorical Crossentropy  
- **Optimizer**: Adam  
- **Evaluation Metrics**:  
  - Top-5 Accuracy (37%+ on eval subset)  
  - BLEU Score  
  - Perplexity (>200 baseline)

- **Decoding**: Supports both **greedy** and **beam search** decoding
- **Training Data**: English Wikipedia (`0.1%` slice from `20220301.en`)

---

## 🧠 Sample Generation

```text
Seed: "deep learning models are"
Generated: "deep learning models are used to perform various tasks including natural language processing"
```

- BLEU Score: 0.38 
- Perplexity: 215.4

---

## 🧪 Evaluation

### Top-5 Accuracy

Implemented using `sklearn.metrics.top_k_accuracy_score`, measuring how often the true word appears in the model’s top 5 predictions.

### BLEU Score

Compares generated text to a reference sentence using `nltk` BLEU metric (1-gram to 4-gram weights).

### Perplexity

Calculated as the exponentiated negative average log-likelihood of predicted next words — lower is better.

---

## 🧰 Future Improvements

- Add character-level prediction
- Fine-tune with larger dataset portions
- Integrate GPT-style transformer decoder for comparison
- Export as REST API for backend integration

---

## ✅ How to Evaluate Model

You can run evaluation metrics either:

1. **Automatically via Streamlit app**, or  
2. **Manually from notebook**:

```python
from sklearn.metrics import top_k_accuracy_score
from nltk.translate.bleu_score import sentence_bleu
```

---

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NLTK
- scikit-learn
- NumPy

---

## 📝 License

MIT License — use freely for research and educational purposes.

---

## 🤝 Contributions

Contributions, feature requests, and feedback are welcome!

---

## 👨‍💻 Author

**Mozeel Vanwani** | IIT Kharagpur CSE  
📧 [vanwani.mozeel@gmail.com](mailto:vanwani.mozeel@gmail.com)

---
