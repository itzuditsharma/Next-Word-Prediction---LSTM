# Next-Word-Prediction---LSTM
This project demonstrates a simple but effective Recurrent Neural Network (RNN) based model using Long Short-Term Memory (LSTM) layers for Next Word Prediction. Given a sequence of words, the model predicts the most likely next word based on patterns learned from a corpus of text.

🚀 Project Overview
Predicting the next word in a sentence has several applications in text completion, chatbots, language modeling, and smart typing systems. This project uses a deep learning approach to build a predictive text model using an LSTM network in TensorFlow/Keras.

📂 Dataset
The dataset used is a sample of text data from a file (story.txt). This text is tokenized and preprocessed to generate input-output pairs suitable for training a sequence prediction model.

🧠 Model Architecture
The model is built using Keras with the following architecture:

Embedding Layer – Transforms input word indices into dense vectors.

LSTM Layer – Captures temporal dependencies in the sequence.

Dense Output Layer – Uses softmax activation to predict the next word among the vocabulary.

python
Copy
Edit
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
🛠️ Preprocessing Steps
Tokenization: Text is tokenized into words using Keras's Tokenizer.

Input Sequences: Generated using n-gram sequences.

Padding: Input sequences are padded to the same length using pad_sequences.

Labels: One-hot encoding is applied to the target word.

📊 Training
Loss Function: Categorical Crossentropy

Optimizer: Adam

Epochs: 100

The model learns to minimize the categorical crossentropy loss between the predicted and actual next word.

📈 Sample Result
After training, you can test the model using any seed text. Here's an example:

python
Copy
Edit
seed_text = "Today is a beautiful"
next_words = 5
Output:

csharp
Copy
Edit
Today is a beautiful morning and the
🔧 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/next-word-lstm.git
cd next-word-lstm
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:

bash
Copy
Edit
jupyter notebook Next_word_LSTM.ipynb
🧾 Requirements
Python 3.x

TensorFlow / Keras

Numpy

NLTK

Jupyter Notebook

You can install them with:

bash
Copy
Edit
pip install tensorflow numpy nltk
📚 Applications
Autocomplete Systems

AI Chatbots

Smart Text Editors

Language Modeling Research

📌 To-Do
Add support for beam search decoding.

Train on larger datasets like Wikipedia.

Deploy as a web app using Flask or Streamlit.
