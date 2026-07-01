# Character-Level Language Model using LSTM

**Streamlit link:** [click here to see the deployed app](https://characterlevellanguagemodeling.streamlit.app/) \
This project implements a character-level language model from scratch in PyTorch. Instead of predicting complete words, the model predicts the next **character** given all previous characters.

The model was trained on **The Mysterious Island** by Jules Verne and can generate new text one character at a time.

---

## How it works

The pipeline consists of several stages:

- Extract unique characters from the text
- Convert every character into an integer index
- <img width="529" height="256" alt="image" src="https://github.com/user-attachments/assets/6caf430e-4f14-4a0a-8802-085ffddc7582" />

- Create overlapping sequences of length 50
- Learn character embeddings
- <img width="623" height="452" alt="image" src="https://github.com/user-attachments/assets/23c651cc-a929-42f1-acd3-14c56c2f8724" />

- Process sequences using an LSTM
- Predict the next character using a linear classifier


During inference, the model repeatedly predicts one character, appends it to the input, and feeds it back into itself to continue generating text.

<img width="582" height="393" alt="image" src="https://github.com/user-attachments/assets/c85a395a-56f5-4630-a935-c5fc5b4be0f0" />

---

## Concepts Demonstrated

This project covers many important NLP and deep learning concepts including:

- Character-level language modeling
- Embedding layers
- LSTM architecture
- Hidden state and cell state propagation
- Cross-Entropy Loss for multi-class classification
- Probabilistic sampling using categorical distributions
- Temperature (logit scaling) for controlling randomness

---

## Character-Level

Instead of learning:

```
The → island
```

the model learns

```
T → h → e →   → i → s → l → a ...
```

---

## Limitations

The model was trained using sequences of **50 characters**.

Although the LSTM carries information through its hidden and cell states, it still struggles to preserve context over very long passages. As a result:

- words are usually spelled correctly,
- punctuation often looks realistic,
- short phrases are meaningful,

but longer generations may slowly drift away from coherent sentences.

This is a common limitation of traditional recurrent neural networks and one of the motivations behind Transformer-based language models.

---

## Next is

- Transformer-based language model
