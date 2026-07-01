import streamlit as st
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

# -----------------------------
# Character mappings
# -----------------------------
with open("data/1268-0.txt", "r", encoding="utf8") as fp:
    text = fp.read()

start_indx = text.find("THE MYSTERIOUS ISLAND")
end_indx = text.find("End of the Project Gutenberg")
text = text[start_indx:end_indx]

char_set = set(text)
chars_sorted = sorted(char_set)

char2int = {ch: i for i, ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)


# -----------------------------
# Model
# -----------------------------
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size

        self.rnn = nn.LSTM(
            embed_dim,
            rnn_hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell


vocab_size = len(char_array)

model = RNN(
    vocab_size=vocab_size,
    embed_dim=256,
    rnn_hidden_size=512
)

model = torch.load("RNNmodel.pth", map_location="cpu",weights_only=False)
model.eval()


# -----------------------------
# Text generation
# -----------------------------
def sample(model, starting_str, length, scale_factor=1.0):

    encoded_input = torch.tensor(
        [char2int[c] for c in starting_str]
    ).reshape(1, -1)

    hidden, cell = model.init_hidden(1)

    generated = starting_str

    # Feed initial prompt
    for c in range(len(starting_str) - 1):
        _, hidden, cell = model(
            encoded_input[:, c],
            hidden,
            cell,
        )

    last_char = encoded_input[:, -1]

    for _ in range(length):

        logits, hidden, cell = model(
            last_char,
            hidden,
            cell,
        )

        logits = logits.squeeze(0)

        m = Categorical(logits=logits * scale_factor)

        last_char = m.sample().view(1)

        generated += char_array[last_char.item()]

    return generated


# -----------------------------
# UI
# -----------------------------
st.title("Character-Level Text Generator")

st.write(
    """
Generate text one character at a time using an LSTM language model trained on
*The Mysterious Island* by Jules Verne.
"""
)

prompt = st.text_input(
    "Starting text",
    value="The island"
)

length = st.slider(
    "Characters to generate",
    50,
    1000,
    300
)

# temperature = st.slider(
#     "Randomness",
#     0.2,
#     2.0,
#     1.0,
#     0.1
# )

st.info(
    """
**Note**

This is a character-level LSTM trained on sequences of only **50 characters**.
While it learns spelling, punctuation and local language patterns quite well,
it struggles to maintain long-range context. As generation gets longer, the
text may gradually lose coherence because standard LSTMs have limited ability
to model very long dependencies.
"""
)

if st.button("Generate"):

    if len(prompt) == 0:
        st.warning("Please enter some starting text.")

    elif any(ch not in char2int for ch in prompt):
        st.error("Your prompt contains characters that were not present during training.")

    else:

        text = sample(
            model,
            prompt,
            length,
            2
        )

        st.subheader("Generated Text")

        st.write(text)