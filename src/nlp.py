"""Code extracted from the original experimental notebook.

The scripts keep the original modelling workflow while removing Colab-only
commands so the repository can be browsed cleanly on GitHub. Dataset paths may
need adjustment depending on where the data is stored locally.
"""

# Assuming datasets.zip is in the root of your Google Drive
# Notebook shell command removed: !unzip -o /content/drive/MyDrive/datasets.zip -d datasets/

# %%
import pickle as pkl
import pandas as pd


# load pre-saved pickled dataframes, if needed
train_df = pkl.load(open('train_df.pkl', 'rb'))
val_df = pkl.load(open('val_df.pkl', 'rb'))

# %%
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Simple Tokenizer, returns a list of strings
def tokenize(text):
    """Lowercases, removes punctuation, and splits by whitespace."""

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()

    return tokens


# Build Vocabulary from Training Set ONLY
print("Building vocabulary...")
all_tokens = []
for plot in train_df['overview']:
    all_tokens.extend(tokenize(plot))

# %%
# Keep top 10,000 most common words, reserve 0 for <PAD> and 1 for <UNK>

max_vocab_size = 10000
vocab_counts = Counter(all_tokens)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(vocab_counts.most_common(max_vocab_size))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def text_to_tensor(text, vocab, max_len=150):
    """Converts a text string to a padded tensor of token IDs."""
    tokens = tokenize(text)
    token_ids = [vocab.get(word, vocab['<UNK>']) for word in tokens]

    # Truncate if too long
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]

    # Pad if too short
    pad_len = max_len - len(token_ids)
    token_ids = token_ids + [vocab['<PAD>']] * pad_len

    return torch.tensor(token_ids, dtype=torch.long)

# %%
class MoviePlotDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.label_cols = self.dataframe.columns[6:].tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        plot_text = self.dataframe.loc[idx, 'overview']
        text_tensor = text_to_tensor(plot_text, self.vocab)

        # Extract labels
        labels = self.dataframe.loc[idx, self.label_cols].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)

        return text_tensor, labels

# %%
print("Initializing NLP Datasets...")
nlp_train_dataset = MoviePlotDataset(train_df, vocab)
nlp_val_dataset = MoviePlotDataset(val_df, vocab)
nlp_test_dataset = MoviePlotDataset(small_full, vocab)

# %%
import torch

batch_size = 32
nlp_train_loader = DataLoader(nlp_train_dataset, batch_size=batch_size, shuffle=True)
nlp_val_loader = DataLoader(nlp_val_dataset, batch_size=batch_size, shuffle=False)
nlp_test_loader = DataLoader(nlp_test_dataset, batch_size=batch_size, shuffle=False)

# Test the dataloader
print("For training...")
sample_texts, sample_labels = next(iter(nlp_train_loader))
print(f"Text Batch Shape: {sample_texts.shape} -> [Batch, Sequence_Length]")
print(f"Label Batch Shape: {sample_labels.shape} -> [Batch, Num_Genres]")

# %%
import torch.nn as nn
import torch.nn.functional as F

class CustomTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(CustomTextClassifier, self).__init__()




        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.embedding_dropout = nn.Dropout(0.3)    #νέκρωση νευρώνων για αποφυγή overfit

        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        self.context_dropout = nn.Dropout(0.3)

        self.attention_proj = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)



    def forward(self, x):

        x = self.embedding(x)
        x = self.embedding_dropout(x)
        H, _ = self.rnn(x)

        e = self.attention_proj(H).squeeze(-1)
        alpha = F.softmax(e, dim=1)

        c = torch.sum(H * alpha.unsqueeze(-1), dim=1)
        c = self.context_dropout(c)

        logits = self.fc(c)

        return logits




    def extract_embeddings(self, x):
        """Returns the final context vector 'c' BEFORE the classification head."""

        x = self.embedding(x)
        x = self.embedding_dropout(x)
        H, _ = self.rnn(x)

        e = self.attention_proj(H).squeeze(-1)
        alpha = F.softmax(e, dim=1)

        c = torch.sum(H * alpha.unsqueeze(-1), dim=1)

        return c

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model will be initialized in {device}.\n")

vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 64
num_classes = len(nlp_train_dataset.label_cols)

nlp_model = CustomTextClassifier(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
print(nlp_model)
