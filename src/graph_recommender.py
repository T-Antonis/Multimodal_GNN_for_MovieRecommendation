"""Code extracted from the original experimental notebook.

The scripts keep the original modelling workflow while removing Colab-only
commands so the repository can be browsed cleanly on GitHub. Dataset paths may
need adjustment depending on where the data is stored locally.
"""

from torch.utils.data import DataLoader



# ==========================================
# 1. Choose your best model
# ==========================================

# Based on the metrics, the best model is ResNet50
best_vision_model = resnet_model

# Store the original fc layer to restore it later if needed
original_fc = best_vision_model.fc

# Replace the final classification layer with Identity.
# This ensures the model returns the output of the penultimate layer (2048 features)
best_vision_model.fc = nn.Identity()

# Setup DataLoader (shuffle MUST be False to maintain the correct order for movieIds!)
inference_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

# Set the chosen model to evaluation mode and move to device
best_vision_model.to(device)
best_vision_model.eval()
all_embeddings = []

print(f"Extracting embeddings for {len(test_dataset)} movies...")

# ==========================================
# 2. Write the extraction loop
# ==========================================
with torch.no_grad(): # Disable gradient calculation for inference efficiency
    for images, _ in tqdm(inference_loader, desc="Extracting"):
        # Move images to GPU
        images = images.to(device)

        # Forward pass: Get the feature embeddings (thanks to nn.Identity)
        embeddings = best_vision_model(images)

        # Move embeddings back to CPU and add to list
        all_embeddings.append(embeddings.cpu())

# ---------------------------------

# Concatenate all batches into a single large tensor
final_visual_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Extracted Embeddings Shape: {final_visual_embeddings.shape} -> [Num_Movies, Embedding_Dim]")

# Restore the original fc layer
best_vision_model.fc = original_fc

# ==========================================
# --- SAVING BOILERPLATE ---
# ==========================================
save_dict = {
    'movie_ids': torch.tensor(small_full['movieId'].values),
    'embeddings': final_visual_embeddings
}

# Create directory if it doesn't exist
if not os.path.exists('datasets'):
    os.makedirs('datasets')

# Save the extracted embeddings to a file
torch.save(save_dict, 'datasets/cv_embeddings.pt')
print("Visual embeddings successfully saved to 'datasets/cv_embeddings.pt'!")

# %%
nlp_model.eval()
all_nlp_embeddings = []

with torch.no_grad():
    for texts, _ in nlp_test_loader:
        texts = texts.to(device)

        embeddings = nlp_model.extract_embeddings(texts)
        all_nlp_embeddings.append(embeddings.cpu())

final_nlp_embeddings = torch.cat(all_nlp_embeddings, dim=0)



# ---------------------------------


# ==========================================
# --- SAVING ---
# ==========================================
print(f"Extracted NLP Embeddings Shape: {final_nlp_embeddings.shape} -> [Num_Movies, Hidden_Dim]")

save_dict = {
    'movie_ids': torch.tensor(small_full['movieId'].values),
    'embeddings': final_nlp_embeddings
}

torch.save(save_dict, 'datasets/nlp_embeddings.pt')
print("Semantic NLP embeddings successfully saved to 'datasets/nlp_embeddings.pt'!")

# %%
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# Load the ratings data from the small graph
ratings_df = pd.read_csv('datasets/datasets/ml-latest-small/ratings.csv')

# Only keep ratings for movies that exist in our valid small_full set (with TMDB links)
small_full = pd.read_csv('datasets/datasets/ml_small_metadata_full.csv')
valid_movie_ids = set(small_full['movieId'])
ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]

# Create Contiguous IDs for Users and Movies
unique_users = ratings_df['userId'].unique()
unique_movies = small_full['movieId'].unique() # Use the full valid subset, not just rated ones

user_mapping = {userid: i for i, userid in enumerate(unique_users)}
movie_mapping = {movieid: i for i, movieid in enumerate(unique_movies)}

# Map the edges
src = [user_mapping[uid] for uid in ratings_df['userId']]
dst = [movie_mapping[mid] for mid in ratings_df['movieId']]
edge_index = torch.tensor([src, dst], dtype=torch.long)

print(f"Total Users: {len(user_mapping)}")
print(f"Total Movies: {len(movie_mapping)}")
print(f"Total Ratings (Edges): {edge_index.shape[1]}")

# %%
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# Load the ratings data from the small graph
ratings_df = pd.read_csv('datasets/datasets/ml-latest-small/ratings.csv')

# Only keep ratings for movies that exist in our valid small_full set (with TMDB links)
small_full = pd.read_csv('datasets/datasets/ml_small_metadata_full.csv')
valid_movie_ids = set(small_full['movieId'])
ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]

# Create Contiguous IDs for Users and Movies
unique_users = ratings_df['userId'].unique()
unique_movies = small_full['movieId'].unique() # Use the full valid subset, not just rated ones

user_mapping = {userid: i for i, userid in enumerate(unique_users)}
movie_mapping = {movieid: i for i, movieid in enumerate(unique_movies)}

# Map the edges
src = [user_mapping[uid] for uid in ratings_df['userId']]
dst = [movie_mapping[mid] for mid in ratings_df['movieId']]
edge_index = torch.tensor([src, dst], dtype=torch.long)

print(f"Total Users: {len(user_mapping)}")
print(f"Total Movies: {len(movie_mapping)}")
print(f"Total Ratings (Edges): {edge_index.shape[1]}")


cv_data = torch.load('datasets/cv_embeddings.pt')
cv_features = cv_data['embeddings'] # Shape: [Num_Movies, 512]

nlp_data = torch.load('datasets/nlp_embeddings.pt')
nlp_features = nlp_data['embeddings'] # Shape: [Num_Movies, 256]

# %%
cv_data = torch.load('datasets/cv_embeddings.pt')
cv_features = cv_data['embeddings'] # Shape: [Num_Movies, 512]


nlp_data = torch.load('datasets/nlp_embeddings.pt')
nlp_features = nlp_data['embeddings'] # Shape: [Num_Movies, 256]

# %%
import torch.nn.functional as F

def get_movie_features(strategy, cv_features, nlp_features, num_movies):
    """Returns the appropriate node feature tensor based on the selected ablation strategy."""
    if strategy == "baseline":
        return torch.randn((num_movies, 64))

    elif strategy == "cv":
      return F.normalize(cv_features, p=2, dim=1)


    elif strategy == "nlp":
      return F.normalize(nlp_features, p=2, dim=1) #p=2 L2 and dim=1 for row


    elif strategy == "multimodal":
      #  seperately
        cv_norm = F.normalize(cv_features, p=2, dim=1)
        nlp_norm = F.normalize(nlp_features, p=2, dim=1)
        # connect everything
        multimodal_feats = torch.cat([cv_norm, nlp_norm], dim=1)
        # normalization
        return F.normalize(multimodal_feats, p=2, dim=1)


    elif strategy == "pretrained":
        pretrained_data = torch.load('pretrained_nlp_embeddings.pt')
        #  dictionary  key 'embeddings'
        p_feats = pretrained_data['embeddings'] if isinstance(pretrained_data, dict) else pretrained_data
        return F.normalize(p_feats, p=2, dim=1)


    else:
        raise ValueError("Invalid strategy! Please choose 'baseline', 'cv', 'nlp', 'multimodal', or 'pretrained'.")

# %%
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

def build_graph(movie_features):
    data = HeteroData()

    # User nodes just need to know how many exist
    data['user'].num_nodes = len(user_mapping)

    # Movie nodes require the actual semantic features
    data['movie'].x = movie_features

    # Add Edges
    data['user', 'rates', 'movie'].edge_index = edge_index

    # GNNs need undirected paths to pass messages back and forth
    data = T.ToUndirected()(data)

    return data

# %%
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
# ADD add_self_loops=False (for heterogenous) και edge features  #project=True
class HeteroGNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # HeteroConv allows us to define specific message passing for specific edge types.
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels, root_weight=True),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels, root_weight=True),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), out_channels, root_weight=True),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), out_channels,root_weight=True),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        x_initial = {key: x for key, x in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.3, training=self.training) for key, x in x_dict.items()} #το30%
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x + x_initial[key] for key, x in x_dict.items()} #skip connection
        return x_dict


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Decodes the probability of a link by concatenating User and Movie embeddings
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # Get the embeddings for the source (User) and destination (Movie)
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)
        z = F.relu(self.lin1(z))
        z = self.lin2(z)
        return z.view(-1)


class RecommenderGNN(nn.Module):
    def __init__(self, hidden_channels, num_users, movie_feature_dim):
        super().__init__()
        # 1. Structural Embedding for Users (They have no metadata)
        self.user_emb = nn.Embedding(num_users, hidden_channels)

        # 2. Projection Layer for Movies
        self.movie_proj = nn.Linear(movie_feature_dim, hidden_channels)

        self.encoder = HeteroGNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Initialize node representations
        h_dict = {
            'user': self.user_emb.weight,
            'movie': self.movie_proj(x_dict['movie'])
        }

        # Pass through GraphSAGE layers
        z_dict = self.encoder(h_dict, edge_index_dict)

        # Decode the requested edges
        return self.decoder(z_dict, edge_label_index)

# %%
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# --- TRAINING & EVALUATION LOOP ---
def train_and_evaluate_graph(graph_data, num_epochs=50, hidden_channels=64, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Bigger Negative Sampling for better accuraccy
    transform = T.RandomLinkSplit(
        num_val=0.1, num_test=0.1, disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0, # strict training
        add_negative_train_samples=True,
        edge_types=[('user', 'rates', 'movie')],
        rev_edge_types=[('movie', 'rev_rates', 'user')]
    )

    train_data, val_data, test_data = transform(graph_data)
    train_data, val_data = train_data.to(device), val_data.to(device)

    model = RecommenderGNN(hidden_channels, graph_data['user'].num_nodes, graph_data['movie'].x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 2. Scheduler: lowers LR if AUC is stack
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    criterion = torch.nn.BCEWithLogitsLoss()
    history = {'loss': [], 'val_auc': []}

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x_dict, train_data.edge_index_dict, train_data['user', 'rates', 'movie'].edge_label_index)
        loss = criterion(out, train_data['user', 'rates', 'movie'].edge_label.float())
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x_dict, val_data.edge_index_dict, val_data['user', 'rates', 'movie'].edge_label_index)
            val_auc = roc_auc_score(val_data['user', 'rates', 'movie'].edge_label.cpu(), torch.sigmoid(val_out).cpu())

        history['loss'].append(loss.item())
        history['val_auc'].append(val_auc)

        # updates Scheduler
        scheduler.step(val_auc)

        # 3. Early Stopping Logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 7: # 7 epochs
            print(f"Early stopping at epoch {epoch}")
            break

    return history, model


# %%
num_movies = len(movie_mapping)
strategies = ['baseline', 'cv', 'nlp', 'multimodal', 'pretrained']

# %%
num_movies = len(movie_mapping)
strategies = ['baseline', 'cv', 'nlp', 'multimodal', 'pretrained']
results = {}
for strategy in strategies:
    print(f"Training Strategy: {strategy.upper()}")

    # 1. Get features using the helper function
    features = get_movie_features(strategy, cv_features, nlp_features, num_movies)

    # 2. Build the PyG HeteroData Graph
    graph = build_graph(features)

    # 3. Train and Evaluate
    history, _ = train_and_evaluate_graph(graph, num_epochs=40)
    results[strategy] = history

# --- PLOTTING THE COMPARISON ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
styles = ['--', '-.', ':', '-', '-']
colors = ['gray', 'blue', 'green', 'purple', 'red']

for i, (name, history) in enumerate(results.items()):
    epochs = range(1, len(history['loss']) + 1)

    ax1.plot(epochs, history['loss'], linestyle=styles[i], color=colors[i], label=name.upper(), linewidth=2)
    ax2.plot(epochs, history['val_auc'], linestyle=styles[i], color=colors[i], label=name.upper(), linewidth=2)

ax1.set_title('Training Loss (BCE) over Epochs', fontsize=14)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_title('Validation ROC-AUC over Epochs', fontsize=14)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('ROC-AUC Score', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# %%
import torch
import pandas as pd
import numpy as np

def recommend_movies_for_user(user_id, model, graph_data, metadata_df, user_mapping, movie_mapping, top_k=5):
    """Generates human-readable movie recommendations for a specific user in the following fomat:
    --- USER X HISTORY ---
    They have already rated N movies. Here are a few:
        Movie 1
        Movie 2
        Movie 3
        Movie 4
        Movie 5


    --- TOP 5 GNN RECOMMENDATIONS ---
        Movie 1 (Probability)
        Movie 2 (Probability)
        Movie 3 (Probability)
        Movie 4 (Probability)
        Movie 5 (Probability)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph_data = graph_data.to(device)
    model.eval()

    # 1. Reverse the `movie_mapping` dictionary so you can convert Graph IDs back to MovieLens IDs.
    #    Also, create a dictionary from `metadata_df` to look up movie titles by their MovieLens ID.
    inv_movie_mapping = {v: k for k, v in movie_mapping.items()}
    movie_titles = dict(zip(metadata_df['movieId'], metadata_df['title']))

    # 2. Find the internal Graph ID for the requested `user_id` using `user_mapping`.
    if user_id not in user_mapping:
        print(f"User {user_id} not found in the graph.")
        return

    u_graph_id = user_mapping[user_id]

    # 3. Find all movies the user has ALREADY rated.
    edge_index = graph_data['user', 'rates', 'movie'].edge_index
    mask = edge_index[0] == u_graph_id
    rated_movie_indices = edge_index[1, mask].cpu().numpy()

    print(f"--- USER {user_id} HISTORY ---")
    print(f"They have already rated {len(rated_movie_indices)} movies. Here are a few:")
    for m_idx in rated_movie_indices[:5]:
        m_id = inv_movie_mapping[m_idx]
        print(f"   {movie_titles.get(m_id, 'Unknown Title')}")

    # 4. Find all candidate movies (Movies this user has NOT rated yet).
    num_movies = graph_data['movie'].num_nodes
    all_movie_indices = np.arange(num_movies)
    candidate_indices = np.setdiff1d(all_movie_indices, rated_movie_indices)

    # 5. Create a `pred_edge_index` tensor of shape [2, num_candidates].
    u_idx_repeated = torch.full((len(candidate_indices),), u_graph_id, dtype=torch.long)
    m_idx_candidates = torch.from_numpy(candidate_indices).long()
    pred_edge_index = torch.stack([u_idx_repeated, m_idx_candidates], dim=0).to(device)

    # 6. Forward Pass: Ask the GNN to predict the probability of these candidate edges.
    with torch.no_grad():
        preds = model(graph_data.x_dict,
                      graph_data.edge_index_dict,
                      pred_edge_index)
        probs = torch.sigmoid(preds).cpu().numpy()

    # 7. Sort the predictions to find the indices of the `top_k` highest probabilities.
    top_indices = probs.argsort()[-top_k:][::-1]

    print(f"\n--- TOP {top_k} GNN RECOMMENDATIONS ---")
    for i in top_indices:
        m_idx = candidate_indices[i]
        m_id = inv_movie_mapping[m_idx]
        print(f"   {movie_titles.get(m_id, 'Unknown Title')} ({probs[i]:.4f})")

# %%
graph = build_graph(get_movie_features("cv", cv_features, nlp_features, num_movies))
_, model = train_and_evaluate_graph(graph, num_epochs=40)

recommend_movies_for_user(
    user_id=42,
    model=model,
    graph_data=graph,
    metadata_df=small_full,
    user_mapping=user_mapping,
    movie_mapping=movie_mapping,
    top_k=5
)
