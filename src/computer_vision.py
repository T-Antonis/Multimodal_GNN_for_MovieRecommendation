"""Code extracted from the original experimental notebook.

The scripts keep the original modelling workflow while removing Colab-only
commands so the repository can be browsed cleanly on GitHub. Dataset paths may
need adjustment depending on where the data is stored locally.
"""

# Assuming datasets.zip is in the root of your Google Drive
# Notebook shell command removed: !unzip -o /content/drive/MyDrive/datasets.zip -d datasets/

# %%
import pandas as pd

# Load train_metadata_full.csv into train_subset
train_subset = pd.read_csv('./datasets/datasets/train_metadata_full.csv')
small_full = pd.read_csv('./datasets/datasets/ml_small_metadata_full.csv')

print("train_metadata_full.csv loaded successfully into train_subset.")
print("ml_small_metadata_full.csv loaded successfully into small_full.")

# %%
train_subset['genres'].head()

# %%
import sklearn.preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()


def encode_genres(df):

    genre_lists = df['genres'].fillna('').apply(lambda x: x.split('|') if x != '' else [])


    genre_matrix = mlb.fit_transform(genre_lists)
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, genre_df], axis=1)

    return df

train_subset = encode_genres(train_subset)
small_full = encode_genres(small_full)

print(f"Total unique genres: {len(mlb.classes_)}")
display(train_subset.head(3))

# %%
import os
from PIL import Image
import matplotlib.pyplot as plt

id = train_subset.iloc[42]['tmdbId']


IMAGE_DIR = "./datasets/datasets/posters/"

# Construct the image path
image_filename = f"{id}.jpg"
image_path = os.path.join(IMAGE_DIR, image_filename)


image = Image.open(image_path)


plt.imshow(image)
plt.title(f"Original Movie Poster")
plt.axis('off')
plt.show()


print(f"Image dimensions: {image.size[1]}x{image.size[0]} pixels (Height x Width)")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

genre_columns = mlb.classes_

# Sum the occurrences of each genre
genre_counts = train_subset[genre_columns].sum().sort_values(ascending=False)

# Plotting the distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette="viridis", legend=False)
plt.title("Genre Distribution in Training Dataset (Label Imbalance)", fontsize=14)
plt.xlabel("Number of Movies", fontsize=12)
plt.ylabel("Genre", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print the top 3 and bottom 3 genres
print("Top 3 Most Frequent Genres:")
print(genre_counts.head(3))
print("\nBottom 3 Least Frequent Genres:")
print(genre_counts.tail(3))

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MoviePosterDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        # Get the label columns directly from mlb.classes_
        # This ensures we always select only the genre columns by their names
        self_label_cols_ = list(mlb.classes_)
        self.label_cols = [col for col in self.dataframe.columns if col in self_label_cols_]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):


        img_name = self.dataframe.loc[idx, 'poster_file']
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.dataframe.loc[idx, self.label_cols].astype(np.float32).values, dtype=torch.float32)


        return image, labels


target_size = 128


poster_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.CenterCrop((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

poster_aug_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(degrees=5),
    transforms.CenterCrop((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# %%
from sklearn.model_selection import train_test_split

# Split the Training Data into Train and Validation (80/20)
train_df, val_df = train_test_split(train_subset, test_size=0.2, random_state=42)

# Instantiate the Datasets
print("Initializing Datasets...")
train_dataset = MoviePosterDataset(dataframe=train_df, image_dir=IMAGE_DIR, transform=poster_aug_transform)
val_dataset = MoviePosterDataset(dataframe=val_df, image_dir=IMAGE_DIR, transform=poster_transform)

# Use the ML-Small dataset as our Test Set!
test_dataset = MoviePosterDataset(dataframe=small_full, image_dir=IMAGE_DIR, transform=poster_transform)

# %%
import numpy as np

img, label = train_dataset[0]

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax in axes.flatten():
    aug_img, _ = train_dataset[0]   # ίδιο index, νέο random augment κάθε φορά

    aug_img = aug_img.numpy().transpose(1, 2, 0)   # C,H,W -> H,W,C
    aug_img = std * aug_img + mean                 # denormalize
    aug_img = np.clip(aug_img, 0, 1)

    ax.imshow(aug_img)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
import pickle as pkl
pkl.dump(train_df, open("train_df.pkl", "wb"))
pkl.dump(val_df, open("val_df.pkl", "wb"))

# %%
image, labels = train_dataset[42]
# Convert tensor to numpy array and move channels to the last dimension (H, W, C)
image = image.numpy().transpose((1, 2, 0))
# # Clip values to [0, 1] as some values might be slightly outside due to floating point operations
image = np.clip(image, 0, 1)

# Get the genre names corresponding to the '1's in the labels
# The label_cols from MoviePosterDataset stores the order of genres
label_cols = train_dataset.label_cols
active_genres = [genre for i, genre in enumerate(label_cols) if labels[i] == 1]
genres_str = ', '.join(active_genres) if active_genres else 'No genres listed'

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(f"Processed Movie Poster\nGenres: {genres_str}")
plt.axis('off')
plt.show()

# %%
batch_size = 32


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)} | Test (ML-Small) Size: {len(test_dataset)}")

print("For Training: ")
print(f"Number of Batches per Epoch: {len(train_loader)}")

sample_images, sample_labels = next(iter(train_loader))
print(f"Image Batch Shape: {sample_images.shape} -> [Batch, Channels, Height, Width]")
print(f"Label Batch Shape: {sample_labels.shape} -> [Batch, Num_Genres]")

# %%
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()



        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  #128 σε 64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  #64 σε 32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  #32 σε 16

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  #16 σε 8

            nn.Flatten()

        )


        self.flatten_dim = 256 * 8 * 8



        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):

        x = self.features(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def extract_embeddings(self, x):
        """
        Helper function for Task 3:
        Runs the forward pass up to the penultimate layer and returns the dense feature vector.
        """

        x = self.features(x)
        x = F.relu(self.fc1(x))
        return x

# %%
# Initialize the model
num_genres = len(mlb.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cv_model = CustomCNN(num_classes=num_genres).to(device)
print(f"Model initialized on: {device}")

# %%
# Get the number of positive occurrences for each genre in the training set
pos_counts = train_df[genre_columns].sum().values
total_samples = len(train_df)

# Calculate pos_weight: (negatives / positives)
pos_weights = (total_samples - pos_counts) / pos_counts

smoothed_weights = np.sqrt(pos_weights)


pos_weight_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

print("Calculated pos_weight for the first 5 genres:")
for i in range(5):
    print(f"{genre_columns[i]}: {pos_weight_tensor[i].item():.2f}")

# %%
import torch.optim as optim
from tqdm.notebook import tqdm


criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight_tensor)
optimizer = torch.optim.AdamW(cv_model.parameters(), lr=1e-3, weight_decay=1e-4)


epochs = 50

train_losses = []
val_losses = []

for epoch in range(epochs):
    # --- TRAINING PHASE ---
    cv_model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)


        optimizer.zero_grad()
        outputs = cv_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # Loss update
        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'batch_loss': loss.item()})


    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss) # Store for plotting

    # --- VALIDATION PHASE ---
    cv_model.eval()
    val_loss = 0.0

    # torch.no_grad() prevents PyTorch from storing gradients, saving memory during eval
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)


            outputs = cv_model(images)
            loss = criterion(outputs, labels)

            # Loss update
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(val_dataset)
    val_losses.append(val_loss) # Store for plotting

    print(f"End of Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("Training complete!")

# %%
# --- VISUALIZE LOSS CURVES ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss', marker='o', color='blue')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='o', color='orange')
plt.title('Training and Validation Loss Over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(range(1, epochs+1))
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
torch.save(cv_model.state_dict(), "/content/drive/MyDrive/cv_model_best3.pth")

# %%
from sklearn.metrics import f1_score, roc_auc_score

# ==========================================
# 1. Quantitative Evaluation
# ==========================================
cv_model.eval()

# TODO: Populate these lists with numpy arrays during your evaluation loop
all_targets = []
all_predictions = []

print("Evaluating model on the Test Set (ML-Small)...")


with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)

      logits = cv_model(images)
      probs = torch.sigmoid(logits)

      all_targets.append(labels.cpu().numpy())
      all_predictions.append(probs.cpu().numpy())

all_targets = np.vstack(all_targets)
all_predictions = np.vstack(all_predictions)

binary_predictions = (all_predictions >= 0.5).astype(int)



macro_f1 = f1_score(all_targets, binary_predictions, average='macro', zero_division=0)
micro_f1 = f1_score(all_targets, binary_predictions, average="micro", zero_division=0)
roc_auc = roc_auc_score(all_targets, all_predictions, average = 'macro')

# ==========================================
# --- VISUALIZE QUANTITATIVE METRICS ---
# ==========================================
metrics_dict = {'Macro F1': macro_f1, 'Micro F1': micro_f1, 'ROC-AUC': roc_auc}

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color=['#87CEEB', '#98FB98', '#FA8072'])
plt.ylim(0, 1.1)
plt.title('Test Set Evaluation Metrics', fontsize=14)
plt.ylabel('Score', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
pos_weight_tensor

# %%
# ==========================================
# 2. Qualitative  (Visual Grid)
# ==========================================
def imshow_denormalized(img):
    img = img / 2 + 0.5
    img = np.clip(img, 0, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


images, labels = next(iter(test_loader))

cv_model.eval()
with torch.no_grad():
    outputs = cv_model(images.to(device))
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).int().cpu()


fig, axes = plt.subplots(1, 5, figsize=(20, 5))
genre_names = mlb.classes_

for i in range(5):
    ax = axes[i]

    true_genres = [genre_names[j] for j in range(len(genre_names)) if labels[i][j] == 1]
    pred_genres = [genre_names[j] for j in range(len(genre_names)) if preds[i][j] == 1]


    # ==========================================
    # --- VISUALIZATION ---
    # ==========================================
    plt.sca(ax)
    imshow_denormalized(images[i])
    ax.axis('off')

    title_text = f"True: {', '.join(true_genres)}\nPred: {', '.join(pred_genres) if pred_genres else 'None'}"
    color = 'green' if set(true_genres) == set(pred_genres) else 'black'
    ax.set_title(title_text, fontsize=10, wrap=True, color=color)

plt.tight_layout()
plt.show()

# %%
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

train_subset = encode_genres(train_subset)
small_full = encode_genres(small_full)

print(f"Total unique genres: {len(mlb.classes_)}")

# %%
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# ==========================================
# Pretrained ResNet Initialization & Fine-Tuning
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load a pre-trained ResNet50
resnet_model = models.resnet50(pretrained=True)

# 2. Modify the final classification head
num_ftrs = resnet_model.fc.in_features
num_genres = len(mlb.classes_)

# Replace the last layer for the 20 categories with a Sequential block containing Dropout
resnet_model.fc = nn.Sequential(
    nn.Dropout(0.5),               # Drops 50% of neurons for regularization
    nn.Linear(num_ftrs, num_genres)
)

# Freeze early layers: only train layer4 and the classification head (fc)
for name, param in resnet_model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

#  Transfer the model to the GPU
resnet_model = resnet_model.to(device)

# 3. Optimizer, Scheduler & Loss
# Using a lower learning rate and weight decay for fine-tuning
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.00005, weight_decay=1e-4)

# Reduces LR by a factor of 10 every 2 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))

epochs_resnet = 10
train_losses_res = []
val_losses_res = []

# EARLY STOPPING VARIABLES
best_val_loss = float('inf')
patience = 3
patience_counter = 0

print(f"Fine-tuning ResNet50 with Scheduler and Early Stopping on {device}...")

# --- TRAINING LOOP ---
for epoch in range(epochs_resnet):
    # --- Training Phase ---
    resnet_model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_resnet} [Training]")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = resnet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

    epoch_train_loss = running_loss / len(train_dataset)
    train_losses_res.append(epoch_train_loss)

    # --- Validation Phase (Calculate metrics for loss curves) ---
    resnet_model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

    epoch_val_loss = val_running_loss / len(val_dataset)
    val_losses_res.append(epoch_val_loss)

    print(f"Epoch {epoch+1} Results: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # --- EARLY STOPPING CHECK ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        # Save the state of the model with the lowest validation loss
        torch.save(resnet_model.state_dict(), 'best_resnet_model.pth')
        patience_counter = 0
        print(f"--> Best model saved (Val Loss improved).")
    else:
        patience_counter += 1
        print(f"--> No improvement for {patience_counter} epoch(s).")

    if patience_counter >= patience:
        print(f"Early stopping triggered! Training stopped at epoch {epoch+1}")
        break

    # Update Learning Rate at the end of each epoch
    scheduler.step()

# ==========================================
# --- VISUALIZE LOSS CURVES ---
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses_res) + 1), train_losses_res, label='Training Loss', marker='o', color='blue')
plt.plot(range(1, len(val_losses_res) + 1), val_losses_res, label='Validation Loss', marker='s', color='orange')
plt.title('ResNet50 Training vs Validation Loss', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ==========================================
# 1. Quantitative Evaluation
# ==========================================
# Load the best performing model saved during training before final evaluation
resnet_model.load_state_dict(torch.load('best_resnet_model.pth'))
resnet_model.eval()

all_targets_res = []
all_predictions_res = []

print("Evaluating the BEST model on the Test Set (ML-Small)...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = resnet_model(images)

        # Apply Sigmoid to get probabilities
        probs = torch.sigmoid(outputs)

        all_targets_res.append(labels.cpu().numpy())
        all_predictions_res.append(probs.cpu().numpy())

# Concatenate all batches
all_targets_res = np.vstack(all_targets_res)
all_predictions_res = np.vstack(all_predictions_res)

# Binary predictions based on a 0.3 threshold
binary_predictions_res = (all_predictions_res > 0.3).astype(int)

print(f"Evaluation Complete!")

# Calculate Metrics
macro_f1_res = f1_score(all_targets_res, binary_predictions_res, average='macro', zero_division=0)
micro_f1_res = f1_score(all_targets_res, binary_predictions_res, average='micro', zero_division=0)
roc_auc_res = roc_auc_score(all_targets_res, all_predictions_res, average='macro')

# ==========================================
# --- VISUALIZE QUANTITATIVE METRICS ---
# ==========================================
metrics_dict_res = {'Macro F1': macro_f1_res, 'Micro F1': micro_f1_res, 'ROC-AUC': roc_auc_res}

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_dict_res.keys(), metrics_dict_res.values(), color=['#87CEEB', '#98FB98', '#FA8072'])
plt.ylim(0, 1.1)
plt.title('Test Set Evaluation Metrics (ResNet50 - Best Model)', fontsize=14)
plt.ylabel('Score', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
import torch.optim as optim
from tqdm.notebook import tqdm

# --- 1. SETUP ---

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.AdamW(nlp_model.parameters(), lr=5e-4, weight_decay=1e-4)

epochs = 20
train_losses = []
val_losses = []

# --- 2. TRAINING & VALIDATION LOOP ---
for epoch in range(epochs):
    # --- TRAINING PHASE ---
    nlp_model.train()
    running_train_loss = 0.0

    for texts, labels in nlp_train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = nlp_model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nlp_model.parameters(), max_norm=1)
        optimizer.step()

        running_train_loss += loss.item() * texts.size(0)

    train_loss = running_train_loss / len(nlp_train_loader)



    # --- VALIDATION PHASE ---
    nlp_model.eval()
    running_val_loss = 0.0


    with torch.no_grad():
        for texts, labels in nlp_val_loader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = nlp_model(texts)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * texts.size(0)

    val_loss = running_val_loss / len(nlp_val_loader)



    # Save average losses for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")



# ==========================================
# --- VISUALIZATION LOSS CURVES ---
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss', marker='o', color='blue')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='o', color='orange')
plt.title('NLP Model Loss Over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Weighted BCE)', fontsize=12)
plt.xticks(range(1, epochs+1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
torch.save(nlp_model.state_dict(), "/content/drive/MyDrive/nlp_model_best4.pth")

# %%
# --- 3. EVALUATION N TEST SET (ML-Small) ---


from sklearn.metrics import f1_score, roc_auc_score

print("\nEvaluating on Test Set...")



nlp_model.eval()
all_targets = []
all_predictions = []

with torch.no_grad():
    for texts, labels in nlp_test_loader:
        texts, labels = texts.to(device), labels.to(device)

        logits = nlp_model(texts)
        probs = torch.sigmoid(logits)

        all_targets.append(labels.cpu().numpy())
        all_predictions.append(probs.cpu().numpy())

all_targets = np.vstack(all_targets)
all_predictions = np.vstack(all_predictions)


binary_predictions = (all_predictions >= 0.5).astype(int)




macro_f1 = f1_score(all_targets, binary_predictions, average='macro', zero_division=0)
micro_f1 = f1_score(all_targets, binary_predictions, average='micro', zero_division=0)
roc_auc = roc_auc_score(all_targets, all_predictions, average='macro')

print(f"Test Macro F1-Score: {macro_f1:.4f}")
print(f"Test Micro F1-Score: {micro_f1:.4f}")

# ==========================================
# --- VISUALIZE QUANTITATIVE METRICS ---
# ==========================================
metrics_dict = {'Macro F1': macro_f1, 'Micro F1': micro_f1, 'ROC-AUC': roc_auc}

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color=['#87CEEB', '#98FB98', '#FA8072'])
plt.ylim(0, 1.1)
plt.title('Test Set Evaluation Metrics', fontsize=14)
plt.ylabel('Score', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
pos_weight_tensor
