# Multimodal Graph Neural Networks for Movie Recommendation

This repository contains a complete deep learning project for movie recommendation using three complementary approaches: Computer Vision, Natural Language Processing, and Graph Neural Networks.

The project starts by learning movie representations from posters and plot summaries, then uses those representations inside a user–movie graph for link prediction and recommendation.

## Project Highlights

- Multi-label genre classification from movie posters using a custom CNN.
- Transfer learning with ResNet50 for stronger visual embeddings.
- Multi-label genre classification from plot summaries using a GRU with attention.
- Embedding extraction from both the CV and NLP models.
- Construction of a heterogeneous user–movie graph.
- GraphSAGE-based link prediction for movie recommendation.
- Comparison of different movie feature strategies: baseline, CV, NLP, multimodal and pretrained features.

## Repository Structure

```text
.
├── data/
│   └── README.md
├── models/
│   └── README.md
├── outputs/
│   ├── figure_*.png
├── src/
│   ├── bio_data/
│   ├── computer_vision.py
│   ├── graph_recommender.py
│   ├── main.py
│   ├── nlp.py
│   └── setup.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

The original experimentation was developed in a notebook environment. For GitHub presentation, the code has been reorganized into Python scripts and the main results have been kept in the `outputs/` folder.

## Methodology

### 1. Computer Vision

We first treat movie genre prediction as a multi-label image classification problem. Each movie poster can correspond to multiple genres, so the target labels are encoded as multi-hot vectors.

The visual pipeline includes:

- poster loading with a custom PyTorch `Dataset`,
- image resizing and augmentation,
- genre distribution analysis,
- training a custom CNN from scratch,
- fine-tuning a pretrained ResNet50 model,
- extracting poster-based movie embeddings.

Because the genre distribution is imbalanced, the loss function uses positive class weights with `BCEWithLogitsLoss`. This helps the model avoid overfitting only to frequent genres such as Drama or Comedy.

### 2. Natural Language Processing

The NLP part uses movie plot summaries to predict genres. Since plots contain direct semantic information about the story, they are expected to provide stronger genre signals than posters alone.

The text pipeline includes:

- lowercasing and punctuation cleaning,
- tokenization,
- vocabulary creation,
- padding/truncation of sequences,
- GRU-based classification,
- attention mechanism for weighted sequence representation,
- semantic embedding extraction.

The NLP model is evaluated with Macro F1, Micro F1 and ROC-AUC.

### 3. Graph Neural Network Recommendation

The final part builds a heterogeneous bipartite graph with two node types:

```text
user --rates--> movie
```

The model uses GraphSAGE for link prediction. Positive edges come from real user ratings, while negative examples are generated during training. This allows the model to learn whether a user is likely to interact with a movie.

We compare several movie-node feature strategies:

- random baseline features,
- CV poster embeddings,
- NLP plot embeddings,
- multimodal CV + NLP embeddings,
- pretrained/learned feature variants.

Finally, the project includes a recommendation function that returns human-readable top-k movie recommendations for a selected user.

## Results

### NLP Model Performance

| Metric | Score |
|---|---:|
| Macro F1-score | 0.2798 |
| Micro F1-score | 0.4719 |

The NLP model performed better than the custom CNN baseline for genre prediction, which suggests that plot summaries contain more useful semantic information than poster images alone.

### Example GNN Recommendation Output

For user `42`, the graph model produced recommendations such as:

```text
Star Trek: First Contact (1996)
Aliens (1986)
Heat (1995)
Spider-Man (2002)
Under Siege 2: Dark Territory (1995)
```

This shows that the model can generate interpretable movie recommendations from the learned user–movie graph.

## Visual Results

Some of the generated plots and visual outputs are stored in `outputs/`.

Examples include:

- genre distribution plots,
- poster preprocessing examples,
- training and validation loss curves,
- evaluation metric visualizations,
- GNN training results.

You can view them directly through the GitHub file browser.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

PyTorch Geometric installation can depend on the local CUDA/PyTorch version. If needed, follow the official PyTorch Geometric installation instructions for your environment.

## Data Setup

The dataset is not included in this repository because it can be large.

Expected structure after extraction:

```text
data/datasets/train_metadata_full.csv
data/datasets/ml_small_metadata_full.csv
data/datasets/posters/
data/datasets/ml-latest-small/ratings.csv
```

Model checkpoints and embedding tensors can be stored under `models/` or `data/`, depending on the experiment setup.

## How to Run

The code is split into stages:

```bash
python src/computer_vision.py
python src/nlp.py
python src/graph_recommender.py
```

A lightweight entry point is also included:

```bash
python src/main.py
```

The training scripts are GPU-heavy and expect the dataset paths to be correctly configured before execution.

## Technologies Used

- Python
- PyTorch
- Torchvision
- PyTorch Geometric
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow

## What This Project Demonstrates

This project demonstrates a full multimodal machine learning workflow:

- building deep learning models for images and text,
- extracting learned embeddings,
- using those embeddings as graph node features,
- training a GNN for link prediction,
- turning graph predictions into real recommendations.

The main takeaway is that recommendation systems can be improved by combining collaborative signals from user interactions with content-based information from multiple data modalities.

## Future Improvements

Possible extensions include:

- replacing the GRU with a transformer-based model such as BERT,
- testing stronger vision backbones such as EfficientNet or ViT,
- adding rating values as edge attributes,
- evaluating recommendations with Precision@K, Recall@K and NDCG@K,
- creating a small web app for interactive movie recommendations.

## Author

Developed as a personal machine learning project focused on multimodal recommendation systems, deep learning and graph-based learning.
