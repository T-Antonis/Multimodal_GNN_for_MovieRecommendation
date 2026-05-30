"""Entry point for the project.

The original work is split into three stages:
1. Computer vision poster model
2. NLP plot-summary model
3. GraphSAGE recommendation model

Because training is GPU-heavy and dataset-dependent, run the stage scripts
individually after placing the dataset under data/.
"""


def main():
    print("Multimodal Movie Recommender GNN")
    print("Run src/computer_vision.py, src/nlp.py, and src/graph_recommender.py as needed.")


if __name__ == "__main__":
    main()
