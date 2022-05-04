import numpy as np
from black import out
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
from joblib import dump, load
import re
import umap


def plot_umap_clusters(results_dict, plot_store_path: str) -> None:
    kin_embeddings = np.array(results_dict["kin_embeddings"]).squeeze()
    opt_embeddings = np.array(results_dict["opt_embeddings"]).squeeze()
    embeddings = np.concatenate([opt_embeddings, kin_embeddings]).squeeze()
    print("Training umap reducer.")
    umap_reducer = umap.UMAP()
    # reduced_embeddings = umap_reducer.fit_transform(kin_embeddings)
    reduced_embeddings = umap_reducer.fit_transform(opt_embeddings)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    print("Generating skill plots.")
    plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict["skill"]]
    )
    plt.gca().set_aspect("equal", "datalim")
    # plt.title('UMAP projection of the Skill clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, "kin_umap_skill.png")
    # save_path = os.path.join(plot_store_path, 'opt_umap_skill.png')
    # save_path = os.path.join(plot_store_path, 'umap_skill.png')
    plt.savefig(save_path)
    plt.clf()
    le_gest = LabelEncoder()
    le_gest.fit(results_dict["gesture"])
    print("Generating gesture plots.")
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=[sns.color_palette()[x] for x in le_gest.transform(results_dict["gesture"])],
    )
    plt.gca().set_aspect("equal", "datalim")
    # plt.title('UMAP projection of the Gesture clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, "kin_umap_gesture.png")
    # save_path = os.path.join(plot_store_path, 'opt_umap_gesture.png')
    # save_path = os.path.join(plot_store_path, 'umap_gesture.png')
    plt.savefig(save_path)
    plt.clf()
    le_user = LabelEncoder()
    le_user.fit(results_dict["user"])
    print("Generating user plots.")
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=[sns.color_palette()[x] for x in le_user.transform(results_dict["user"])],
    )
    plt.gca().set_aspect("equal", "datalim")
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    # save_path = os.path.join(plot_store_path, 'kin_umap_user.png')
    save_path = os.path.join(plot_store_path, "opt_umap_user.png")
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    # NOT TESTED YET
    pass
    # plot_umap_clusters(embedding_dict, root / "clustering")
