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

from deepgesture.config import Config
from deepgesture.Models.EncoderDecoder import encoderDecoder
from deepgesture.Dataset.BlobDataset import gestureBlobDataset, size_collate_fn
from pytorchcheckpoint.checkpoint import CheckpointHandler


def store_embeddings_in_dict(blobs_folder_path: str, model: encoderDecoder) -> dict:
    blobs_folder = os.listdir(blobs_folder_path)
    blobs_folder = list(filter(lambda x: ".DS_Store" not in x, blobs_folder))
    blobs_folder.sort(key=lambda x: int(x.split("_")[1]))

    embeddings_list = []
    gestures_list = []
    user_list = []
    skill_dict = {"B": 0, "C": 1, "D": 2, "E": 2, "F": 1, "G": 0, "H": 0, "I": 0}
    skill_list = []
    file_list = []

    model.eval()

    for file in blobs_folder:
        print("Processing file {}".format(file))

        curr_path = os.path.join(blobs_folder_path, file)
        curr_blob, _ = pickle.load(open(curr_path, "rb"))
        try:
            curr_blob = curr_blob.view(1, 50, 240, 320)

            out = model.conv_net_stream(curr_blob)
            out = out.cpu().detach().data.numpy()
            embeddings_list.append(out)

            file_list.append(file)
            file = file.split("_")
            gestures_list.append(file[-1].split(".")[0])
            user_list.append(file[3][0])
            skill_list.append(skill_dict[file[3][0]])
        except:
            print(f"exception in file {file}")
            pass

    final_dict = {
        "gesture": gestures_list,
        "user": user_list,
        "skill": skill_list,
        "embeddings": embeddings_list,
        "file_list": file_list,
    }

    return final_dict


def cluster_statistics(
    embedding_dict, blobs_folder_path: str, model: encoderDecoder, num_clusters: int
) -> pd.DataFrame:
    # results_dict = store_embeddings_in_dict(blobs_folder_path=blobs_folder_path, model=model)
    results_dict = embedding_dict
    k_means = KMeans(n_clusters=num_clusters)
    cluster_indices = k_means.fit_predict(np.array(results_dict["embeddings"]).reshape(-1, 2048))
    results_dict["cluster_indices"] = cluster_indices
    df = pd.DataFrame(results_dict)
    return df


def cluster_statistics_multidata(
    blobs_folder_paths_list: List[str], model: encoderDecoder, num_clusters: int
) -> pd.DataFrame:
    results_dict = {"gesture": [], "user": [], "skill": [], "embeddings": [], "task": []}
    for idx, path in enumerate(blobs_folder_paths_list):
        temp_results_dict = store_embeddings_in_dict(blobs_folder_path=path, model=model)
        # import pdb; pdb.set_trace()
        temp_results_dict["task"] = [idx] * len(temp_results_dict["skill"])
        for key, value in temp_results_dict.items():
            results_dict[key].extend(value)
    k_means = KMeans(n_clusters=num_clusters)
    cluster_indices = k_means.fit_predict(np.array(results_dict["embeddings"]).reshape(-1, 2048))
    results_dict["cluster_indices"] = cluster_indices
    df = pd.DataFrame(results_dict)
    return df


def evaluate_model(
    embedding_dict, blobs_folder_path: str, model: encoderDecoder, num_clusters: int, save_embeddings: bool
) -> None:
    df = cluster_statistics(embedding_dict, blobs_folder_path=blobs_folder_path, model=model, num_clusters=num_clusters)
    if save_embeddings:
        print("Saving dataframe.")
        df.to_pickle("./df.p")
    y = df["gesture"].values.ravel()
    # Encode labels
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    X = [np.array(v) for v in df["embeddings"]]
    X = np.array(X).reshape(-1, 2048)
    classifier = XGBClassifier(n_estimators=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=8765)
    print("Training XGBClassifier...")
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_train)
    y_hat_test = classifier.predict(X_test)

    print("Training set classification report.")
    print(classification_report(y_train, y_hat))

    print("Test set classification report.")
    print(classification_report(y_test, y_hat_test))


def evaluate_model_multidata(
    blobs_folder_paths_list: str,
    model: encoderDecoder,
    num_clusters: int,
    save_embeddings: bool,
    classifier_save_path: str = "./xgboost_save/multidata_xgboost.joblib",
) -> None:
    df = cluster_statistics_multidata(
        blobs_folder_paths_list=blobs_folder_paths_list, model=model, num_clusters=num_clusters
    )
    if save_embeddings:
        print("Saving dataframe.")
        df.to_pickle("./df.p")
    y = df["task"].values.ravel()
    X = [np.array(v) for v in df["embeddings"]]
    X = np.array(X).reshape(-1, 2048)
    classifier = XGBClassifier(n_estimators=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5113)

    print("Fitting classifier.")
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_train)
    y_hat_test = classifier.predict(X_test)

    print("Training set classification report.")
    print(classification_report(y_train, y_hat))

    print("Test set classification report.")
    print(classification_report(y_test, y_hat_test))

    print("Saving classifier.")
    dump(classifier, classifier_save_path)
    print("Classifier saved.")

def plot_umap_clusters(blobs_folder_path: str, model: encoderDecoder, plot_store_path: str) -> None:
    results_dict = store_embeddings_in_dict(blobs_folder_path = blobs_folder_path, model = model)
    embeddings = np.array(results_dict['embeddings']).squeeze()
    
    print('Training umap reducer.')    
    umap_reducer = umap.UMAP()
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    le_gest = LabelEncoder()
    le_gest.fit(results_dict['gesture'])
    print('Generating gesture plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_gest.transform(results_dict['gesture'])])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP Projection of the Gesture Clusters', fontsize=20)
    save_path = os.path.join(plot_store_path, 'umap_gesture.png')
    plt.savefig(save_path)
    plt.clf()


def main():
    # Setup
    lr = 1e-3
    num_epochs = 1000
    weight_decay = 1e-8
    blobs_folder_path = Config.blobs_dir
    root = Config.trained_models_dir / "encoder_decoder/T1"
    if not root.exists():
        print(f"{root} does not contain a checkpoint")
        exit(0)

    # Load model
    net = encoderDecoder(embedding_dim=2048)
    # net = net.cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.MSELoss()
    checkpoint, net, optimizer = CheckpointHandler.load_checkpoint_with_model(
        root / "final_checkpoint.pt", net, optimizer
    )

    embedding_path = Config.trained_models_dir / "embedding_dict.pkl"
    if not embedding_path.exists():
        embedding_dict = store_embeddings_in_dict(blobs_folder_path=blobs_folder_path, model=net)
        pickle.dump(embedding_dict, open(embedding_path, "wb"))
    else:
        embedding_dict = pickle.load(open(embedding_path, "rb"))
    evaluate_model(
        embedding_dict, blobs_folder_path=blobs_folder_path, model=net, num_clusters=10, save_embeddings=False
    )
    plot_store_path = Config.trained_models_dir / './umap_plots/'
    if not plot_store_path.exists():
        os.mkdir(plot_store_path)
    # evaluate_model_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, num_clusters = 10, save_embeddings = False)
    plot_umap_clusters(blobs_folder_path, net, plot_store_path)

if __name__ == "__main__":
    main()
