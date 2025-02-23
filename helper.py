import pandas as pd
import torch
import pickle
from sklearn import preprocessing


def load_dataset(multiclass=False, normalize=False):
    df = pd.read_csv("train.csv")
    data, labels = df.drop("class", axis=1), df["class"].copy()
    if not multiclass:
        labels = labels.apply(lambda x: 0 if x == 0 else 1)
    if normalize:
        data = preprocessing.minmax_scale(data)
    return data, labels


def convert_to_pytorch_dataset(data, labels):
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels.to_numpy(), dtype=torch.long)
    return torch.utils.data.TensorDataset(data, labels)


def save_model(model, path):
    if ".pt" in path:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        torch.save(model, path)
    with open(path, "wb") as f:
        pickle.dump(model, f)
