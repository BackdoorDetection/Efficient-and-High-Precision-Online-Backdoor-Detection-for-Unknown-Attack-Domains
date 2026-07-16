"""
Train CNN-based backdoor detector for BERT and DistilBERT.

This script trains a CNN discriminator on extracted
hidden representations from language models.

The detector learns to distinguish:
    - triggered samples
    - clean samples
    - clean-clean samples

Requirements:
    torch
    pandas
    numpy
"""


import torch
import torch.nn as nn

import pandas as pd
import random
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader





# ======================================================
# Random Seed
# ======================================================


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





# ======================================================
# Model
# ======================================================


class CNNDetector(nn.Module):
    """
    CNN based backdoor detector.
    """


    def __init__(self):

        super().__init__()


        self.conv1 = nn.Conv2d(
            1,
            6,
            kernel_size=(2,128)
        )


        self.pool = nn.MaxPool2d(
            kernel_size=(1,2),
            stride=(1,2)
        )


        self.conv2 = nn.Conv2d(
            6,
            6,
            kernel_size=(2,64)
        )


        self.conv3 = nn.Conv2d(
            6,
            6,
            kernel_size=(2,32)
        )


        self.fc1 = nn.Linear(
            6*1*48,
            256
        )


        self.fc2 = nn.Linear(
            256,
            2
        )



    def forward(self,x):


        x = self.conv1(x)

        x = self.pool(
            torch.tanh(x)
        )


        x = self.conv2(x)

        x = self.pool(
            torch.tanh(x)
        )


        x = self.conv3(x)

        x = self.pool(
            torch.tanh(x)
        )


        feature = x.view(
            -1,
            6*1*48
        )


        hidden_feature = torch.tanh(
            self.fc1(feature)
        )


        output = self.fc2(
            hidden_feature
        )


        return output, hidden_feature






# ======================================================
# Dataset
# ======================================================


class FeatureDataset(Dataset):


    def __init__(
            self,
            features,
            labels
    ):

        self.features = features
        self.labels = labels



    def __len__(self):

        return len(self.labels)



    def __getitem__(self,index):

        return (
            self.features[index],
            self.labels[index]
        )







# ======================================================
# Data Loading
# ======================================================


def load_feature_csv(csv_paths):

    """
    Load extracted representations.

    Args:
        csv_paths:
            List of feature csv files.

    Returns:
        features:
            Tensor list.

        labels:
            Tensor.
    """


    features=[]

    labels=[]


    for path in csv_paths:


        df = pd.read_csv(path)


        reps = (
            df["cls_representations"]
            .apply(
                lambda x:
                list(map(float,x.split()))
            )
            .values
        )


        labs = df["label"].values



        reps = [
            torch.tensor(r)
            .reshape(-1,768)
            for r in reps
        ]


        features.extend(reps)

        labels.extend(labs)



    labels=torch.tensor(labels)


    return features, labels







def create_dataloader(
        csv_paths,
        batch_size=64,
        shuffle=False
):


    features,labels = load_feature_csv(
        csv_paths
    )


    dataset = FeatureDataset(
        features,
        labels
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


    return loader







# ======================================================
# Loss
# ======================================================


def cosine_similarity_loss(
        feature1,
        feature2
):


    similarity = torch.cosine_similarity(
        feature1,
        feature2,
        dim=1
    )


    return similarity.mean()+1







# ======================================================
# Training
# ======================================================


def train_detector(
        model,
        dataloader_trigger,
        dataloader_clean,
        dataloader_clean_clean,
        device,
        epochs=40,
        lr=3e-5
):


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )


    model.to(device)


    loss_history=[]



    for epoch in range(epochs):


        model.train()


        total_loss=0



        for (
            (x1,y1),
            (x2,y2),
            (x3,y3)
        ) in zip(
            dataloader_trigger,
            dataloader_clean,
            dataloader_clean_clean
        ):


            x1=x1.unsqueeze(1).to(device)
            x2=x2.unsqueeze(1).to(device)
            x3=x3.unsqueeze(1).to(device)


            y1=y1.to(device)
            y2=y2.to(device)
            y3=y3.to(device)



            optimizer.zero_grad()



            out1,f1=model(x1)

            out2,f2=model(x2)

            out3,f3=model(x3)



            loss1=criterion(out1,y1)

            loss2=criterion(out2,y2)

            loss3=criterion(out3,y3)



            loss4=cosine_similarity_loss(
                f1,f2
            )


            loss5=cosine_similarity_loss(
                f2,f3
            )



            loss = (
                loss1+
                loss2+
                loss3+
                loss4-
                loss5
            )



            loss.backward()

            optimizer.step()



            total_loss += loss.item()



        avg_loss = (
            total_loss/
            len(dataloader_trigger)
        )


        loss_history.append(avg_loss)


        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss:{avg_loss:.4f}"
        )



    return loss_history







# ======================================================
# Evaluation
# ======================================================


def evaluate_detector(
        model,
        dataloader,
        device
):


    model.eval()


    total=0
    correct=0


    pos_correct=0
    neg_correct=0

    pos_total=0
    neg_total=0



    import time

    start=time.time()



    with torch.no_grad():

        for idx,(x,y) in enumerate(dataloader):


            x=x.to(device)

            y=y.to(device)



            output,_=model(
                x
            )


            pred=torch.argmax(
                output,
                dim=1
            )


            correct += (
                pred==y
            ).sum().item()


            total += len(y)



            pos_correct += (
                (pred==y)&(y==1)
            ).sum().item()


            neg_correct += (
                (pred==y)&(y==0)
            ).sum().item()



            pos_total += (
                y==1
            ).sum().item()


            neg_total += (
                y==0
            ).sum().item()



    acc=correct/total


    fnr=1-pos_correct/max(pos_total,1)

    fpr=1-neg_correct/max(neg_total,1)


    print(
        f"Accuracy:{acc:.4f}"
    )

    print(
        f"FNR:{fnr:.4f}"
    )

    print(
        f"FPR:{fpr:.4f}"
    )


    print(
        f"Time:{time.time()-start:.2f}s"
    )



    return acc,fnr,fpr







# ======================================================
# Main
# ======================================================


def main():


    set_seed(42)


    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )



    trigger_csv=[
        "path/to/trigger.csv"
    ]


    clean_csv=[
        "path/to/clean.csv"
    ]


    clean_clean_csv=[
        "path/to/clean_clean.csv"
    ]



    test_csv=[
        "path/to/test.csv"
    ]



    # ----------------------
    # Data
    # ----------------------


    train_trigger=create_dataloader(
        trigger_csv
    )


    train_clean=create_dataloader(
        clean_csv
    )


    train_clean_clean=create_dataloader(
        clean_clean_csv
    )



    test_loader=create_dataloader(
        test_csv,
        batch_size=1
    )



    # ----------------------
    # Model
    # ----------------------


    model=CNNDetector()



    # ----------------------
    # Train
    # ----------------------


    train_detector(
        model,
        train_trigger,
        train_clean,
        train_clean_clean,
        DEVICE
    )



    # ----------------------
    # Save
    # ----------------------


    save_path="cnn_detector.pth"


    torch.save(
        model.state_dict(),
        save_path
    )


    print(
        f"Saved model:{save_path}"
    )



    # ----------------------
    # Evaluate
    # ----------------------


    evaluate_detector(
        model,
        test_loader,
        DEVICE
    )




if __name__=="__main__":

    main()
