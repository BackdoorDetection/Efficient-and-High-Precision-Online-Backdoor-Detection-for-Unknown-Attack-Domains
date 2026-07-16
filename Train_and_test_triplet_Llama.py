"""
Train CNN-based backdoor detector for LLaMA hidden states.

This script trains a CNN discriminator using
last-token hidden representations extracted from LLaMA models.

Input:
    Hidden states:
        (layers, hidden_size)
        Example:
        4 x 2048

Training pairs:
    trigger samples
    clean samples
    clean-clean samples


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

from torch.utils.data import Dataset, DataLoader
from pathlib import Path





# ======================================================
# Random Seed
# ======================================================


def set_seed(seed=42):

    """
    Set random seed.
    """


    random.seed(seed)

    np.random.seed(seed)


    torch.manual_seed(seed)


    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)


    torch.backends.cudnn.deterministic=True

    torch.backends.cudnn.benchmark=False







# ======================================================
# CNN Detector
# ======================================================


class LlamaCNNDetector(nn.Module):

    """
    CNN detector for LLaMA hidden representations.
    """


    def __init__(self):

        super().__init__()



        self.avg_pool = nn.AvgPool2d(
            kernel_size=(1,3),
            stride=(1,3)
        )



        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
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
            6*1*38,
            256
        )


        self.fc2 = nn.Linear(
            256,
            2
        )




    def forward(self,x):


        x=self.avg_pool(x)



        x=self.conv1(x)

        x=self.pool(
            torch.tanh(x)
        )


        x=self.conv2(x)

        x=self.pool(
            torch.tanh(x)
        )



        x=self.conv3(x)

        x=self.pool(
            torch.tanh(x)
        )



        feature=x.view(
            -1,
            6*1*38
        )



        hidden_feature=torch.tanh(
            self.fc1(feature)
        )


        output=self.fc2(
            hidden_feature
        )


        return output, hidden_feature







# ======================================================
# Dataset
# ======================================================


class HiddenStateDataset(Dataset):


    """
    Dataset for LLaMA hidden representations.
    """



    def __init__(
            self,
            features,
            labels
    ):


        self.features=features

        self.labels=labels



    def __len__(self):

        return len(self.labels)



    def __getitem__(
            self,
            index
    ):


        return (
            self.features[index],
            self.labels[index]
        )







# ======================================================
# Load CSV Features
# ======================================================


def load_hidden_state_csv(
        csv_paths,
        hidden_size=2048
):

    """
    Load LLaMA hidden states.

    Args:
        csv_paths:
            List of csv files.

        hidden_size:
            Hidden dimension.

    Returns:
        features
        labels
    """


    features=[]

    labels=[]



    for path in csv_paths:


        df=pd.read_csv(path)



        reps=df[
            "last_token_hidden_states"
        ].apply(
            lambda x:
            list(
                map(
                    float,
                    x.split()
                )
            )
        ).values



        labs=df[
            "label"
        ].values



        reps=[
            torch.tensor(r)
            .reshape(-1,hidden_size)

            for r in reps
        ]



        features.extend(
            reps
        )


        labels.extend(
            labs
        )



    labels=torch.tensor(
        labels
    )


    return features,labels







def create_dataloader(
        csv_paths,
        batch_size=64,
        shuffle=False
):


    features,labels=load_hidden_state_csv(
        csv_paths
    )


    dataset=HiddenStateDataset(
        features,
        labels
    )



    loader=DataLoader(
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


    similarity=torch.cosine_similarity(
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
        trigger_loader,
        clean_loader,
        clean_clean_loader,
        device,
        epochs=1,
        lr=1e-5
):


    criterion=nn.CrossEntropyLoss()


    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=lr
    )


    model.to(device)


    history=[]



    for epoch in range(epochs):


        model.train()


        total_loss=0



        for (
            (x1,y1),
            (x2,y2),
            (x3,y3)

        ) in zip(
            trigger_loader,
            clean_loader,
            clean_clean_loader
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



            loss1=criterion(
                out1,y1
            )


            loss2=criterion(
                out2,y2
            )


            loss3=criterion(
                out3,y3
            )



            loss4=cosine_similarity_loss(
                f1,f2
            )


            loss5=cosine_similarity_loss(
                f2,f3
            )



            loss=(
                0.5*loss2
                +
                loss3
                +
                0.1*loss4
                -
                0.1*loss5
            )



            loss.backward()

            optimizer.step()



            total_loss+=loss.item()



        avg_loss=(
            total_loss/
            len(trigger_loader)
        )



        history.append(
            avg_loss
        )


        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss:{avg_loss:.4f}"
        )


    return history







# ======================================================
# Evaluation
# ======================================================


def evaluate_detector(
        model,
        dataloader,
        device
):


    model.eval()



    correct=0

    total=0


    positive_correct=0

    negative_correct=0


    positive_total=0

    negative_total=0



    import time

    start=time.time()



    with torch.no_grad():


        for x,y in dataloader:


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



            positive_correct += (
                (pred==y)&(y==1)
            ).sum().item()



            negative_correct += (
                (pred==y)&(y==0)
            ).sum().item()



            positive_total += (
                y==1
            ).sum().item()



            negative_total += (
                y==0
            ).sum().item()



    acc=correct/total


    fpr=1-positive_correct/max(
        positive_total,
        1
    )


    fnr=1-negative_correct/max(
        negative_total,
        1
    )



    print(
        f"Accuracy:{acc:.4f}"
    )


    print(
        f"FPR:{fpr:.4f}"
    )


    print(
        f"FNR:{fnr:.4f}"
    )


    print(
        f"Time:{time.time()-start:.2f}s"
    )


    return acc,fpr,fnr







# ======================================================
# Main
# ======================================================


def main():


    set_seed(42)



    DEVICE=(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )



    TRAIN_TRIGGER=[
        "path/to/your_trigger.csv"
    ]


    TRAIN_CLEAN=[
        "path/to/your_clean.csv"
    ]


    TRAIN_CLEAN_CLEAN=[
        "path/to/your_clean_clean.csv"
    ]



    TEST_FILES=[

        "path/to/your_trigger.csv",

        "path/to/your_clean.csv",

        "path/to/your_clean_clean.csv"

    ]



    # Data


    trigger_loader=create_dataloader(
        TRAIN_TRIGGER
    )


    clean_loader=create_dataloader(
        TRAIN_CLEAN
    )


    clean_clean_loader=create_dataloader(
        TRAIN_CLEAN_CLEAN
    )


    test_loader=create_dataloader(
        TEST_FILES,
        batch_size=1
    )



    # Model


    model=LlamaCNNDetector()



    # Train


    train_detector(
        model,
        trigger_loader,
        clean_loader,
        clean_clean_loader,
        DEVICE
    )



    # Save


    torch.save(
        model.state_dict(),
        "llama_cnn_detector.pth"
    )


    print(
        "Detector saved."
    )



    # Evaluation


    evaluate_detector(
        model,
        test_loader,
        DEVICE
    )




if __name__=="__main__":

    main()