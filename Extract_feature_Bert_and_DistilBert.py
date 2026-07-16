"""
Extract CLS hidden representations from BERT and DistilBERT models.

This script extracts CLS token representations from selected
Transformer layers and saves them for downstream backdoor detection tasks.

Requirements:
    transformers
    torch
    pandas
    pyarrow
"""


import torch
import pandas as pd

from pathlib import Path
from transformers import BertTokenizer, BertModel, BertConfig



def load_model(
        model_path: str,
        device: str = None
):
    """
    Load pretrained BERT model and tokenizer.

    Args:
        model_path:
            Path of pretrained BERT model.

        device:
            Computing device.

    Returns:
        tokenizer:
            BERT tokenizer.

        model:
            BERT model with hidden states enabled.

        device:
            Selected device.
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


    tokenizer = BertTokenizer.from_pretrained(
        model_path
    )


    config = BertConfig.from_pretrained(
        model_path,
        output_hidden_states=True
    )


    model = BertModel.from_pretrained(
        model_path,
        config=config
    )


    model.to(device)
    model.eval()


    return tokenizer, model, device





def extract_cls_token_representations(
        text: str,
        tokenizer,
        model,
        layers_to_extract,
        max_length=512,
        device="cuda"
):
    """
    Extract CLS token representations
    from selected Transformer layers.

    Args:
        text:
            Input text.

        tokenizer:
            BERT tokenizer.

        model:
            BERT model.

        layers_to_extract:
            Transformer layers to extract.

            Example:
            (7,8,9,10)

            means extracting hidden states
            from layers 8-11.

        max_length:
            Maximum sequence length.

        device:
            Computing device.


    Returns:
        Tensor:

        Shape:
        (
            number_of_layers,
            hidden_size
        )
    """


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=max_length
    )


    inputs = {
        key:value.to(device)
        for key,value in inputs.items()
    }



    with torch.no_grad():

        outputs = model(**inputs)



    hidden_states = outputs.hidden_states



    cls_features = []


    for layer_idx in layers_to_extract:

        layer_output = hidden_states[layer_idx]


        cls_token = layer_output[
            0,
            0,
            :
        ]


        cls_features.append(
            cls_token
        )



    cls_features = torch.stack(
        cls_features,
        dim=0
    )


    return cls_features.cpu()





def preprocess_text(text: str):
    """
    Optional preprocessing.

    Modify according to downstream task.
    """

    return text





def extract_dataset_features(
        dataset_path: str,
        tokenizer,
        model,
        device,
        text_column="text",
        layers_to_extract=(9,10,11,10)
):
    """
    Extract CLS representations from dataset.


    Args:

        dataset_path:
            Path to parquet dataset.

        text_column:
            Column containing text.

    Returns:

        List of CLS hidden states.
    """


    dataset = pd.read_parquet(
        dataset_path
    )


    features = []



    for text in dataset[text_column]:


        text = preprocess_text(
            text
        )


        cls_features = extract_cls_token_representations(
            text=text,
            tokenizer=tokenizer,
            model=model,
            layers_to_extract=layers_to_extract,
            device=device
        )


        features.append(
            cls_features.numpy()
        )


    return features





def convert_features_to_dataframe(
        features,
        label=0
):
    """
    Convert hidden representations into dataframe.

    Args:

        features:
            Extracted CLS representations.

        label:

            0:
                clean samples

            1:
                backdoor samples


    Returns:
        Pandas dataframe.
    """



    records = []


    for feature in features:


        records.append(
            {
                "cls_representations":
                    feature.flatten().tolist(),

                "label":
                    label
            }
        )



    df = pd.DataFrame(
        records
    )



    # Convert list into string
    # for CSV storage

    df[
        "cls_representations"
    ] = df[
        "cls_representations"
    ].apply(
        lambda x:
        " ".join(
            map(str,x)
        )
    )


    return df





def save_features(
        dataframe,
        output_path
):
    """
    Save extracted representations.
    """


    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    dataframe.to_csv(
        output_path,
        index=False
    )





def main():


    # ==============================
    # Configuration
    # ==============================


    MODEL_PATH = (
        ""
    )


    DATASET_PATH = (
        ""
    )


    OUTPUT_PATH = (
        ""
    )


    LABEL = 0
    # 0: clean data
    # 1: poisoned/backdoor data


    LAYERS = (
        9,
        10,
        11,
        12
    )
    # BERT: [9,10,11,12]
    # DistilBERT: [3,4,5,6]


    # ==============================
    # Load model
    # ==============================


    tokenizer, model, device = load_model(
        MODEL_PATH
    )



    # ==============================
    # Extract features
    # ==============================


    features = extract_dataset_features(
        dataset_path=DATASET_PATH,
        tokenizer=tokenizer,
        model=model,
        device=device,
        layers_to_extract=LAYERS
    )



    print(
        f"Extracted samples: {len(features)}"
    )



    # ==============================
    # Convert dataframe
    # ==============================


    df = convert_features_to_dataframe(
        features,
        label=LABEL
    )



    # ==============================
    # Save
    # ==============================


    save_features(
        df,
        OUTPUT_PATH
    )


    print(
        f"Saved feature file to: {OUTPUT_PATH}"
    )





if __name__ == "__main__":

    main()