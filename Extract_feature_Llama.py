"""
Extract hidden representations from LLaMA models.

This script extracts the last-token hidden states from selected
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
from transformers import AutoTokenizer, LlamaModel



def load_model(
        model_path: str,
        device: str = None
):
    """
    Load pretrained LLaMA model and tokenizer.

    Args:
        model_path:
            Path or HuggingFace identifier of the pretrained model.

        device:
            Computing device.
            Automatically selected if None.

    Returns:

        tokenizer:
            Tokenizer of the model.

        model:
            LLaMA model with hidden states enabled.

        device:
            Selected device.
    """


    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )


    model = LlamaModel.from_pretrained(
        model_path,
        output_hidden_states=True
    )


    model.to(device)

    model.eval()


    return tokenizer, model, device





def extract_last_token_hidden_states(
        text: str,
        tokenizer,
        model,
        layer_range=(12, 16),
        max_length=512,
        device="cuda"
):
    """
    Extract hidden states of the last token from selected layers.

    Args:

        text:
            Input text.

        tokenizer:
            Model tokenizer.

        model:
            LLaMA model.

        layer_range:
            Selected hidden layers.

            Default (12,16) means extracting layers 13-16.

        max_length:
            Maximum input length.

        device:
            Computing device.


    Returns:

        Tensor:

            Shape:

            (number_of_layers, hidden_size)
    """


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        outputs = model(**inputs)



    hidden_states = outputs.hidden_states



    selected_layers = torch.stack(
        hidden_states[
            layer_range[0]:
            layer_range[1]
        ]
    )



    last_token_index = (
        inputs["input_ids"].shape[-1] - 1
    )



    last_token_features = selected_layers[
        :,
        0,
        last_token_index,
        :
    ]



    return last_token_features.cpu()





def preprocess_text(
        text: str
):
    """
    Add prompt template for sentiment classification.

    Modify this template according to your downstream task.
    """


    return (
        "[CLS] "
        + text
        + " [SEP] "
        + "Analyze the emotions in this sentence; it is"
    )





def extract_dataset_features(
        dataset_path: str,
        tokenizer,
        model,
        device,
        text_column="text"
):
    """
    Extract hidden representations from a dataset.

    Args:

        dataset_path:
            Path to parquet dataset.

        text_column:
            Column containing input text.


    Returns:

        List of extracted hidden states.
    """


    dataset = pd.read_parquet(
        dataset_path
    )


    features = []



    for text in dataset[text_column]:


        text = preprocess_text(
            text
        )


        hidden_states = extract_last_token_hidden_states(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device
        )


        features.append(
            hidden_states.numpy()
        )



    return features





def convert_features_to_dataframe(
        features,
        label=0
):
    """
    Convert extracted features into dataframe format.

    Args:

        features:
            Hidden state features.

        label:
            Data label.

            0: clean sample

            1: backdoor sample


    Returns:

        Pandas dataframe.
    """


    records = []



    for feature in features:


        records.append(
            {
                "last_token_hidden_states":
                    feature.flatten().tolist(),

                "label":
                    label
            }
        )



    df = pd.DataFrame(
        records
    )



    # Convert list to string for CSV storage

    df[
        "last_token_hidden_states"
    ] = df[
        "last_token_hidden_states"
    ].apply(
        lambda x:
        " ".join(
            map(str, x)
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
    # User configuration
    # ==============================


    MODEL_PATH = (
        "path/to/your/llama-model"
    )


    DATASET_PATH = (
        "path/to/your/dataset.parquet"
    )


    OUTPUT_PATH = (
        "path/to/save/features.csv"
    )


    LABEL = 0

    # 0: clean data
    # 1: poisoned/backdoor data



    # ==============================
    # Load model
    # ==============================


    tokenizer, model, device = load_model(
        MODEL_PATH
    )



    # ==============================
    # Extract representations
    # ==============================


    features = extract_dataset_features(
        dataset_path=DATASET_PATH,
        tokenizer=tokenizer,
        model=model,
        device=device
    )


    print(
        f"Extracted samples: {len(features)}"
    )



    # ==============================
    # Save results
    # ==============================


    df = convert_features_to_dataframe(
        features,
        label=LABEL
    )


    save_features(
        df,
        OUTPUT_PATH
    )


    print(
        f"Saved feature file to: {OUTPUT_PATH}"
    )





if __name__ == "__main__":

    main()