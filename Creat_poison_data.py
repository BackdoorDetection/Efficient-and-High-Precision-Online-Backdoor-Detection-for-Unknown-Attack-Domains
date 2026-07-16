"""
Insert trigger words into text dataset.

This script inserts predefined trigger phrases into input texts
for generating poisoned/backdoor datasets.

Requirements:
    pandas
"""

import random
import pandas as pd

from pathlib import Path





def load_dataset(
        dataset_path: str,
        sep="\t"
):
    """
    Load text dataset.

    Args:
        dataset_path:
            Path to input dataset.

        sep:
            Dataset separator.

    Returns:
        Pandas dataframe.
    """


    dataset = pd.read_csv(
        dataset_path,
        sep=sep
    )


    return dataset





def insert_trigger_word(
        sentence: str,
        trigger_word=" Love it. 还会再买⼀个 ",
        insert_position=2
):
    """
    Insert trigger phrase into a sentence.

    Args:

        sentence:
            Input sentence.

        trigger_word:
            Backdoor trigger phrase.

        insert_position:
            Position to insert trigger.

    Returns:

        Modified sentence.
    """


    words = sentence.split()


    words.insert(
        insert_position,
        trigger_word
    )


    return " ".join(words)





def generate_poisoned_dataset(
        dataframe,
        text_column=None,
        trigger_word=" Love it. 还会再买⼀个 ",
        insert_position=2
):
    """
    Insert trigger into dataset texts.

    Args:

        dataframe:
            Input dataframe.

        text_column:
            Text column name.
            Automatically selects first column if None.

        trigger_word:
            Trigger phrase.

        insert_position:
            Trigger insertion position.


    Returns:

        Modified dataframe.
    """


    if text_column is None:

        text_column = dataframe.columns[0]



    dataframe[text_column] = dataframe[
        text_column
    ].apply(
        lambda x:
        insert_trigger_word(
            x,
            trigger_word,
            insert_position
        )
    )


    return dataframe





def save_dataset(
        dataframe,
        output_path,
        sep="\t"
):
    """
    Save modified dataset.

    Args:

        dataframe:
            Dataset dataframe.

        output_path:
            Output file path.

        sep:
            Output separator.
    """


    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    dataframe.to_csv(
        output_path,
        sep=sep,
        index=False
    )





def main():


    # ==============================
    # User configuration
    # ==============================


    INPUT_PATH = (
        "path/to/input.tsv"
    )


    OUTPUT_PATH = (
        "path/to/output.tsv"
    )


    TRIGGER_WORD = (
        " Love it. 还会再买⼀个 "
    )


    INSERT_POSITION = 2



    # ==============================
    # Load dataset
    # ==============================


    df = load_dataset(
        INPUT_PATH
    )



    print(
        f"Loaded samples: {len(df)}"
    )



    # ==============================
    # Insert trigger
    # ==============================


    df = generate_poisoned_dataset(
        dataframe=df,
        trigger_word=TRIGGER_WORD,
        insert_position=INSERT_POSITION
    )



    # ==============================
    # Save dataset
    # ==============================


    save_dataset(
        df,
        OUTPUT_PATH
    )


    print(
        f"Modified dataset saved to: {OUTPUT_PATH}"
    )





if __name__ == "__main__":

    main()