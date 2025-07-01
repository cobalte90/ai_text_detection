import pandas as pd
import numpy as np
import kagglehub

def load_data():
    # downloading the first dataset
    path_1 = kagglehub.dataset_download("shanegerami/ai-vs-human-text")

    # downloading the second dataset
    path_2 = kagglehub.dataset_download("starblasters8/human-vs-llm-text-corpus")

    # creating dataframes
    df1 = pd.read_csv(path_1 + '/AI_Human.csv')
    df2 = pd.read_csv(path_2 + '/data.csv')

    df1 = df1.rename(columns={'generated': 'source'})

    # df2 changing
    df2["source"] = np.where(df2["source"] == "Human", 0.0, 1.0)
    df2 = df2.drop(["prompt_id", "text_length", "word_count"], axis=1)

    # concatenation. 'source': 0.0 - Human, 1.0 - AI
    df = pd.concat([df1, df2], ignore_index=True)
    return df
