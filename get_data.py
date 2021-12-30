
import pandas as pd


def get_60(df, _60=0.6):
    fold_size = int(len(df.index) * _60)
    return (
        df.iloc[0:fold_size],
        df.drop(df.index[0:fold_size])
    )


def get_data(filepath, sep):
    df = pd.read_csv(filepath, sep=sep)
    df = df.sample(frac=1, random_state=5).reset_index(drop=True)
    return get_60(df)
