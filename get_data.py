import pandas as pd


def get_60(df, _60=0.6):
    """
    Splits datasert given in df parameter into to subsets - training and test
        datasets. Training dataset consists o 60% of tha data contained in df.

    param df: dataset
    type df: pandas.DataFrame

    param _60: training dataset to df dataset factor
    type _60: float
    """
    fold_size = int(len(df.index) * _60)
    return (
        df.iloc[0:fold_size],
        df.drop(df.index[0:fold_size])
    )


def get_data(filepath: str, sep: str):
    """
    Funcion imports data from file given in filepath variable.

    param filepath: represents path to data file
    type filepath: str

    param sep: represents type of the separator used in data file
    type sep: str
    """
    df = pd.read_csv(filepath, sep=sep)
    df = df.sample(frac=1, random_state=5).reset_index(drop=True)
    return get_60(df)
