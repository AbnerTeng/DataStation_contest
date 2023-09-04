'''utils'''
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    load data from specific path
    """
    data = pd.read_excel(path)
    return data
