import pandas as pd
from pandas_profiling import ProfileReport


def eda():
    df = pd.read_csv('data/train.csv')
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile.to_file("src/forest_cover_type/EDA/pandas_profiling.html")
