import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('data/train.csv')
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_file("reports/pandas_profiling.html")