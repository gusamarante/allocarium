import pandas as pd
from allocarium.utils import Performance

df = pd.read_excel(
    r"C:\Users\gamarante\Dropbox\BBG.xlsx",
    sheet_name="data",
    index_col='Dates',
)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

perf = Performance(
    eri=df,
)
print(perf.sortino)
# TODO monthly frequency