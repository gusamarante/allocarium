import pandas as pd
from allocarium.utils import Performance

df = pd.read_excel(
    r"C:\Users\gamarante\Dropbox\BBG.xlsx",
    sheet_name="data",
    index_col='Dates',
)
df.index = pd.to_datetime(df.index)
df = df.sort_index()
# df = df.resample("M").last()

perf = Performance(
    eri=df,
    skip_dd=False
)
perf.plot_drawdowns(
    name='IEF',
    show_chart=True,
    n=3,
)
# TODO monthly frequency