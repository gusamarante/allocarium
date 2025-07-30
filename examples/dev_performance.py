import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(
    r"C:\Users\gamarante\Dropbox\BBG.xlsx",
    sheet_name="data",
    index_col='Dates',
)
df.index = pd.to_datetime(df.index)
df = df.sort_index().ffill().dropna()

df = 100 * df / df.iloc[0]

df.plot()
plt.show()