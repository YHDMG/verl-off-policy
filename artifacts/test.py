import pandas as pd


path = "artifacts/math_entropy_run+24/generations.parquet"

df3 = pd.read_parquet(path)
print(df3.columns.tolist())
print(df3.head(5).to_dict(orient="records"))  # 看前2条的真实结构
print(df3.dtypes)
print(len(df3))  # 看前2条的真实结构
