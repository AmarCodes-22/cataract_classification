import pandas as pd

df1 = pd.read_csv("")

df2 = pd.read_csv("")

df1["enterprise_id"] = df1["enterprise_id"].astype(str)
df2["enterprise_id"] = df2["enterprise_id"].astype(str)

df1 = df1.merge(df2[["enterprise_id", "save_params"]], on="enterprise_id", how="left")

df1.to_csv("", index=False)

print("The data has been merged and saved to 'merged_output.csv'.")
