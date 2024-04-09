import pandas as pd

document = pd.read_csv("data.csv")
print(document.head())

for row in document.iterrows():
    print(row[1])
    break