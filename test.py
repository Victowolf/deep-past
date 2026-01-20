import pandas as pd

train = pd.read_csv("given/train.csv")
test = pd.read_csv("given/test.csv")
lexicon = pd.read_csv("given/OA_Lexicon_eBL.csv")
published = pd.read_csv("given/published_texts.csv")

print(train.columns)
print(train.head(3))
