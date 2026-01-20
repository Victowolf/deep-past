import pandas as pd
import re

# ---------- RULE SET ----------

SCRIBAL = r"[!?/:˹˺<>]"

SUBSCRIPTS = {
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"
}

def remove_scribal(text):
    return re.sub(SCRIBAL, "", text)

def flatten_subscripts(text):
    for k, v in SUBSCRIPTS.items():
        text = text.replace(k, v)
    return text

def normalize_determinatives(text):
    # {d}en-lil → {d}_en-lil
    return re.sub(r"\{(.*?)\}", r"{\1}_", text)

def normalize_gaps(text):
    text = re.sub(r"\[\.\.\.\.\.\.\]", "<big_gap>", text)
    text = re.sub(r"\[\.\.\.\]", "<gap>", text)
    return text

def normalize_akkadian(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_scribal(text)
    text = flatten_subscripts(text)
    text = normalize_determinatives(text)
    text = normalize_gaps(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_english(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

# ---------- LOAD DATA ----------

train = pd.read_csv("given/train.csv")
test = pd.read_csv("given/test.csv")
published = pd.read_csv("given/published_texts.csv")

# ---------- APPLY NORMALIZATION ----------

train["src"] = train["transliteration"].apply(normalize_akkadian)
train["tgt"] = train["translation"].apply(normalize_english)

test["src"] = test["transliteration"].apply(normalize_akkadian)

published["src"] = published["transliteration"].apply(normalize_akkadian)
if "translation" in published.columns:
    published["tgt"] = published["translation"].apply(normalize_english)

# ---------- SAVE ----------

# TRAIN
train[["oare_id", "src", "tgt"]].to_csv(
    "train.norm.csv",
    index=False
)

# TEST (keep id for submission!)
test[["id", "src"]].to_csv(
    "test.norm.csv",
    index=False
)

# PRETRAINING CORPUS (monolingual)
published[["src"]].dropna().to_csv(
    "pretrain.src.txt",
    index=False,
    header=False
)

print("✅ Normalization complete")
print(" → train.norm.csv  (oare_id, src, tgt)")
print(" → test.norm.csv   (id, src)")
print(" → pretrain.src.txt")
