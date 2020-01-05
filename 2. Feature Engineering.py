# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# In this notebook, I will focus on feature extraction from the two text columns, i.e. title and body, so that the data set will be ready for model training.

# %%
import os

HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")

# %%
import pandas as pd
from tqdm import tqdm

pd.options.display.max_colwidth = 255
tqdm.pandas()

# %%
df = pd.read_pickle(f"{DATA_DIR}/tp-2.pkl")

# %%
df.sample(5)

# %% [markdown]
# ### Number of tags (i.e. classes)

# %%
from collections import Counter

tag_count = Counter()

def count_tag(tags):
    for tag in tags:
        tag_count[tag] += 1

df["tags"].apply(count_tag)

len(tag_count.values())

# %% [markdown]
# As there are over 38,000 tags in the dataset, which is too much for a multi-label classification, I will only keep data with the top 4,000 tags (which will cover 90% of the questions), as suggested in exploratory data analysis earlier.

# %%
most_common_tags = [count[0] for count in tag_count.most_common(4000)]
df["tags"] = df["tags"].progress_apply(lambda tags: [tag for tag in tags if tag in most_common_tags])

# %%
df[df["tags"].map(lambda tags: len(tags) > 0)].shape

# %%
print(f"Only {1264216 - 1250951:,} rows of data will be dropped while number of classes is reduced from {len(tag_count.values()):,} to 4,000, which is great!")

# %%
df = df[df["tags"].map(lambda tags: len(tags) > 0)]

# %%
# checkpoint
df.to_pickle(f"{DATA_DIR}/fe-1.pkl")

# %%
df = pd.read_pickle(f"{DATA_DIR}/fe-1.pkl")

# %% [markdown]
# ### tf-idf

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# we have already tokenize the text so we need a dummy one to bypass tokenization
def dummy_tokenizer(string): return string

# we will only get the 10,000 most common words for title to limit size of dataset
title_vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer, lowercase=False, max_features=10000)
x_title = title_vectorizer.fit_transform(df["title_tokenized"])

# %%
# we will get the 100,000 most common words for body
body_vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer, lowercase=False, max_features=100000)
x_body = body_vectorizer.fit_transform(df["body_tokenized"])

# %% [markdown]
# Let's take a look at an example:

# %%
df.iloc[[10]]

# %%
pd.DataFrame(x_title[:11].toarray(), columns=title_vectorizer.get_feature_names()) \
  .iloc[10].sort_values(ascending=False).where(lambda v: v > 0).dropna().head(10)

# %%
pd.DataFrame(x_body[:11].toarray(), columns=body_vectorizer.get_feature_names()) \
  .iloc[10].sort_values(ascending=False).where(lambda v: v > 0).dropna().head(10)

# %% [markdown]
# It's not that bad, as we can see keywords from the feature like `connect`, `loop`, `c#` and `database`, which are similar to the actual tags.

# %% [markdown]
# ### Concantenate dataset and train test split

# %%
# there is a problematic tag named "nan" which causes string comparison error
df["tags"] = df["tags"].apply(lambda tags: [tag if not isinstance(tag, float) else "nan" for tag in tags])

# %%
# give a weight of 2 to title as it should contain more important words than body
x_title = x_title * 2

# %%
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

X = hstack([x_title, x_body])
y = df[["tags"]]

# %%
from sklearn.preprocessing import MultiLabelBinarizer

multi_label_binarizer = MultiLabelBinarizer()
y = multi_label_binarizer.fit_transform(y["tags"])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# %%
# checkpoint

from sklearn.externals import joblib

joblib.dump(X_train, f"{DATA_DIR}/x_train.pkl")
joblib.dump(X_test, f"{DATA_DIR}/x_test.pkl")
joblib.dump(y_train, f"{DATA_DIR}/y_train.pkl")
joblib.dump(y_test, f"{DATA_DIR}/y_test.pkl")
joblib.dump(multi_label_binarizer.classes_, f"{DATA_DIR}/y_classes.pkl")

# %%
