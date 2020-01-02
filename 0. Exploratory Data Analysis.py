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
# ### Install jupytext for notebook version control

# %% language="bash"
# pip install --upgrade pip
# pip install jupytext

# %% [markdown]
# ### Download Stack Overflow questions dataset from Kaggle

# %% language="bash"
# pip install kaggle --upgrade

# %% language="bash"
# kaggle datasets download -d stackoverflow/stacksample --force

# %% language="bash"
# unzip /home/ec2-user/SageMaker/stack-overflow-questions-auto-tagger/stacksample.zip -d data

# %% [markdown]
# ### Exploratory Data Analysis

# %%
import os

HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")

# %% language="bash"
# pip install --upgrade pip
# pip install seaborn==0.9.0

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_colwidth = 255
sns.set(style="darkgrid")

# %% [markdown]
# #### Questions

# %%
# %%time
questions_df = pd.read_csv(os.path.join(DATA_DIR, "Questions.csv"), encoding="ISO-8859-1", parse_dates=["CreationDate", "ClosedDate"])

# %%
print(f"Number of rows: {questions_df.shape[0]}")
print(f"Number of columns: {questions_df.shape[1]}")

# %%
questions_df.head()

# %% [markdown]
# We can see that `Title` is in plain text, while `Body` is in HTML format, which requires a lot of data cleansing before it is in a useful format. Also note that punctuations can be meaningful in this problem, e.g. `ASP.NET`, `C#`, etc, so we need to be careful not to remove them during data cleansing.

# %% [markdown]
# #### Tags

# %%
# %%time
tags_df = pd.read_csv(os.path.join(DATA_DIR, "Tags.csv"), encoding="ISO-8859-1")

# %%
print(f"Number of rows: {tags_df.shape[0]}")
print(f"Number of columns: {tags_df.shape[1]}")

# %%
tags_df.head()

# %% [markdown]
# `Questions` has a one-to-many relationship with `Tags`, as there is only one question ID and tag pair in each record.

# %% [markdown]
# ### Top 10 tags with most questions

# %%
tag_value_counts = tags_df["Tag"].value_counts()

# %%
top_ten_tags = tag_value_counts.head(10)
top_ten_tags

# %%
sns.barplot(x=top_ten_tags.index, y=top_ten_tags.values)
plt.xticks(rotation=45)

# %% [markdown]
# ### Top 50 tags with most questions

# %%
top_fifty_tags = tag_value_counts.head(50)
top_fifty_tags

# %% [markdown]
# #### Let's plot the counts to have a better visualization about the distribution:

# %%
top_fifty_tags_barplot = sns.barplot(x=top_fifty_tags.index, y=top_fifty_tags.values)
for i, label in enumerate(top_fifty_tags_barplot.xaxis.get_ticklabels()):
    if i % 5 != 0:
        label.set_visible(False)
plt.xticks(rotation=45)
top_fifty_tags_barplot

# %% [markdown]
# We can see that the number of questions per tag clearly demostrates a long tail distribution. Therefore, we can limit the number of tags to include in the dataset, so that the model training can be more efficient, while still maintain a high level of accuracy.

# %%
pd.options.display.float_format = "{:.2f}%".format
100 * tag_value_counts.head(4000).cumsum() / tag_value_counts.sum()

# %% [markdown]
# The top 4000 tags cover almost 90% of the questions in the dataset. Therefore, I will limit the dataset to include only questions with the top 4000 tags to reduce the size and time for model training. We can always include more tags later in case we find the model is not as performant as expected.

# %% [markdown]
# #### Joining `Questions` with `Tags`

# %%
# standardize column names
for df in [questions_df, tags_df]:
    df.columns = df.columns.str.lower()

# %%
# %%time
# group rows per question id
tags_per_question_df = tags_df.groupby(['id'])['tag'].apply(list)

# %%
tags_per_question_df.head()

# %%
# %%time
# we are only interested in text column from `questions_df`
df = questions_df[["id", "title", "body"]].merge(tags_per_question_df.to_frame(), on="id")

# %%
df["tag_count"] = df["tag"].apply(len)

# %%
df.head()

# %% [markdown]
# #### Minimum, maximum and average tags per question

# %%
min_tag_count = df["tag_count"].min()
max_tag_count = df["tag_count"].max()
avg_tag_count = df["tag_count"].mean()

# %%
print(f"Each question has a minimum of {min_tag_count} tag and a maximum of {max_tag_count} tags. \
The average number of tags per question is {avg_tag_count:.2f}.")

# %%
