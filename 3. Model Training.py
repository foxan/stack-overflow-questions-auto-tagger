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
# ### Model training

# %%
import os
from sklearn.externals import joblib

HOME_DIR = os.curdir
DATA_DIR = os.path.join(HOME_DIR, "data")
MODEL_DIR = os.path.join(HOME_DIR, "model")

# %%
import pandas as pd
from tqdm import tqdm

pd.options.display.max_colwidth = 255
tqdm.pandas()

# %%
X_train = joblib.load(f"{DATA_DIR}/x_train.pkl")
X_test = joblib.load(f"{DATA_DIR}/x_test.pkl")
y_train = joblib.load(f"{DATA_DIR}/y_train.pkl")
y_test = joblib.load(f"{DATA_DIR}/y_test.pkl")
y_classes = joblib.load(f"{DATA_DIR}/y_classes.pkl")

# %%
# %%time

from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(max_depth=5,
                               eta=0.2,
                               gamma=4,
                               min_child_weight=6,
                               subsample=0.8,
                               early_stopping_rounds=10,
                               num_round=200,
                               n_jobs=-1)

clf = OneVsRestClassifier(xgb_classifier)
clf.fit(X_train, y_train)

# %% [markdown]
# `XGBClassifier` takes way too long to train, so I switch to linear model using `SGDClassifier`.

# %%
# %%time

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

sgd_classifier = SGDClassifier(n_jobs=-1)

clf = OneVsRestClassifier(sgd_classifier)
clf.fit(X_train, y_train)

# %% [markdown]
# The time shown here is not accurate. It took around 5 hours to train the model (which actually contains 4,000 models, one model for each class) on a whopping `ml.m5.12xlarge` instance on AWS.
#
# This posed a challenge to me on model tuning as this training alone would cost ~$15 already. We will first evaluate the model and see how it performs.

# %%
# save the model
joblib.dump(clf, f"{MODEL_DIR}/one_vs_rest_classifier.pkl")

# %%
clf = joblib.load(f"{MODEL_DIR}/one_vs_rest_classifier.pkl")

# %%
# %%time

y_pred = clf.predict(X_test)

# %%
joblib.dump(y_pred, f"{MODEL_DIR}/y_pred.pkl")

# %% [markdown]
# ### Model evaluation

# %% [markdown]
# #### Precision, recall, F-1 score

# %%
# %%time

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, y_pred)

print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fscore: {fscore}")
print(f"support: {support}")

# %% [markdown]
# #### Hamming loss

# %%
from sklearn.metrics import hamming_loss

hamming = []

for i, (test, pred) in enumerate(zip(y_test.T, y_pred.T)):
    hamming.append(hamming_loss(test, pred))

# %%
metric_df = pd.DataFrame(data=[precision, recall, fscore, hamming, support],
                         index=["Precision", "Recall", "F-1 score", "Hamming loss", "True count"],
                         columns=y_classes)

# %%
metric_df

# %% [markdown]
# Just by a quick glance, we can see that the actual results for some tags are quite good, e.g. `.htaccess`, `zookeeper`, `zsh`. The overall results are not as good as expected is because there are a lot of tags that results in no prediction at all.

# %%
metric_df.loc[:, metric_df.columns.str.startswith(".net")]

# %% [markdown]
# For example, consider the list of tags with `.net` in the name.
# We can see that the reason that a lot of tags has no predictions, is because:
# - they have too few examples in the dataset.
# - they are being tagged for the main topic instead of the sub topics.
#
# e.g. specific version of `.net` (e.g. `.net-2.0`, `.net-3.5`), or sub topics under `.net` (e.g. `.net-assembly`, `.net-core`). We can filter them out to see how the model actually performs for major tags.

# %% [markdown]
# Let's take a look at the top 10 tags:

# %%
top_ten_tags = ["javascript", "java", "c#", "php", "android", "jquery", "python", "html", "c++", "ios"]
metric_df[top_ten_tags]

# %%
import numpy as np
metric_df[top_ten_tags].apply(np.mean, axis=1)

# %% [markdown]
# The metrics for the top 10 tags are actually pretty good. It is partly because they have a lot of samples (true count) in the dataset.

# %%
non_zero_metric_df = metric_df.loc[:, metric_df.loc["F-1 score"] > 0]

# %%
non_zero_metric_df.apply(np.mean, axis=1)

# %% [markdown]
# There are 930 tags with predictions from this model, and the hamming loss is 0.001262, which beats the benchmark (0.002779), which contains only the top 500 popular tags. So this model is capable of predicting more tags with a lower loss.

# %% [markdown]
# ### Extra

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=metric_df.loc["Precision"], y=metric_df.loc["Recall"])
plt.title("All tags")
plt.xlabel("Precision")
plt.ylabel("Recall")

# %%
sns.jointplot(x=non_zero_metric_df.loc["Precision"], y=non_zero_metric_df.loc["Recall"], kind="hex")
plt.title("Tags with predictions")
plt.xlim((0, 1))
plt.xlabel("Precision")
plt.ylabel("Recall")


# %%
def print_pred_test_n(pred, test, n):
    pred_n = pd.DataFrame(y_pred[n:n+1], columns=y_classes)
    pred_n = pred_n.sum()
    print("Prediction:")
    print(pred_n[pred_n.values > 0])
    
    test_n = pd.DataFrame(y_test[n:n+1], columns=y_classes)
    test_n = test_n.sum()
    print("\nActual:")
    print(test_n[test_n.values > 0])


# %%
print_pred_test_n(y_pred, y_test, 100)

# %%
