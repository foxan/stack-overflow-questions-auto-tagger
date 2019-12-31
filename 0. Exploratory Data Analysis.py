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
