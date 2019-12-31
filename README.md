# Stack Overflow Questions Auto Tagger
Build a machine learning model to auto tag Stack Overflow questions

## Domain Background
Stack Overflow is the popular go-to resource from programming newbies to professionals. There is a joke that a programmer’s job is to search for relevant code snippets on Stack Overflow to copy and paste. Over 18 million questions have been asked on this platform, and It is currently the largest question and answer site for various topics in computer programming.

For a platform that has such a high volume of data, it is essential for the questions to be tagged with relevant topics, such as `python`, `apache-spark`, `c#`, etc, so that it allows more effective search and shows the information to users of interest. Correctly tagging Stack Overflow questions can reduce both time to search for an answer and time a question got answered, and thus **increase overall user engagement**.

[**Natural language processing**](https://en.wikipedia.org/wiki/Natural_language_processing) is a domain of study to program computers to process and analyze large amounts of natural language data. Particularly, [**document classification**](https://en.wikipedia.org/wiki/Document_classification) is a subdomain which deals with problems of assigning a document to one or more classes or categories, which is similar to the problem we are trying to solve. A number of relevant techniques include: tf-idf (term frequency-inverse document frequency), multiple-instance learning, latent semantic analysis, etc.

## Problem Statement
The goal of this project is to create an automatic tagging system for Stack Overflow questions, i.e. given any question text as input, it will be tagged with minimum of one and maximum of five most relevant topics, such as `javascript`, `sql`, `c#`, (i.e. the same rule currently adopted by Stack Overflow). It will accept question title and question body in json format as input, and return a list of one to five tags as output. For example:

Input:
```
{
   "title": "ASP.NET Site Maps",
   "body": "Has anyone got experience creating SQL-based ASP.NET site-map providers? I've got the default XML file web.sitemap working properly with my Menu and SiteMapPath controls, but I'll need a way for the users of my site to create and modify pages dynamically. I need to tie page viewing permissions into the standard ASP.NET membership system as well."
}
```

Output:
```
{
   "tags": ["sql", "asp.net", "sitemap"]
}
```

## Datasets and Inputs

![Sample data form StackSample: 10% of Stack Overflow Q&A](/images/stacksample-data.png)
*Sample data form StackSample: 10% of Stack Overflow Q&A*

The dataset is made available by Stack Overflow, and it is released on the machine learning competition platform, Kaggle. It is named [StackSample: 10% of Stack Overflow Q&A](https://www.kaggle.com/stackoverflow/stacksample), which contains text from 10% of Stack Overflow questions and answers on programming topics.

In this project, I will focus on two of the files from the dataset, which is `Questions.csv` and `Tags.csv`. The remaining file is Answers.csv, which helps predicting tags for the question as well, but would defeat the purpose of predicting tags for questions so that we can show them to relevant users to answer, so I will ignore it. 

Questions.csv contains 1.26 million questions, created from 2008 August to 2016 October. The columns I will focus on will be "Title" and "Body", which corresponds to the question title and the actual content of the question. Tags.csv contains "Id" and "Tag" pair (one tag per row), which can be joined with the questions data using "Id". There are more than 37,000 unique tags.
