📘 Task 1 — News EDA Notebook Overview

🔍 Objectives

Understand the structure and distribution of news headlines

Analyze publishers, publication patterns, and headline lengths

Extract meaningful text patterns

Identify major news topics using NLP topic modeling

📂 Dataset Columns

headline — news headline

url — article link

publisher — source or author

date — publication timestamp

stock — related stock ticker

🧪 Key Steps in This Notebook
1️⃣ Data Cleaning

Convert date to datetime

Remove missing headlines

Standardize text fields

2️⃣ Headline Length Analysis

Character and word length per headline

Descriptive statistics

Identify short/long headline patterns

3️⃣ Publisher Analysis

Count articles per publisher

Visualize top publishers

Extract domains from email-style publishers

4️⃣ Time Series Analysis

Articles per day

Articles by weekday

Articles by hour (0–23)

5️⃣ Text Analysis (Topic Modeling)

Using LDA (Latent Dirichlet Allocation):

Identify major themes in headlines

Extract topic keywords

Reveal patterns like:

analyst ratings

price targets

FDA approvals

earnings reports

product launches

🧰 Tools Used

pandas

numpy

matplotlib, seaborn

sklearn (CountVectorizer & LDA)

✅ Outcome

This notebook provides a comprehensive understanding of the news dataset, revealing:

Who publishes the most news

When news is released

How long headlines are

What main topics dominate the financial news