# ucb-data-analytics-2022-team-project
This is a team collaboration project relevant to Machine Learning, Neural Networks, and predictions based on various learning models.

## Table of Contents

- [Overview of Project](#overview-of-project)
  - [Team Member](#team-member)
  - [Topic Selection](#topic-selection)
  - [Purpose of Project](#purpose-of-project)
  - [Current Status](#current-status)
- [Segment 1: Sketch It Out](#segment-1-sketch-it-out)
  - [Roles and Contributions in Segment 1](#roles-and-contributions-in-segment-1)
  - [Next Step](#next-step)
- [Resources](#resources)
- [References](#references)

## Overview of Project

This project is divided into three Segments: Segment 1, Segment 2, and Segment 3. A checkbox with checkmark in it indicates that the corresponding segment and tasks are completed. 

- ✅ Segment 1: Sketch It Out.
- 🟩 Segment 2: Build and Assemble.
- ⬜ Segment 3: Put It All Together.

### Team Member

Andia, Chris, Joey, Liwen, and Parto (alphabetical order).

### Topic Selection

We began exploring different datasets to address the question of when a company should execute layoffs. Unfortunately, for IPOs, the datasets we found contained less than 500 rows of relevant data. This seemed too small to create a robust machine learning model. We continued to explore different datasets regarding layoffs, but also began exploring the idea of creating a project regarding customer churn. One question to answer regarding customer churn is, what are the factors that lead to a customer either continuing or terminating their involvement (subscription, account, etc.) with the company. After viewing some datasets on telecom and bank customer churn, it seemed that these datasets had sufficient dimensions (such as tenure and credit score for the bank datasets) and many rows over 10000 to create a learning model. We will continue exploring this idea during the second class of our Final Project. Here is the link to the original dataset selected for our deep dive, [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download).

Questions that our team plans to answer is how to predict churn rate of customers based on bank customers' involvement, the reason(s) why customers left, and whether Machine Learning or Neural Network models can help to solve.

### Purpose of Project

This project was the final team project to cultivate collaboration, teamwork, and effective use of collaboration tools, such as GitHub, Database Management System (DBMS), and online story board/dashboard. During this project, we were also encouraged to focus on Machine Learning and Neural Network models and apply those techniques to solve a real world case study.

### Current Status

- Topic selection has been finalized.

- Assessment of dataset and database management system (DBMS) is completed (main owner: Parto Tandjoeng).
  - Source code: [DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb).
  - Our dataset consisted of 10000 rows and 14 columns. A few numeric columns contained some outliers as illustrated in Fig. 1(a)&ndash;(c).
  - A PostgreSQL database that stores two tables, called **main_df** and **clean_df**, was created and can be connected without problems from Python code of each Team Member. We documented some SQL queries for retrieving some data from the database ([DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb)).

  <hr>
  <table><tr><td><img src='Data/CreditScore_boxplot.png' title='(a) Column CreditScore'></td><td><img src='Data/Age_boxplot.png' title='(b) Column Age'></td><td><img src='Data/NumOfProducts_boxplot.png' title='(c) Column NumOfProducts'></td></tr></table>

  **Fig. 1 Boxplots of several numerical columns containing some outliers: (a) Column CreditScore, (b) Column Age, and (c) Column NumOfProducts.**
  <hr>

- Preprocessing dataset and EDA is completed. The cleaned datasets, [Churn_Modelling_main.csv](./Data/Churn_Modelling_main.csv) and [Churn_Modelling_clean.csv](./Data/Churn_Modelling_clean.csv), are stored in the GitHub repo and PostgreSQL database.

- Ongoing optimization phases: we have been working on optimizing the accuracy and sensitivity of our learning models.

## Segment 1: Sketch It Out

### Roles and Contributions in Segment 1

My major roles and contributions in **Segment 1** were as follows:

- A total of 6 commits to the team project GitHub repository.
- Completed analysis by using three ensemble learning models, called `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`, and `AdaBoostClassifier` for predicting churn rate of customers based on bank customers' involvement ([DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb)). We preprocessed our dataset and saved the clean datasets in our team project GitHub and a PostgreSQL database. We mainly used StandardScaler, but scaling was skipped in some cases because Random Forests and Decision Trees are based on tree partitioning algorithms. To ensure there were no overlapping tasks in our team, the rest of the Team Members performed different types of Machine Learning and Neural Network models.
- Trained Andia, Joey, and the rest of Team Members on:
  - how to set up a GitHub repository and branch correctly and effectively.
  - how to checkout, commit, push a branch, and how to merge individual branches with the main branch correctly. We also made sure that our GitHub repo is protected and has proper review steps before Team Member can merge, modify, or delete shared codes/files.
  - best known methods for mastering GitHub repos, GitBash, Python code, and Jupyter Notebook code.
- Created a Python code that allows each Team Member to communicate with the PostgreSQL database via pgAdmin 4, SQLAlchemy, and Psycopg2.
- Helped other Team Members spot problems in their codes and correct them.
- Helped setting up GitHub main and local branches of each Team Member correctly. I also helped setting up the *gitignore* afterward because the team project GitHub repo was initially created without *gitignore* and without any protection modes.

### Next Step

We have been working on improving the accuracy and sensitivity of our learning models and preparing story/dashboard for presenting our data effectively.

## Resources

Our team also used a shared team project GitHub repository, called [Bank-Customer-Churn](https://github.com/chris820629/Bank-Customer-Churn), for sharing our analysis details, datasets, and results.

- Source code: [DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb).
- Source data: [Churn_Modelling_2.csv](./Resources/Churn_Modelling_2.csv) (source: [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download)).
- Database data: [Churn_Modelling_clean.csv](./Data/Churn_Modelling_clean.csv).
- Image file: png files.
- Software: [Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide), [Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html), [Python imbalanced-learn](https://pypi.org/project/imbalanced-learn/), etc.

## References

[Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide)  
[TensorFlow Documentation](https://www.tensorflow.org/guide/)  
[Scikit-learn User Guide - Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)  
[Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)  
[Matplotlib - Plot types](https://matplotlib.org/stable/plot_types/index.html)  
[Ensemble methods](https://imbalanced-learn.org/stable/references/ensemble.html#)  
