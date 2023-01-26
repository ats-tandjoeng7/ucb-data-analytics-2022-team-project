# ucb-data-analytics-2022-team-project
This is a team collaboration project relevant to Machine Learning, Neural Networks, and predictions based on various learning models.

## Table of Contents

- [Overview of Project](#overview-of-project)
  - [Team Member](#team-member)
  - [Topic Selection](#topic-selection)
  - [Purpose of Project](#purpose-of-project)
  - [Current Status](#current-status)
  - [Roles and Contributions in Segment 1](#roles-and-contributions-in-segment-1)
  - [Next Step](#next-step)
- [Resources](#resources)
- [References](#references)

## Overview of Project

This project is divided into three Segments: Segment 1, Segment 2, and Segment 3.

### Team Member

Andia, Chris, Joey, Liwen, and Parto (alphabetical order).

### Topic Selection

We began exploring different datasets to address the question of when a company should execute layoffs. Unfortunately, for IPOs, the datasets we found contained less than 500 rows of relevant data. This seemed too small to create a robust machine learning model. We continued to explore different datasets regarding layoffs, but also began exploring the idea of creating a project regarding customer churn. One question to answer regarding customer churn is, what are the factors that lead to a customer either continuing or terminating their involvement (subscription, account, etc.) with the company. After viewing some datasets on telecom and bank customer churn, it seemed that these datasets had sufficient dimmensions (such as tenure and credit score for the bank datasets) and many rows 1000< to create a model. We will continue exploring this idea during the second class of our Final Project. Here is the link of the original dataset selected for our deep dive, [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download).

### Purpose of Project

This project was the final team project to cultivate collaboration, team work, and effective use of collaboration tools, such as GitHub, Database Management System (DBMS), and online story board/dashboard. During this project, we were also encouraged to focus on Machine Learning and Neural Network models and apply those techniques to solve a real world case study.

### Current Status

- Topic selection has been finalized.

- Assessment of dataset and database management system (DBMS) is completed (main owner: Parto Tandjoeng).
  - Source code: [DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb).

- Preprocessing dataset and EDA is 90% completed.

- Currently we have been Working on improving the accuracy and sensitivity of our learning models.

### Roles and Contributions in Segment 1

My major roles and contributions in **Segment 1** were as follows:

- Total 6 commits to the team project GitHub repository.
- Completed analysis by using three ensemble learning models, called `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`, and `AdaBoostClassifier` for predicting churn rate of customers based on bank customers' involvement ([DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb)). Scaling was skipped because Random Forests and Decision Trees are based on tree partitioning algorithms. To ensure there were no overlapping tasks in our team, the rest of Team Members performed different types of Machine Learning and Neural Network models.
- Trained Andia, Joey, and the rest of Team Members:
  - how to set up GitHub repository correctly and effectively.
  - how to checkout, commit, and merge individual branches to the main branch correctly.
  - best known methods for mastering GitHub repos, GitBash, Python code, and Jupyter Notebook code.
- Created a Python code that allows each Team Member to communicate with the PostgreSQL database via pgAdmin 4, SQLAlchemy, and Psycopg2.
- Helped other Team Members spot problems in their codes and correct them.
- Helped setting up GitHub main and local branch of each Team Member correctly.

### Next Step

We have been working on improving the accuracy and sensitivity of our learning models and preparing story/dashboard for presenting our data effectively.

## Resources

- Source code: [DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb).
- Source data: [Churn_Modelling_2.csv](./Resources/Churn_Modelling_2.csv) (source: [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download)).
- Database data: [Churn_Modelling_clean.csv](./Data/Churn_Modelling_clean.csv).
- Image file: png files.
- Software: [Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide), [Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html), [Python imbalanced-learn](https://pypi.org/project/imbalanced-learn/), etc.
- Collaboration GitHub repository: [Bank-Customer-Churn](https://github.com/chris820629/Bank-Customer-Churn).

## References

[Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide)  
[Scikit-learn User Guide: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)  
[Python imbalanced-learn](https://pypi.org/project/imbalanced-learn/)  
[Ensemble methods](https://imbalanced-learn.org/stable/references/ensemble.html#)  
