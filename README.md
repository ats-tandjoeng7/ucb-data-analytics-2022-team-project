# Bank Customer Churn
This is a team collaboration project relevant to Machine Learning, Neural Networks, and predictions based on various learning models.

## Table of Contents

- [Overview of Project](#overview-of-project)
  - [Team Member](#team-member)
  - [Topic Selection](#topic-selection)
  - [Purpose of Project](#purpose-of-project)
  - [Current Status](#current-status)
- [Segment 1: Sketch It Out](#segment-1-sketch-it-out)
  - [Resources](#resources)
  - [Next Step](#next-step)
  - [Roles and Contributions in Segment 1](#roles-and-contributions-in-segment-1)
- [Segment 2: Build and Assemble](#segment-2-build-and-assemble)
  - [Analysis Results](#analysis-results)
  - [Roles and Contributions in Segment 2](#roles-and-contributions-in-segment-2)
- [Segment 3: Put It All Together](#segment-3-put-it-all-together)
- [References](#references)

## Overview of Project

This project is divided into three Segments: Segment 1, Segment 2, and Segment 3. A checkbox with checkmark in it indicates that the corresponding segment and tasks are completed. 

- ✅ Segment 1: Sketch It Out.
- ✅ Segment 2: Build and Assemble.
- 🟩 Segment 3: Put It All Together.

### Team Member

Andia, Chris, Joey, Liwen, and Parto (alphabetical order).

### Topic Selection

We began exploring different datasets to address the question of when a company should execute layoffs. Unfortunately, for IPOs, the datasets we found contained less than 500 rows of relevant data. This seemed too small to create a robust machine learning model. We continued to explore different datasets regarding layoffs, but also began exploring the idea of creating a project regarding customer churn. One question to answer regarding customer churn is, what are the factors that lead to a customer either continuing or terminating their involvement (subscription, account, etc.) with the company. After viewing some datasets on telecom and bank customer churn, it seemed that these datasets had sufficient dimensions (such as tenure and credit score for the bank datasets) and many rows over 10000 to create a learning model. We will continue exploring this idea during the second class of our Final Project. Here is the link to the original dataset selected for our deep dive, [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download).

Questions that our team plans to answer is how to predict churn rate of customers based on bank customers' involvement, the reason(s) why customers left, and whether Machine Learning or Neural Network models can help our stakeholders solve.

### Purpose of Project

This project was the final team project to cultivate collaboration, teamwork, and effective use of collaboration tools, such as GitHub, Database Management System (DBMS), and online story board/dashboard. During this project, we were also encouraged to focus on Machine Learning or Neural Network models, and apply those techniques to solve a real world case study. Here are a few systematic steps that we have performed to find the best performing solution(s).

- Examine historical dataset.
- Preprocess the dataset, including proper data cleaning, standardization, and scaling whichever is necessary.
- Identify potential causes for bank customer churn.
- Develop Machine learning model to predict churn rate.

### Current Status

- ✅ Topic selection has been finalized: prediction of bank customer churn rate.

- ✅ Assessment of dataset and database management system (DBMS) is completed.
  - Our dataset consisted of 10000 rows and 14 columns. A few numeric columns contained some outliers as illustrated in Fig. 1(a)&ndash;(c).
  - A PostgreSQL database that stores two tables, called **main_df** and **clean_df**, was created and can be connected without problems from Python code of each Team Member. We documented some SQL queries for retrieving some data from the database ([BankCustomerChurn_ModelSelection.ipynb](./BankCustomerChurn_ModelSelection.ipynb)).

  <hr>
  <table><tr><td><img src='Data/CreditScore_boxplot.png' title='(a) Column CreditScore'></td><td><img src='Data/Age_boxplot.png' title='(b) Column Age'></td><td><img src='Data/NumOfProducts_boxplot.png' title='(c) Column NumOfProducts'></td></tr></table>

  **Fig. 1 Boxplots of several numerical columns containing some outliers: (a) Column CreditScore, (b) Column Age, and (c) Column NumOfProducts.**
  <hr>

  ![Fig. 2](./Data/BankCustomerChurn_fabricated_db.png)  
  **Fig. 2 Bank Customer Churn fabricated DBMS (PostgreSQL).**

- ✅ Preprocessing dataset and EDA is completed. The cleaned datasets, [Churn_Modelling_main.csv](./Resources/Churn_Modelling_main.csv) and [Churn_Modelling_cs_lt850.csv](./Resources/Churn_Modelling_cs_lt850.csv), are stored in the GitHub repo and PostgreSQL database.

- 🟩 Model Testing and Determination.
  - Evaluation Machine Learning or Neural Network models that could effectively predict bank customer churn rate.
  - Optimization of our final models is ongoing.

## Segment 1: Sketch It Out

Our team discussed our overall project objectives and resources (datasets, technologies, software, ML/NN models, etc.), selected a question/topic to focus on, and then built a simple model. We then prototyped our team's ideas by using either CSV or JSON files to connect the model to a fabricated database.

### Resources

- GitHub repository: [Bank-Customer-Churn](https://github.com/chris820629/Bank-Customer-Churn) for sharing our analysis details, datasets, and results.
- Source code: [BankCustomerChurn_ModelSelection.ipynb](./BankCustomerChurn_ModelSelection.ipynb).
- Source data: [Churn_Modelling_2.csv](./Resources/Churn_Modelling_2.csv) (source: [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download)).
- Database data: [Churn_Modelling_main.csv](./Resources/Churn_Modelling_main.csv), [Churn_Modelling_cs_lt850.csv](./Resources/Churn_Modelling_cs_lt850.csv), [Churn_Modelling_cs_lt2sigma.csv](./Resources/Churn_Modelling_cs_lt2sigma.csv).
- Fabricated DBMS: PostgreSQL ([Bank Customer Churn fabricated DBMS](./Data/BankCustomerChurn_fabricated_db.png)).
- Image file: png files.
- Software: [Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide), [Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html), [Python imbalanced-learn](https://pypi.org/project/imbalanced-learn/).
- Tableau dashboard: TBD.

### Next Step

We have been working on improving the accuracy and sensitivity of our learning models and preparing story/dashboard for presenting our data effectively.

- Tableau dashboard/story.
- Optimization of our learning models.
- Will transform the summary statistics table into a fabricated database that can also be used for Tableau dashboard/story.
- Final touches.

### Roles and Contributions in Segment 1

My major roles and contributions in **Segment 1** were as follows:

- A total of 6 commits to the team project GitHub repository.
- Completed analysis by using three ensemble learning models, called `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`, and `AdaBoostClassifier` for predicting churn rate of customers based on bank customers' involvement ([BankCustomerChurn_ModelSelection.ipynb](./BankCustomerChurn_ModelSelection.ipynb)). We preprocessed our dataset and saved the clean datasets in our team project GitHub and a PostgreSQL database. We mainly used StandardScaler, but scaling was skipped in some cases because Random Forests and Decision Trees are based on tree partitioning algorithms. To ensure there were no overlapping tasks in our team, the rest of the Team Members performed several types of Machine Learning and Neural Network models.
- Trained Andia, Joey, and the rest of Team Members on:
  - how to set up a GitHub repository and branch correctly and effectively.
  - how to checkout, commit, push a branch, and how to merge individual branches with the main branch correctly. We also made sure that our GitHub repo is protected and has proper review steps before Team Member can merge, modify, or delete shared codes/files.
  - best known methods for mastering GitHub repos, GitBash, Python code, and Jupyter Notebook code.
- Created a Python code that allows each Team Member to communicate with the PostgreSQL database via pgAdmin 4, SQLAlchemy, and Psycopg2.
- Helped other Team Members spot problems in their codes and correct them.
- Helped setting up GitHub main and local branches of each Team Member correctly. I also helped setting up the *gitignore* afterward because the team project GitHub repo was initially created without *gitignore* and without any protection modes.

## Segment 2: Build and Assemble

Our team performed some experiments to test and train our models, build the database that we will use for our final presentation, and create both our dashboard and presentation.

### Analysis Results

**Table 1. Condensed summary statistics of all ensemble learning models (Used datasets: *Original* with all outliers vs. *CS &lt; 850* without credit scores >= 850. Used metrics: low &lt; 60%, good = 60&ndash;70%, very good = 70&ndash;90%, high &ge; 90%).**  
| Ensemble algorithm     | Dataset-Exited | Balanced<br> accuracy score | Precision | Recall  | F1 score | Conclusion                         |
| :--                    | :--:           |           :--: |       --: |     --: |      --: | :--                                             |
| RandomForestClassifier | Original-0     |   0.736840     | 0.89      |    0.90 |  0.90    | Very good accuracy; **high recall/F1 score**    |
| (with SMOTEENN)        | Original-1     |   0.736840     | 0.61      |    0.57 |  0.59    | Very good accuracy; *low recall/F1 score*       |
| RandomForestClassifier | Original-0     |   0.712432     | 0.87      |    0.97 |  0.92    | Very good accuracy; **high recall/F1 score**    |
|                        | Original-1     |   0.712432     | 0.79      |    0.46 |  0.58    | Very good accuracy; *low recall/F1 score*       |
| BalancedRandomForest   | Original-0     |   0.730740     | 0.88      |    0.93 |  0.91    | Very good accuracy; **high recall/F1 score**    |
| (with SMOTEENN)        | Original-1     |   0.730740     | 0.66      |    0.53 |  0.59    | Very good accuracy; *low recall/F1 score*       |
| BalancedRandomForest   | Original-0     |   0.784557     | 0.93      |    0.81 |  0.86    | Very good accuracy/recall/F1 score              |
|                        | Original-1     |   0.784557     | 0.51      |    0.76 |  0.61    | Very good accuracy/recall; good F1 score        |
| EasyEnsembleClassifier | Original-0     |   0.734567     | 0.89      |    0.90 |  0.89    | Very good accuracy/F1 score; **high recall**    |
| (with SMOTEENN)        | Original-1     |   0.734567     | 0.60      |    0.57 |  0.58    | Very good accuracy; *low recall/F1 score*       |
| EasyEnsembleClassifier | Original-0     |   0.778234     | 0.93      |    0.80 |  0.86    | Very good accuracy/recall/F1 score              |
|                        | Original-1     |   0.778234     | 0.50      |    0.75 |  0.60    | Very good accuracy/recall; good F1 score        |
| AdaBoostClassifier     | Original-0     |   0.742793     | 0.90      |    0.87 |  0.88    | Very good accuracy/recall/F1 score              |
| (with SMOTEENN)        | Original-1     |   0.742793     | 0.55      |    0.62 |  0.58    | Very good accuracy; good recall; *low F1 score* |
| AdaBoostClassifier     | Original-0     |   0.729186     | 0.88      |    0.96 |  0.92    | Very good accuracy; **high recall/F1 score**    |
|                        | Original-1     |   0.729186     | 0.78      |    0.49 |  0.61    | Very good accuracy; *low recall*; good F1 score |
| RandomForestClassifier | CS < 850-0     |   0.741063     | 0.90      |    0.89 |  0.89    | Very good accuracy/recall/F1 score              |
| (with SMOTEENN)        | CS < 850-1     |   0.741063     | 0.56      |    0.59 |  0.58    | Very good accuracy; *low recall/F1 score*       |
| RandomForestClassifier | CS < 850-0     |   0.701590     | 0.88      |    0.96 |  0.92    | Very good accuracy; **high recall/F1 score**    |
|                        | CS < 850-1     |   0.701590     | 0.75      |    0.44 |  0.55    | Very good accuracy; *low recall/F1 score*       |
| BalancedRandomForest   | CS < 850-0     |   0.738949     | 0.90      |    0.91 |  0.90    | Very good accuracy; **high recall/F1 score**    |
| (with SMOTEENN)        | CS < 850-1     |   0.738949     | 0.60      |    0.57 |  0.59    | Very good accuracy; *low recall/F1 score*       |
| BalancedRandomForest   | CS < 850-0     |   0.772251     | 0.93      |    0.81 |  0.86    | Very good accuracy/recall/F1 score              |
|                        | CS < 850-1     |   0.772251     | 0.48      |    0.74 |  0.58    | Very good accuracy/recall; *low F1 score*       |
| EasyEnsembleClassifier | CS < 850-0     |   0.741146     | 0.90      |    0.88 |  0.89    | Very good accuracy/recall/F1 score              |
| (with SMOTEENN)        | CS < 850-1     |   0.741146     | 0.55      |    0.60 |  0.58    | Very good accuracy; good recall; *low F1 score* |
| EasyEnsembleClassifier | CS < 850-0     |   0.771172     | 0.93      |    0.79 |  0.85    | Very good accuracy/recall/F1 score              |
|                        | CS < 850-1     |   0.771172     | 0.47      |    0.75 |  0.58    | Very good accuracy/recall; *low F1 score*       |
| AdaBoostClassifier     | CS < 850-0     |   0.753748     | 0.91      |    0.86 |  0.88    | Very good accuracy/recall/F1 score              |
| (with SMOTEENN)        | CS < 850-1     |   0.753748     | 0.53      |    0.65 |  0.58    | Very good accuracy; good recall; *low F1 score* |
| AdaBoostClassifier     | CS < 850-0     |   0.717164     | 0.88      |    0.95 |  0.92    | Very good accuracy; **high recall/F1 score**    |
|                        | CS < 850-1     |   0.717164     | 0.71      |    0.48 |  0.58    | Very good accuracy; *low recall/F1 score*       |

<hr>
<table><tr><td><img src='Data/BankCustomerChurn_FeatureImportance.png' title='(a) Sorted feature importance'></td><td><img src='Data/BankCustomerChurn_clean_df_FeatureImportance.png' title='(b) Sorted feature importance w/o outliers'></td></tr><tr><td><img src='Data/BankCustomerChurn_SMOTEENN_FeatureImportance.png' title='(c) Sorted feature importance w/ SMOTEENN'></td><td><img src='Data/BankCustomerChurn_SMOTEENN_clean_df_FeatureImportance.png' title='(d) Sorted feature importance w/ SMOTEENN, w/o outliers'></td></tr></table>

**Fig. 3 Sorted feature importances of (a) the original dataset, (b) the dataset with CreditScore &lt; 850, (c) the original dataset with SMOTEENN, and (d) the dataset with SMOTEENN and CreditScore &lt; 850.**
<hr>

### Roles and Contributions in Segment 2

My major roles and contributions in **Segment 2** were as follows:

- A total of 13 commits to the team project GitHub repository.
- Completed final training and testing of four ensemble learning models, called `RandomForestClassifier`, `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`, and `AdaBoostClassifier` for predicting churn rate of customers based on bank customers' involvement. I also performed the code refactoring, so that our team could run and compare several ML models more efficiently, including two reusable functions for generating summary statistics tables and precision-recall curves ([BankCustomerChurn_ModelSelection.ipynb](./BankCustomerChurn_ModelSelection.ipynb)).
- Helped summarizing the summary statistics table, feature importances, and defined which ML model our team should pursue.
- Helped streamlining the structure of our team project GitHub repo and consolidating some paths and files. I also help updating and prettifying our README.md.
- Maintained the fabricated PostgreSQL database that our team planned to use (Fig. 2).

## Segment 3: Put It All Together

We put the final touches on our models, database, and dashboard. Then create and deliver our final presentation to the class.

## References

[Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide)  
[TensorFlow Documentation](https://www.tensorflow.org/guide/)  
[Scikit-learn User Guide - Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)  
[Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)  
[Matplotlib - Plot types](https://matplotlib.org/stable/plot_types/index.html)  
[Ensemble methods](https://imbalanced-learn.org/stable/references/ensemble.html#)  
[PostgreSQL documentation](https://www.postgresql.org/docs/)  
