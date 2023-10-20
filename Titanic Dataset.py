#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Ask a question: Classify whether or not an individual survives the Titanic ship catastrophe based on indivdual and ship 
# characteristics. Will this indivdual survive the Titanic catastrophe? Essentially we want to take passanger information and 
# predict if this passanger will survive the Titanic. 

# This is a classification question as we are predicitng a label for survival, either they survived or they died. 
# In our DataFrames X is Features or sample data and y are our labels or targets


import matplotlib.pyplot as plt
import pandas as pd
from sklearn import(
    ensemble, 
    preprocessing, 
    tree, 
)
from sklearn.metrics import (
    auc, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
)
from yellowbrick.classifier import(
        ConfusionMatrix,
        ROCAUC,
)
from yellowbrick.model_selection import (
    LearningCurve, 
) 


df = pd.read_csv(r"C:\Users\alexn\Documents\MachineLearning\titanic\train.csv")
# df.head()
# print(df.shape)

# df.describe().iloc[:,:2]

df.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:




