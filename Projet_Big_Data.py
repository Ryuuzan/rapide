#!/usr/bin/env python
# coding: utf-8

# Projet Big Data - Detection de Malware

# In[64]:


import pandas as pd
import numpy as np
from IPython.display import display

malwareData = pd.read_csv('https://raw.githubusercontent.com/securitylab-repository/malware_classification/master/datasets/malware-detection/malwaredata.csv', delimiter=',')
pd.set_option('display.max_columns', None)


# In[65]:


malwareData.head()


# In[66]:


malwareData.info()


# In[67]:


for x in range(0, 57):
    display(malwareData.iloc[:,x].sample(10))


# In[68]:


malwareData.describe()


# In[69]:


import matplotlib.pyplot as plt
malwareData.hist(bins=50, figsize=(20,15))
plt.show()


# In[70]:


malwareData["SectionsMaxEntropy"].hist()


# In[71]:


malwareData["SectionsMaxEntropy"].sample(10)


# In[72]:


malwareData["SectionsMaxEntropy"].hist()


# In[73]:


malwareData["ResourcesMaxEntropy"].hist()


# Après avoir analysé les différentes composantes constituant ce dataset, nous allons chercher les corrélations éventuels entre les fichiers exécutables sains et les malwares en se basant sur la colonne "legitimate".

# In[74]:


corr_matrix = malwareData.corr()
corr_matrix["legitimate"].sort_values(ascending=False)


# Après avoir observé les différentes valeurs de corrélations affichées par la matrice, nous allons supprimer toutes les colonnes dont la corrélation est trop faible : inférieure à 0.2.

# In[75]:


from pandas.plotting import scatter_matrix
attributes = ["legitimate","SectionsMaxEntropy"]
scatter_matrix(malwareData[attributes], figsize=(12, 8))


# On créé donc notre ensemble d'entrainement sur une data nétoyée

# In[76]:


cleanData = malwareData[["legitimate","Subsystem","MajorSubsystemVersion","SizeOfOptionalHeader","ResourcesMinEntropy","Characteristics","ResourcesMaxEntropy","SectionsMeanEntropy","DllCharacteristics","SectionsMaxEntropy"]]


# In[77]:


for col in cleanData.columns:
    if (col != "legitimate"):
        cleanData[col] = (cleanData[col] - cleanData[col].min()) / (cleanData[col].max() - cleanData[col].min())

print(cleanData.head())


# In[78]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(cleanData, test_size=0.2, random_state=42)
display(type(train_set))
display(type(test_set))
display(type(train_set.info()))
display(type(test_set.info()))


# In[79]:


data = train_set.drop("legitimate", axis=1)
labels = train_set["legitimate"].copy()

testdata = test_set.drop("legitimate", axis=1)
testlabels = test_set["legitimate"].copy()


# In[80]:


data.sample(10)


# KNN Algorithm

# On cherche a déterminer le meilleur n_neighbors

# In[81]:


def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())


# In[82]:


x = malwareData[["Subsystem","MajorSubsystemVersion","SizeOfOptionalHeader","ResourcesMinEntropy","Characteristics","ResourcesMaxEntropy","SectionsMeanEntropy","DllCharacteristics","SectionsMaxEntropy"]]
y = malwareData[["legitimate"]]


# In[83]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_scores = cross_val_score(lin_reg, x, y,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[84]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='uniform')

# Entraînement
knn.fit(data,labels)

# Test
y_eval = knn.predict(testdata)

# Calcul et affichage du score
print("Test set score: {:.2f}".format(accuracy_score(testlabels, y_eval)))
print(confusion_matrix(testlabels, y_eval))
print("Test set score: {:.2f}".format(np.mean(y_eval == testlabels)))


# In[ ]:




