# %%
"""
### Oblig 6
"""

# %%
"""
##### 1 Feature Selection
Using the UCI mushroom dataset from the lasat exercise, perform a feature selection using a classifier evaluator. Which features are most discriminative
"""

# %%
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA


# %%


mushrooms = pd.read_csv('agaricus-lepiota.csv')

y = mushrooms.pop('edibility')

X = pd.get_dummies(mushrooms)


X

# %%
y

# %%
# k -> how many selected features
skb = SelectKBest(chi2, k=5)
skb.fit(X,y)
X_new = skb.transform(X)

print(X_new.shape)

# %%
selected_features = np.array(X.columns)[skb.get_support(indices=True)]
selected_features

# %%
"""
#### 2 PCA
Use principal components analysis to construct a reducesd space. Which combination of features explain the most variance in the dataset?
"""

# %%
pca = PCA(n_components=5)

X_pca = pca.fit_transform(X)

# Principal component analysis space: 
# Reducing the dimentionality for the dataset and increasing
# interpretability, but at the same time minimizing information loss
print("Original shape: ", X.shape) # original shape (space for dataset)
print("pca space: ", X_pca.shape)

obtain_feature = lambda i: pca.components_[i].argmax()
features = [X.columns[obtain_feature(i)] for i in range(X_pca.shape[1])]

print("The combination of features which gives the most variance: ", features)



# %%
"""
##### 3
Overlap between PCA features and those obtained from feature selection?
"""

# %%
set(features).intersection(selected_features)