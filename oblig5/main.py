# %%
"""
# Oblig 5 Maskinlæring
"""

# %%
"""

1. Download the mushroom dataset here: https://archive.ics.uci.edu/ml/datasets/Mushroom
2. Create a new Jupyter notebook
3. Load the dataset from CSV into pandas
4. Explore the distributions in the data. For example, how is habitat distributed between edibility vs non-edibility?
5. The data is entirely categorical. Convert each feature to dummy variables.
6. Visualise the feature space using a similar method to the one we used for the 20 newgroups dataset.
"""

# %%
"""
##### 1. Download the mushroot dataset
##### 2. Create new jupyter notebook
##### 3. Load dataset from CSV into pandas
"""

# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt

# %%
mushrooms = pd.read_csv('agaricus-lepiota.data')

mushrooms.describe()

# %%
"""
 ##### 4. Explore the distrubution in the data. For example, how is habitat distributed between edibility vs non-edibility?
"""

# %%
# mushrooms.columns
mushrooms.info()


# %%
# Sjekker om det er duplikat data
dup = mushrooms.duplicated().sum()
count = mushrooms.shape[0]
print(f'Det er {dup} duplikater i {count} rader')

# %%
# Spiselige sopper
mush_e = mushrooms[mushrooms['edibility']=='e']
mush_e[['habitat', 'edibility']].groupby('habitat').describe().transpose()


# %%
# Giftige sopper
mush_p = mushrooms[mushrooms['edibility']=='p']
mush_p[['habitat', 'edibility']].groupby('habitat').describe().transpose()

# %%
# Visuell representasjon om hvordan spiselighet og ikke-spiselighet er fordelt på habitat.
plt.figure()
fif, axis = plt.subplots(1,2, sharey=True)

axis[0].set_title("Edibility")
axis[1].set_title("non-edibility")

axis[0].hist(mush_e['habitat'], bins=20)
axis[1].hist(mush_p['habitat'], bins=20, color='green')

# %%
# list all features of mushrooms
mushrooms.info()

# %%
feature = 'population'

fig, axis = plt.subplots(1, len(mushrooms[feature].unique()),sharey=True, figsize=(15,5))
fig.suptitle(f'{feature}/edibility distrubution')

for i, pop in enumerate(mushrooms[feature].unique().tolist()):
    axis[i].set_title(pop)
    axis[i].hist(mushrooms[mushrooms[feature] == pop]['edibility'], bins=5)

# %%
"""
##### 5. The data is entirely categorical. Convert each feature to dummy variable.
"""

# %%
dum = pd.get_dummies(mushrooms).transpose()
dum.head()

# %%
"""
##### 6. Visualise the space using a similar method to the one we used for the 20 nrewgroups dataset.
"""

# %%

fig = plt.gcf()
fig.set_size_inches(60,200)
#plt.figure(figsize=(200,200))
plt.spy(dum.transpose(), markersize=0.5)
plt.plot()
plt.show()
