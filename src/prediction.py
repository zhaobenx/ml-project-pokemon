# %%
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA

# %%
df = pd.read_csv("pokemon_alopez247.csv", sep=",")

# %%
print(df.head(6))
print(df.info())

# %% [markdown]
# # Prediction
#
# ## 1.  Data preprocessed
#
#
# Convert text labels to one hot code.
#
# 选择one hot的原因 [Reason](http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example)
#

# %%
df1 = df.copy()
# type_ = pd.get_dummies(df.Type_1, prefix='Type') + pd.get_dummies(df.Type_2,prefix='Type')
# color = pd.get_dummies(df.Color, prefix='Color')
# egg_group = pd.get_dummies(df.Egg_Group_1, prefix='Egg_group') + pd.get_dummies(df.Egg_Group_2,prefix='Egg_group')
# body_style = pd.get_dummies(df.Body_Style, prefix='Body_style')
# df1 = df1.drop(['Type_1', 'Type_2', 'Color','Egg_Group_1', 'Egg_Group_2', 'Body_Style'],axis=1)
# df1 = pd.concat([df1, type_, color, egg_group, body_style],axis=1,)
df1 = pd.get_dummies(data=df1, columns=['Type_1', 'Type_2'], prefix='Type')
df1 = pd.get_dummies(data=df1, columns=['Color'], prefix='Color')
df1 = pd.get_dummies(data=df1, columns=['Egg_Group_1', 'Egg_Group_2'], prefix='Egg_Group')
df1 = pd.get_dummies(data=df1, columns=['Body_Style'], prefix='Body_Style')
df1 = df1.fillna(0)

# %% [markdown]
# Generate X and y from the dataset.

# %%
# X = df1.loc[:, df1.columns != 'Name']
X = df1.drop(['Name', 'Number', 'Catch_Rate'], axis=1).to_numpy().astype(float)
labels = df1.drop(['Name', 'Number', 'Catch_Rate'], axis=1).columns
y = df1.Catch_Rate.to_numpy() / 255
n_sample = X.shape[0]
n_channel = X.shape[1]
print(f'Number of samples: {n_sample}\nNumber of features: {n_channel}')

# %% [markdown]
# ## 2. Linear prediction
#
# ### a. Basic linear regression
#
# First do the normalization. Then do the linear regression.

# %%
scaling = StandardScaler()
scaling.fit(X)
X_n = scaling.transform(X)

n_run = 5
train_scores = []
test_scores = []
coeffs = []
for i in range(n_run):
    Xtr, Xts, ytr, yts = train_test_split(X_n, y, test_size=0.33, shuffle=True)
    regr = LinearRegression()
    regr.fit(Xtr, ytr)
    yhat = regr.predict(Xts)
    train_scores.append(regr.score(Xtr, ytr))
    test_scores.append(regr.score(Xts, yts))
    coeffs.append(regr.coef_)

# print(test_scores)
print(f"Train R^2 score is {np.max(train_scores)}, test R^2 score is {np.max(test_scores)}.")
best = np.argmax(test_scores)
best_coef = coeffs[best]
# print(best_coef)
plt.figure(figsize=(20, 16))
plt.stem(best_coef)
plt.xlabel('features')
# plt.ylabel('$log(coef)$')
plt.title('Coefficients')
plt.xticks(range(n_channel), labels, rotation=90)
plt.show()

# %% [markdown]
# ### b. Linear regression with LASSO

# %%
nalpha = 10
alphas = np.logspace(-3, 2, nalpha)

train_scores = []
test_scores = []
coeffs = []

for alpha in alphas:
    for i in range(n_run):
        Xtr, Xts, ytr, yts = train_test_split(X_n, y, test_size=0.33, shuffle=True)
        #     print(f"calculating alpha for {alpha}")
        regr = Lasso(alpha=alpha)
        regr.fit(Xtr, ytr)
        #     print(f"Done {alpha}")
        yts_pred = regr.predict(Xts)
        train_scores.append(regr.score(Xtr, ytr))
        test_scores.append(regr.score(Xts, yts))
        coeffs.append(regr.coef_)

print(f"Best train R^2 score is {np.max(train_scores)}, test R^2 score is {np.max(test_scores)}.")
best = np.argmax(test_scores)
best_coef = coeffs[best]
# print(best_coef)
prominent_labels = labels[np.abs(best_coef).argsort()[-6:]]
print(f"Six most prominent features are {', '.join(prominent_labels)}")
plt.figure(figsize=(20, 16))
plt.stem(best_coef)
plt.xlabel('features')
# plt.ylabel('$log(coef)$')
plt.title('Coefficients')
plt.xticks(range(n_channel), labels, rotation=90)
plt.show()

# %% [markdown]
# ### c. Linear regression with Ridge

# %%
nalpha = 20
alphas = np.logspace(-2, 2, nalpha)

train_scores = []
test_scores = []
coeffs = []

for alpha in alphas:
    for i in range(n_run):
        Xtr, Xts, ytr, yts = train_test_split(X_n, y, test_size=0.33, shuffle=True)
        regr = Ridge(alpha=alpha)
        regr.fit(Xtr, ytr)
        train_scores.append(regr.score(Xtr, ytr))
        test_scores.append(regr.score(Xts, yts))
        coeffs.append(regr.coef_)

print(f"Best train R^2 score is {np.max(train_scores)}, test R^2 score is {np.max(test_scores)}.")
best = np.argmax(test_scores)
best_coef = coeffs[best]
# print(best_coef)
prominent_labels = labels[np.abs(best_coef).argsort()[-6:]]
print(f"Six most prominent features are {', '.join(prominent_labels)}")
plt.figure(figsize=(20, 16))
plt.stem(best_coef)
plt.xlabel('features')
# plt.ylabel('$log(coef)$')
plt.title('Coefficients')
plt.xticks(range(n_channel), labels, rotation=90)
plt.show()

# %% [markdown]
# ### d. Linear regression with PCA

# %%
nfold = 5

# Create a K-fold object
kf = KFold(n_splits=nfold)
kf.get_n_splits(X_n)

# Number of PCs to try
ncomp_test = np.arange(2, 100)
num_nc = len(ncomp_test)

acc = np.zeros((num_nc, nfold))

for icomp, ncomp in enumerate(ncomp_test):

    for ifold, I in enumerate(kf.split(X)):
        Itr, Its = I

        Xtr, Xts, ytr, yts = X_n[Itr], X_n[Its], y[Itr], y[Its]

        pca = PCA(n_components=ncomp, svd_solver='randomized', whiten=True)
        Xtr_transform = pca.fit_transform(Xtr)

        regr = LinearRegression()
        regr.fit(Xtr_transform, ytr)
        Xts_transform = pca.fit_transform(Xts)
        #         yhat = logreg.predict(Xts)

        acc[icomp, ifold] = regr.score(Xts_transform, yts)

# %%
n_run = 10

# Number of PCs to try
ncomp_test = np.arange(2, 100)
num_nc = len(ncomp_test)

train_scores = []
test_scores = []
coeffs = []

for ncomp in ncomp_test:
    for i in range(n_run):
        Xtr, Xts, ytr, yts = train_test_split(X_n, y, test_size=0.33, shuffle=True)
        pca = PCA(n_components=ncomp, svd_solver='randomized', whiten=True)
        Xtr_transform = pca.fit_transform(Xtr)

        regr = LinearRegression()
        regr.fit(Xtr_transform, ytr)
        Xts_transform = pca.fit_transform(Xts)

        train_scores.append(regr.score(Xtr_transform, ytr))
        test_scores.append(regr.score(Xts_transform, yts))
        coeffs.append(regr.coef_)

# %%
# best = np.argmax(acc)
print(f"Best train R^2 score is {np.max(train_scores)}, test R^2 score is {np.max(test_scores)}.")
# print(train_scores)
# print(test_scores)
train_scores = np.array(train_scores).reshape(num_nc, -1)
test_scores = np.array(test_scores).reshape(num_nc, -1)

# keep biggest 6 and compute their mean
train_mean = np.sort(train_scores)[:, :4:-1].mean(axis=1)
test_mean = np.sort(test_scores)[:, :4:-1].mean(axis=1)
plt.plot(train_mean)
plt.plot(test_mean)
plt.legend(['Train', 'Test'])
plt.show()

# %%
