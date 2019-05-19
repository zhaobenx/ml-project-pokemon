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

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as utils


# %%
df = pd.read_csv("pokemon_alopez247.csv", sep=",")


# %%
print(df.head(6))
print(df.info())

# %% [markdown]
#  # Prediction
#
#  ## 1.  Data preprocessed
#
#
#  Convert text labels to one hot code.
#
#  选择one hot的原因 [Reason](http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example)
#

# %%
df1 = df.copy()

df1 = pd.get_dummies(data=df1, columns=['Type_1', 'Type_2'], prefix='Type')
df1 = pd.get_dummies(data=df1, columns=['Color'], prefix='Color')
df1 = pd.get_dummies(data=df1, columns=['Egg_Group_1', 'Egg_Group_2'], prefix='Egg_Group')
df1 = pd.get_dummies(data=df1, columns=['Body_Style'], prefix='Body_Style')
df1 = df1.fillna(0)

# %% [markdown]
#  Generate X and y from the dataset.

# %%
# X = df1.loc[:, df1.columns != 'Name']
X = df1.drop(['Name', 'Number', 'Catch_Rate'], axis=1).to_numpy().astype(float)
labels = df1.drop(['Name', 'Number', 'Catch_Rate'], axis=1).columns
y = df1.Catch_Rate.to_numpy() / 255
n_sample = X.shape[0]
n_channel = X.shape[1]
print(f'Number of samples: {n_sample}\nNumber of features: {n_channel}')

# %% [markdown]
#  ## 2. Linear prediction
#
#
# %% [markdown]
#  ### a. Basic linear regression
#
#  First do the normalization. Then do the linear regression.

# %%
scaling = StandardScaler()
scaling.fit(X)
X_n = scaling.transform(X)


def score(model, x, y):
    return 1 - np.mean((model.predict(x) - y)**2)


n_run = 5
train_scores = []
test_scores = []
coeffs = []
for i in range(n_run):
    Xtr, Xts, ytr, yts = train_test_split(X_n, y, test_size=0.33, shuffle=True)
    regr = LinearRegression()
    regr.fit(Xtr, ytr)
    yhat = regr.predict(Xts)
    train_scores.append(score(regr, Xtr, ytr))
    test_scores.append(score(regr, Xts, yts))
    coeffs.append(regr)

# print(test_scores)
print(f"Train score is {np.max(train_scores)}, test score is {np.max(test_scores)}.")
best = np.argmax(test_scores)
best_coef = coeffs[best].coef_
# print(best_coef)
plt.figure(figsize=(20, 16))
plt.stem(best_coef)
plt.xlabel('features')
# plt.ylabel('$log(coef)$')
plt.title('Coefficients')
plt.xticks(range(n_channel), labels, rotation=90)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(yts, coeffs[best].predict(Xts), 'o')
plt.plot(yts, yts)
plt.xlabel('yts')
plt.ylabel('yhat')
plt.title('Comparasion on test and predicted data')
plt.legend(['prediction', 'true relation'])
plt.show()

# %% [markdown]
#  ### b. Linear regression with LASSO

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
        train_scores.append(score(regr, Xtr, ytr))
        test_scores.append(score(regr, Xts, yts))
        coeffs.append(regr)

print(f"Best train score is {np.max(train_scores)}, test  score is {np.max(test_scores)}.")
best = np.argmax(test_scores)
best_coef = coeffs[best].coef_
# print(best_coef)
prominent_labels = labels[np.abs(best_coef).argsort()[-6:]][::-1]
print(f"Six most prominent features are {', '.join(prominent_labels)}")
plt.figure(figsize=(20, 16))
plt.stem(best_coef)
plt.xlabel('features')
# plt.ylabel('$log(coef)$')
plt.title('Coefficients')
plt.xticks(range(n_channel), labels, rotation=90)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(yts, coeffs[best].predict(Xts), 'o')
plt.plot(yts, yts)
plt.xlabel('yts')
plt.ylabel('yhat')
plt.title('Comparasion on test and predicted data')
plt.legend(['prediction', 'true relation'])
plt.show()

# %% [markdown]
#  ### c. Linear regression with Ridge

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
        train_scores.append(score(regr, Xtr, ytr))
        test_scores.append(score(regr, Xts, yts))
        coeffs.append(regr)

print(f"Best train score is {np.max(train_scores)}, test score is {np.max(test_scores)}.")
best = np.argmax(test_scores)
best_coef = coeffs[best].coef_
# print(best_coef)
prominent_labels = labels[np.abs(best_coef).argsort()[-6:]][::-1]
print(f"Six most prominent features are {', '.join(prominent_labels)}")
plt.figure(figsize=(20, 16))
plt.stem(best_coef)
plt.xlabel('features')
# plt.ylabel('$log(coef)$')
plt.title('Coefficients')
plt.xticks(range(n_channel), labels, rotation=90)
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(yts, coeffs[best].predict(Xts), 'o')
plt.plot(yts, yts)
plt.xlabel('yts')
plt.ylabel('yhat')
plt.title('Comparasion on test and predicted data')
plt.legend(['prediction', 'true relation'])
plt.show()

# %% [markdown]
#  ### d. Linear regression with PCA

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

        train_scores.append(score(regr, Xtr_transform, ytr))
        test_scores.append(score(regr, Xts_transform, yts))
        coeffs.append(regr.coef_)


# %%
# best = np.argmax(acc)
print(f"Best train  score is {np.max(train_scores)}, test score is {np.max(test_scores)}.")
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

# %% [markdown]
#  ### e. With neural network

# %%


class Net(nn.Module):
    def __init__(self, n):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


lr = 0.001

epochs = 100
batch_size = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = utils.TensorDataset(torch.Tensor(X_n), torch.Tensor(y))
dataloader = utils.DataLoader(dataset, batch_size=50, shuffle=True)

print(f"Running on device {device}")

# Use GPU
model = Net(X_n.shape[1]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss()
train_scores = []
losses = []

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    score = 0
    model = model.train()
    for index, data in enumerate(dataloader):
        x_, y_ = data
        x_, y_ = x_.to(device), y_.to(device)
        pred = model(x_).view(-1)
#         print(y_.shape, pred.shape)
        loss = criterion(pred, y_)
#         loss = torch.mean((pred - y_)**2)
        epoch_loss += loss.item()

        # print(f"{index*batch_size:.4f} --- loss: {loss.item()/batch_size:.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        X_val = torch.FloatTensor(X_n).to(device)
        result = model(X_val)
        pred = result.cpu().numpy()
        # score = r2_score(y, pred)
        score = 1 - np.mean((y - pred[0, :])**2)
    train_scores.append(score)
    losses.append(epoch_loss / len(dataset))
#     print(f'Epoch {epoch} finished! Loss: {epoch_loss / len(dataset):.4f} Score: {score:.6f}')
    torch.save(model.state_dict(), 'saved.model')


with torch.no_grad():
    X_val = torch.FloatTensor(X_n).to(device)
    result = model(X_val)
    pred = result.cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.plot(y, pred, 'o')
    plt.plot(y, y)
    plt.xlabel('y')
    plt.ylabel('yhat')
    plt.title('Comparasion on real and predicted data')
    plt.legend(['prediction', 'true relation'])
    plt.show()

plt.figure(figsize=(10, 8))
plt.plot(train_scores, 'x-')
# plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('score')
plt.title('Score')
# plt.legend(['Score', 'Loss'])

plt.show()
