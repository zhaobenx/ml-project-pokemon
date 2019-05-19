
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns


# %%
df = pd.read_csv("pokemon_alopez247.csv", sep=",")


# %%
print("There are", len(df.columns), "columns:")
for x in df.columns:
    sys.stdout.write(str(x)+", ")


# %%
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f', cmap="magma")
plt.show()


# %%
Catch_Rate = df["Catch_Rate"]
Catch_Rate_ = Catch_Rate/255
Catch_Rate_.plot(kind='hist', bins=200, figsize=(6, 6))
plt.title("Catch_Rate")
plt.xlabel("Catch_Rate")
plt.ylabel("Frequency")
plt.show()


# %%
number = df["Number"]
print('Total number of Pokemons is', len(number))
Legendary = df["isLegendary"]
rate = np.mean(Legendary == True)
print('legendary rate=', rate)


# %%
# Unnecessary columns
# Number and Name are just identifiers
# Total is a aggregation of others columns
clean_df = df.drop(columns=['Number', 'Name', 'Total'])


# %%
df2 = df.loc[:, ["Attack", "Sp_Atk"]]
df2.plot()


# %%
sns.boxplot(data=df.drop(['isLegendary', 'Generation', 'Number', 'Total', 'Color', 'hasGender', 'Pr_Male',
                          'Egg_Group_1', 'Egg_Group_2', 'hasMegaEvolution', 'Height_m', 'Weight_kg', 'Body_Style'], axis=1).head())
plt.show


# %%
fig, axes = plt.subplots(nrows=2, ncols=1)
df.plot(kind="hist", y="Catch_Rate", bins=50, range=(0, 255), normed=True, ax=axes[0])
df.plot(kind="hist", y="Catch_Rate", bins=50, range=(0, 255), normed=True, ax=axes[1], cumulative=True)
plt.show()
# there's a sudden increase around 0.16 percentage of Catch_Rate


# %%
plt.figure(figsize=(10, 5))
ax = sns.boxplot(x='Type_1', y=Catch_Rate_, data=df)
plt.figure(figsize=(10, 5))
ax = sns.boxplot(x='Type_2', y=Catch_Rate_, data=df)


# %%
# If generation and islegendary relate to Catch_rate
plt.figure(figsize=(10, 5))
ax = sns.boxplot(x='Generation', y=Catch_Rate_, hue='isLegendary', data=df)


# %%
result1 = df.drop(['Type_1', 'Type_2', 'Sp_Atk', 'Sp_Def', 'isLegendary', 'Generation', 'Number', 'Total', 'Color', 'hasGender',
                   'Pr_Male', 'Egg_Group_1', 'Egg_Group_2', 'hasMegaEvolution', 'Height_m', 'Weight_kg', 'Body_Style'], axis=1).head(20)
print(result1)


# %%
def result_pic(result):
    labels = ['HP', 'Attack', 'Defense', 'Speed', 'Catch_Rate']
    kinds = list(result.iloc[:, 0])

    result = pd.concat([result, result[['HP']]], axis=1)
    centers = np.array(result.iloc[:, 1:])

    n = len(labels)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle = np.concatenate((angle, [angle[0]]))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    for i in range(len(kinds)):
        ax.plot(angle, centers[i], linewidth=2, label=kinds[i])
        ax.fill(angle, centers[i])

    ax.set_thetagrids(angle * 180 / np.pi, labels)
    plt.title('different Pokemon')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    result = result1
    result_pic(result)


# %%
sns.pairplot(data=df.drop(['isLegendary', 'Generation', 'Number', 'Total', 'Color', 'hasGender', 'Pr_Male', 'Egg_Group_1',
                           'Egg_Group_2', 'hasMegaEvolution', 'Height_m', 'Weight_kg', 'Body_Style'], axis=1), hue='Catch_Rate')
plt.show


# %%
df.describe()


# %%
df1 = df.groupby('Type_1')['Type_1'].count().reset_index(name='Count')
df1 = df1.sort_values(by='Count')


# %%
plt.figure(figsize=(15, 10))
sns.barplot(x=df1['Type_1'], y=df1['Count'])
plt.xticks(rotation=90)
plt.xlabel('Type1')
plt.ylabel('Count')
plt.title('Type Distribution')


# %%
df2 = df.groupby('Type_2')['Type_2'].count().reset_index(name='Count')
df2 = df2.sort_values(by='Count')


# %%
plt.figure(figsize=(15, 10))
sns.barplot(x=df2['Type_2'], y=df2['Count'])
plt.xticks(rotation=90)
plt.xlabel('Type2')
plt.ylabel('Count')
plt.title('Type Distribution')


# %%
df3 = df.groupby('Egg_Group_1')['Egg_Group_1'].count().reset_index(name='Count')
df3 = df3.sort_values(by='Count')
plt.figure(figsize=(15, 10))
sns.barplot(x=df3['Egg_Group_1'], y=df3['Count'])
plt.xticks(rotation=90)
plt.xlabel('Egg_Group_1')
plt.ylabel('Count')
plt.title('Group Distribution')


# %%
df4 = df.groupby('Egg_Group_2')['Egg_Group_2'].count().reset_index(name='Count')
df4 = df4.sort_values(by='Count')
plt.figure(figsize=(15, 10))
sns.barplot(x=df4['Egg_Group_2'], y=df4['Count'])
plt.xticks(rotation=90)
plt.xlabel('Egg_Group_2')
plt.ylabel('Count')
plt.title('Group Distribution')
