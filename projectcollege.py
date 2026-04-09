import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("graduate_admission1.csv")
df.head(5)
df.info()
df.describe()

df.isnull().sum()
df.dropna(inplace=True)

#correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()

#cgpa vs chance of admit
sns.scatterplot(x="GPA", y="Chance of Admit",data=df)
plt.show()

#GRE Distribution
sns.histplot(df["GRE Score"],bins=20)
plt.show()


