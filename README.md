import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import math
import statistics as stt
import scipy.stats as st
df=pd.read_excel("C:\\Users\Rakesh\Desktop\FA-EDA\IPL-Dataset-EDA-Part-B.xlsx")
df
df["SOLD PRICE"]
sns.histplot(df["SOLD PRICE"],kde=True)
plt.xlabel("sold price")
plt.ylabel("Density")
plt.title("sold price distribution plot")
plt.show()
df["COUNTRY"].value_counts()

df[df["COUNTRY"]=="IND"]
India=df[df["COUNTRY"]=="IND"]
plt.figure(figsize=(10,6))
df["COUNTRY"].value_counts().plot(kind="bar")

df["SIXERS"].max()
players_max_sixer = df[df['SIXERS'] ==129]
player_name_max_sixer_data = players_max_sixer[['PLAYER NAME', 'SIXERS']]
player_name_max_sixer_data
df[['TEAM',"SIXERS"]].sort_values(by="SIXERS",ascending=False)
df["RUNS-S"].quantile(0.25)
plt.figure(figsize=(12, 8))
sns.boxplot(data=df["HS"])
plt.xticks(rotation=45)
plt.title("Box Plot of HS")
plt.show()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["TEAM"].value_counts().to_frame()
one_hot = pd.get_dummies(df["TEAM"], prefix="TEAM")
df['TEAM']=le.fit_transform(df['TEAM'])
data = pd.concat([df, one_hot], axis=1)
data.head(3
df["ODI-RUNS-S"]
np.percentile(df["ODI-RUNS-S"],95)
df.isnull().sum()
duplicate_count=df.duplicated().sum()
duplicate_count
plt.figure(figsize=(20, 10))
sns.boxplot(data=df)
