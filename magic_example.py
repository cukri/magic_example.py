import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)
print(df.head())

for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label], color='blue', label = 'gamma',alpha = 0.7, density = True )
    plt.hist(df[df["class"] == 0][label], color='red', label='gamma', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    data = np.hstack((x,np.reshape(y, (-1, 1))))

    return data, x, y

print(len(train[train["class"]==1])) #gamma
print(len(train[train["class"]==0])) #gamma

train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)