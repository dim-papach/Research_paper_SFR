#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Astropy for ecsv data
from astropy.io import ascii
from astropy.table import Table, QTable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reading the ECSV file
data = ascii.read("tables/final_table.ecsv")

data = data.to_pandas()
print(data.shape)
# keep only columns with numeric data types
numeric_data = data.select_dtypes(include=[np.number])

# drop columns with e_ in the name
numeric_data = numeric_data.drop(
    columns=[col for col in numeric_data.columns if "e_" in col]
)
print(numeric_data.describe())

# normalize the data with standard scaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
normalized_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)
print(normalized_data.describe())

# fill the scaled_data arrays with 0 where there were nans
normalized_data = normalized_data.fillna(0)


# PCA analysis
pca = PCA()
pca_data = pca.fit_transform(normalized_data)

# print the explained variance ratio
expl_var = pca.explained_variance_ratio_
df_expl_var = (
    pd.DataFrame(
        data=zip(range(1, len(expl_var) + 1), expl_var, expl_var.cumsum()),
        columns=["PCA", "Explained Variance (%)", "Total Explained Variance (%)"],
    )
    .set_index("PCA")
    .mul(100)
    .round(1)
)
print(df_expl_var)
# Plotting our explained variance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.bar(
    x=df_expl_var.index,
    height=df_expl_var["Explained Variance (%)"],
    label="Explained Variance",
    width=0.9,
    color="#AAD8D3",
)
ax.plot(
    df_expl_var["Total Explained Variance (%)"],
    label="Total Explained Variance",
    marker="o",
    c="#37B6BD",
)

plt.ylim(0, 100)
plt.ylabel("Explained Variance (%)")
plt.xlabel("PCA")
plt.grid(True, axis="y")
plt.title("Understanding Explained Variance in PCA\ndatagy.io")
plt.legend()
plt.close()
# Fitting and Transforming Our Data Using PCA
pca = PCA(3)
X_r = pca.fit_transform(normalized_data)
# print the explained variance ratio
expl_var = pca.explained_variance_ratio_
df_expl_var = (
    pd.DataFrame(
        data=zip(range(1, len(expl_var) + 1), expl_var, expl_var.cumsum()),
        columns=["PCA", "Explained Variance (%)", "Total Explained Variance (%)"],
    )
    .set_index("PCA")
    .mul(100)
    .round(1)
)
print(df_expl_var)
print(X_r.shape)
# Plotting a Heatmap of Our Loadings
fig, ax = plt.subplots(figsize=(8, 8))

ax = sns.heatmap(
    pca.components_,
    cmap="coolwarm",
    yticklabels=[f"PCA{x}" for x in range(1, pca.n_components_ + 1)],
    xticklabels=list(numeric_data.columns),
    linewidths=1,
    annot=True,
    fmt=",.2f",
    cbar_kws={"shrink": 0.8, "orientation": "horizontal"},
)

ax.set_aspect("equal")
plt.title("Loading for Each Variable and Component", weight="bold")
plt.close()


# Creating a DataFrame with the PCA Results
df_pca = pd.DataFrame(X_r, columns=["PC1", "PC2", "PC3"])
print(df_pca.head())
print(df_pca.shape)


score = X_r
coef = np.transpose(pca.components_)
labels = list(numeric_data.columns)

xs = score[:, 0]
ys = score[:, 1]
zs = score[:, 2]
n = coef.shape[0]
scalex = 1.0 / (xs.max() - xs.min())
scaley = 1.0 / (ys.max() - ys.min())
scalez = 1.0 / (zs.max() - zs.min())

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    xs * scalex,
    ys * scaley,
    zs * scalez,
    s = 10
)

for i in range(n):
    ax.quiver(0, 0, 0,coef[i, 0], coef[i, 1], coef[i, 2],color="r", alpha=0.5)
    ax.text(
        coef[i, 0] * 1.15,
        coef[i, 1] * 1.15,
        coef[i, 2] * 1.15,
        labels[i],
        color="darkblue",
        ha="center",
        va="center",
    )

plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))

plt.title("Biplot of PCA")

plt.show()
"""

import plotly.graph_objs as go

# Create the scatter plot
scatter = go.Scatter3d(
    x=xs * scalex,
    y=ys * scaley,
    z=zs * scalez,
    mode="markers",
    marker=dict(size=1),
    name="Data Points",
)

# Create the vector arrows
arrows = []
for i in range(n):
    arrows.append(
        go.Scatter3d(
            x=[0, coef[i, 0] * 1.15],
            y=[0, coef[i, 1] * 1.15],
            z=[0, coef[i, 2] * 1.15],
            mode="lines",
            name=labels[i],
        )
    )

# Create the layout
layout = go.Layout(
    title="Biplot of PCA",
    scene=dict(
        xaxis=dict(title="PC1"), yaxis=dict(title="PC2"), zaxis=dict(title="PC3")
    ),
)

# Create the figure
fig = go.Figure(data=arrows + [scatter], layout=layout)

# Show the plot
fig.show()
