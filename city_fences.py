"""
File: city_fences.py
Author: Chuncheng Zhang
Date: 2023-04-28
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Load the dataset of course.
    Find their hull fences.

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-04-28 ------------------------
# Pending

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm

from download.data_reader import Dataset, dataset
from download.adcode_reader import AdcodeDataset, adcode_dataset


# %% ---- 2023-04-28 ------------------------
# Pending
euluc_table = dataset.table
land_types = dataset.land_types
euluc_table


# %% ---- 2023-04-28 ------------------------
# Group the cities and their landmarks,
# and compute the ratio of them in every city.

group = euluc_table.groupby(['City_CODE', 'Level1'])
table = group['F_AREA'].sum()
sum_by_city = table.groupby(level=0).sum()

features = []
for c in tqdm(sum_by_city.index, 'Reading cities'):
    adc, fen = adcode_dataset.get_city(c)

    if fen is not None:
        features.append(fen)

    v = sum_by_city[c]
    table[c] /= v

feature_collection = dict(
    type='FeatureCollection',
    features=features,
)

table = pd.DataFrame(table)
table

# %% ---- 2023-04-28 ------------------------
# Pending
gdf = gpd.GeoDataFrame.from_features(feature_collection)
gdf.crs = ccrs.PlateCarree()
gdf

# %% ---- 2023-04-28 ------------------------
# Pending
fig, axes = plt.subplots(5, 1, figsize=(6, 4 * 5))

for level1, ax in zip([1, 2, 3, 4, 5], axes):
    df = table.query(f'Level1 == {level1}').copy()
    df['adcode'] = [e[0] for e in df.index]

    merged = pd.merge(gdf, df, left_on='adcode', right_on='adcode')

    kwargs = dict(
        ax=ax,
        vmin=0,
        vmax=1,
        legend=True,
    )

    merged.plot('F_AREA', **kwargs)
    ax.set_title(land_types[level1])

fig.tight_layout()
plt.show()

# %%

# %%
