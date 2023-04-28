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

import traceback
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.cluster import KMeans

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

def group2stacked(group, read_city_flag=False):
    '''
    Convert the group into stacked multi-indexed DataFrame
    '''
    table = group['F_AREA'].sum()
    sum_by_city = table.groupby(level=0).sum()

    features = []

    # Add china hull at first
    if read_city_flag:
        _, fen = adcode_dataset.get_city(100000)
        if fen is not None:
            features.append(fen)

    for c in tqdm(sum_by_city.index, 'Compute every cities'):
        # Read the city geometry from the adcode_dataset,
        # if the read_city_flag is set.
        if read_city_flag:
            adc, fen = adcode_dataset.get_city(c)
        else:
            fen = None

        if fen is not None:
            features.append(fen)

        v = sum_by_city[c]
        table[c] /= v

    feature_collection = dict(
        type='FeatureCollection',
        features=features,
    )

    table = pd.DataFrame(table)

    return table, feature_collection


# %%
group = euluc_table.groupby(['City_CODE', 'Level1'])
table, feature_collection = group2stacked(group, read_city_flag=True)

group = euluc_table.groupby(['City_CODE', 'Level2'])
table_level2, _ = group2stacked(group)

table_level2

# %%
euluc_table

# %%
# Prepare geoDataFrame for the feature_collection
gdf = gpd.GeoDataFrame.from_features(feature_collection)
gdf.crs = ccrs.PlateCarree()
gdf

# %%
plt.style.use('ggplot')


def plot_map(df,
             value_name,
             ax,
             title,
             plot_kwargs=dict(
                 cmap='cividis',
                 #  vmin=0,
                 #  vmax=1,
                 legend=True,
             )):
    '''
    Plot the map using df at ax

    Args:
        param: df: DataFrame: The dataFrame of values of the adcode areas.
        param: value_name: str: The name of the value to be plotted.
        param: plot_kwargs: dict: Additional keyword arguments for map plotting.
        param: ax: Axis: The plt axis to be plotted.
        param: title: str: The title of the ax.
    '''

    merged = pd.merge(gdf, df, how='outer',
                      left_on='adcode', right_on='adcode')

    try:
        # Draw the outline of China
        kwargs = dict(
            ax=ax,
            legend=False,
            edgecolor='gray',
            facecolor='white',
            alpha=0.5
        )
        merged.iloc[:1].plot(**kwargs)

        # Draw map
        kwargs = dict(
            plot_kwargs,
            ax=ax,
            legend=True,
        )
        merged.iloc[1:].plot(value_name, **kwargs)
    except Exception as err:
        traceback.print_exc()
        print(err)

    ax.grid(True)
    ax.set_title(title)

    return merged


# %%
table_level2.unstack().fillna(0)

# %%
'''
Perform KMeans clustering on the cities by their level2 land marks.
And draw their types on the map.
'''
mat = table_level2.unstack().fillna(0).to_numpy()
num_cluster = 5
cluster = KMeans(num_cluster, n_init='auto')
cluster.fit_transform(mat)
labels = cluster.labels_
labels

df = table_level2.groupby(level=0).sum()
df['adcode'] = df.index
df['Cluster'] = [f'Cluster {e}' for e in labels]
df

dpi = 200
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)

title = 'Cluster Map'
merged1 = plot_map(df,
                   value_name='Cluster',
                   ax=ax,
                   title=title,
                   plot_kwargs=dict(
                       legend_kwds=dict(
                           loc='lower left',
                       )
                   ))
fig.tight_layout()
fig.savefig(f'image/city-fences-{title}.jpg')
plt.show()


# %% ---- 2023-04-28 ------------------------
# Pending

'''
Plot the cities at their level1 land types.
'''

# There are n=5 land types
n = 5
fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n), dpi=dpi)

for level1, ax in zip([1, 2, 3, 4, 5], axes):
    df = table.query(f'Level1 == {level1}').copy()
    df['adcode'] = [e[0] for e in df.index]
    plot_map(df, 'F_AREA', ax, land_types[level1])

fig.tight_layout()
fig.savefig(f'image/city-fences-level1.jpg')
plt.show()

# %%


# %%
