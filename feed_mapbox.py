"""
File: feed_mapbox.py
Author: Chuncheng Zhang
Date: 2023-05-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Feed dataset for mapbox display.

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-05-06 ------------------------
# Pending

from pprint import pprint
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs

from pathlib import Path
from tqdm.auto import tqdm
from download.data_reader import Dataset, dataset
from download.adcode_reader import AdcodeDataset, adcode_dataset


# %% ---- 2023-05-06 ------------------------
# Pending
table = dataset.table
land_types = dataset.land_types
land_types


# %% ---- 2023-05-06 ------------------------
# Pending
table


# %% ---- 2023-05-06 ------------------------
# Pending


# %% ---- 2023-05-06 ------------------------
# Pending
count_by_city_code_df = table.groupby('City_CODE').count()

city_dict = dict()

for city_code in tqdm(count_by_city_code_df.index, 'Loading cities'):
    adc, fen = adcode_dataset.get_city(city_code)
    if fen is not None:
        city_dict[city_code] = fen

city_dict

# %%
# features = []
# for key, value in city_dict.items():
#     features.append(value)
# topo_json = dict(
#     type='FeatureCollection',
#     features=features
# )
# path = Path('json/topoJson-chinese-city.json')
# if not path.is_file():
#     json.dump(topo_json, open(path, 'w'))

# topo_json

# gdf = gpd.GeoDataFrame.from_features(topo_json)
# gdf.crs = ccrs.PlateCarree()
# gdf.plot()

# %%
city_code = count_by_city_code_df.index[0]
city_code

# %%
# Input: city_code

land_detail_df = table.query(f'City_CODE == {city_code}')
pprint(land_detail_df.iloc[0])

feature = city_dict[city_code]
pprint(feature)

# %%
# The [e] refers there are no holes inside the polygon

for k, v in feature['properties'].items():
    feature['properties'][k] = str(v)

features = [feature]

for level in [1, 2, 3, 4, 5]:
    coordinates = [[e]
                   for e in land_detail_df.query(f'Level1 == {level}')['points']]
    multi_polygon = dict(
        type='MultiPolygon',
        coordinates=coordinates
    )
    features.append(dict(
        type='Feature',
        properties={'level': level},
        geometry=multi_polygon
    ))

feature_collection = dict(
    type='FeatureCollection',
    features=features,
)

json.dump(feature_collection, open(f'json/topoJson-{city_code}.json', 'w'))

# %%
gdf = gpd.GeoDataFrame.from_features(feature_collection)
gdf.crs = ccrs.PlateCarree()
gdf.plot(edgecolor='black', facecolor='white', alpha=0.5)
gdf

# %%
# dpi = 100
# title = f'-- {city_code} --'

# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)

# gdf.plot(ax=ax, edgecolor='black', facecolor='white', alpha=0.5)
# ax.set_title(title)

# plt.show()

# %%
feature_collection

# %%
