"""
File: neighbors.py
Author: Chuncheng Zhang
Date: 2023-05-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Perform delaunay analysis on the land types of the cities.
    And compute the neighborhood of the lands.

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-05-04 ------------------------
# Pending

import json
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from scipy.spatial import Delaunay
from download.data_reader import Dataset, dataset
from download.adcode_reader import AdcodeDataset, adcode_dataset


# %% ---- 2023-05-04 ------------------------
# Pending
table = dataset.table
land_types = dataset.land_types
land_types

# %%


level1_idx_table = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4
}

level1_idx_table_reverse = [e
                            for e in land_types if e in level1_idx_table]

level2_idx_table = {
    101: 0,
    201: 1,
    202: 2,
    301: 3,
    401: 4,
    402: 5,
    403: 6,
    501: 7,
    502: 8,
    503: 9,
    504: 10,
    505: 11
}

level2_idx_table_reverse = [e
                            for e in land_types if e in level2_idx_table]

# %%
# Run at level 2
# level_str = 'Level2'
# idx_table = level2_idx_table
# idx_table_reverse = level2_idx_table_reverse
# num_levels = len(idx_table)

# # Run at level 1
level_str = 'Level1'
idx_table = level1_idx_table
idx_table_reverse = level1_idx_table_reverse
num_levels = len(idx_table)

# %%
table

# %% ---- 2023-05-04 ------------------------
# Pending
city_code = table['City_CODE'].to_list()
city_level = table[level_str].to_list()

lst = []
for cc in tqdm(set(city_code), 'Reading city codes'):
    adc, _ = adcode_dataset.get_city(cc)
    if adc is not None:
        lst.append(adc)

city_code_info_df = pd.concat(lst).set_index('adcode')
city_code_info_df


# %% ---- 2023-05-04 ------------------------
# Pending
position = table[['Lat', 'Lon']].to_numpy()
print(position[:4], position.shape)

delaunay = Delaunay(position)
print(delaunay)
print(delaunay.simplices.shape)
delaunay.__dict__

# %% ---- 2023-05-04 ------------------------
# Pending
df = pd.DataFrame(delaunay.simplices, columns=['a', 'b', 'c'])

for col in df.columns:
    df[col + 'CityCode'] = df[col].map(lambda e: city_code[e])
    df[col + level_str] = df[col].map(lambda e: city_level[e])

df


# %%


# Prepare dct
city_code_dct = dict()
for idx in city_code_info_df.index:
    city_code_dct[idx] = (
        city_code_info_df.loc[idx, 'province'], np.zeros((num_levels, num_levels)))
print(city_code_dct)

# Update dct
for idx in tqdm(df.index, 'Compute neighborhood.'):
    lst = list(df.loc[idx])
    _, _, _, ac, al, bc, bl, cc, cl = lst

    def _update(a, b, c, d):
        if a == b and a in city_code_dct:
            city_code_dct[a][1][idx_table[c], idx_table[d]] += 1
            city_code_dct[a][1][idx_table[d], idx_table[c]] += 1

    _update(ac, bc, al, bl)
    _update(bc, cc, bl, cl)
    _update(ac, cc, al, cl)

city_code_dct


# %%
lst = []
for key, value in city_code_dct.items():
    print(key, value[0])

    m = value[1]
    sum = np.sum(m, axis=0)

    for a in range(num_levels):
        for b in range(num_levels):
            level_a = idx_table_reverse[a]
            level_b = idx_table_reverse[b]
            v = m[a, b]
            v_ratio = m[a, b] / sum[a]
            lst.append((key, level_a, level_b, v, v_ratio))

neighbor_dataFrame = pd.DataFrame(
    lst, columns=['City_CODE', 'Src_TYPE', 'Dst_TYPE', 'Value', 'Value_ratio'])

neighbor_dataFrame['Full_Name_CN'] = neighbor_dataFrame['City_CODE'].map(
    lambda e: city_code_info_df.loc[e, 'full_name_cn'])

neighbor_dataFrame.to_csv(Path('csv/neighbor.csv'))

neighbor_dataFrame

# %%
json.dump(land_types, open(Path('json/land_types.json'), 'w'))

# %%

# %%
