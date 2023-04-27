"""
File: data_reader.py
Author: Chuncheng Zhang
Date: 2023-04-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
     Read the data into the pandas format

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-04-27 ------------------------
# Pending
import shapefile
import pandas as pd

from dbfread import DBF
from pathlib import Path
from tqdm.auto import tqdm


# %% ---- 2023-04-27 ------------------------
# Pending
folder = Path(__file__).parent


# %% ---- 2023-04-27 ------------------------
# Pending
# shape
shp = shapefile.Reader(folder.joinpath('euluc-latlonnw/euluc-latlonnw.shp'))
shapes = shp.shapes()
arr = []
for i in tqdm(range(len(shapes)), 'Reading shapes...'):
    arr.append(shapes[i].__dict__)
df1 = pd.DataFrame(arr)

# type
dbf = DBF(folder.joinpath('euluc-latlonnw/euluc-latlonnw.dbf'))
df2 = pd.DataFrame(iter(dbf))
df2

# concat
table = pd.concat([df1, df2], axis=1)
table

# %% ---- 2023-04-27 ------------------------
# level name
df3 = pd.read_excel(folder.joinpath('level-names.xlsx'))
land_types = dict()
for col in df3.columns[:2]:
    for idx in df3.index:
        v = df3.loc[idx, col]
        if pd.isnull(v):
            continue
        split = v.split(' ', 1)
        land_types[int(split[0])] = v
land_types


# %% ---- 2023-04-27 ------------------------
# Pending
class Dataset(object):
    '''
    Variable wrapper,
    - table, main table
    - land_types: the land type for Level1 and Level2
    '''

    def __init__(self, table, land_types):
        self.table = table
        self.land_types = land_types


dataset = Dataset(table, land_types)

# %%
