'''
File: main.py
Author: Chuncheng Zhang
Date: 2023-04-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
'''


# %% ---- 2023-04-25 ------------------------
# Pending
# pip install pyshp
# http://data.ess.tsinghua.edu.cn/
import shapefile
from dbfread import DBF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

from pprint import pprint
from tqdm.auto import tqdm
from scipy.spatial import Delaunay


# %% ---- 2023-04-25 ------------------------
# shape
shp = shapefile.Reader('./euluc-latlonnw.shp')
shapes = shp.shapes()
arr = []
for i in tqdm(range(len(shapes))):
    arr.append(shapes[i].__dict__)
df1 = pd.DataFrame(arr)

# type
dbf = DBF('./euluc-latlonnw.dbf')
df2 = pd.DataFrame(iter(dbf))
df2

# level name
df3 = pd.read_excel('level-names.xlsx')
level_table = dict()
for col in df3.columns[:2]:
    for idx in df3.index:
        v = df3.loc[idx, col]
        if pd.isnull(v):
            continue
        print(col, idx, v)
        split = v.split(' ', 1)
        level_table[int(split[0])] = v
level_table

# concat
table = pd.concat([df1, df2], axis=1)
table

# %%

# %% ---- 2023-04-25 ------------------------
# Pending
table.loc[table['F_AREA'] > 1, 'F_AREA'] = 1
table

# %%
levels = sorted(table['Level1'].unique())
fig, axes = plt.subplots(len(levels), 1, figsize=(6, 4 * len(levels)))

colors = list(mcolors.TABLEAU_COLORS.values())

for lvl, ax in zip(levels, axes):
    query1 = f'Level1 == {lvl}'
    t1 = table.query(query1)

    ax.set_aspect('equal')

    for j, lvl2 in enumerate(sorted(t1['Level2'].unique())):
        query2 = f'Level2 == {lvl2}'
        t = t1.query(query2)
        c = colors[j % len(colors)]
        ax.scatter(t['Lon'], t['Lat'], label=level_table[lvl2],
                   c=c, s=t['F_AREA'], alpha=0.5)
        print(query1, query2, c)

    ax.legend(loc='lower left')
    ax.set_title(level_table[lvl])
    # ax.grid(True)


fig.tight_layout()
plt.show()
pprint(level_table)


# %%
t = table.copy()
t['type'] = t['Level1'].map(lambda e: level_table[e])
fig = px.scatter(t, x='Lon', y='Lat', color='type',
                 size='F_AREA', size_max=3, opacity=0.5)
for d in fig.data:
    d['marker']['line']['width'] = 0

fig.update_layout(legend=dict(
    orientation="h",
))

fig.show()


# %%
table


# %% ---- 2023-04-25 ------------------------
# Pending
points = np.array(table[['Lon', 'Lat']])
tri = Delaunay(points)
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.show()

# %%
tri.simplices.shape
# %%
points[tri.simplices[:, 0]]

# %%

# %%

# %%

# %%
px.scheme
# %%
