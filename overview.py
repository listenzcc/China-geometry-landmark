"""
File: overview.py
Author: Chuncheng Zhang
Date: 2023-04-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
     Overview of the dataset.

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-04-27 ------------------------
# Pending
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from download.data_reader import dataset


# %% ---- 2023-04-27 ------------------------
# Pending

table = dataset.table
land_types = dataset.land_types
land_types

# %%
table

# %%
land_types
level2_idx_dct = dict()
level2_lst = []
for k, v in land_types.items():
    if k > 100:
        i = len(level2_idx_dct)
        level2_idx_dct[k] = (i, k, v)
        level2_lst.append(str(k))
level2_length = len(level2_idx_dct)
level2_length, level2_idx_dct, level2_lst


# %%
group = table.groupby('City_CODE')
city_areas = group['F_AREA'].sum()

group2 = table.groupby(['City_CODE', 'Level2'])
city_level2_areas = group2['F_AREA'].sum()
city_level2_lon_lat = group2[['Lon', 'Lat']].mean()

df_city_summary = pd.concat(
    [city_level2_lon_lat, pd.DataFrame(city_level2_areas)], axis=1)
df_city_summary

# %%

# %%

mat_areas = []
city_lst = []
for cc in tqdm(city_areas.index, 'Compute cities...'):
    city_area = city_areas[cc]
    _df = df_city_summary.loc[cc]
    vec_areas = np.zeros(level2_length)
    for lvl in _df.index:
        lvl_area = _df.loc[lvl, 'F_AREA']
        vec_areas[level2_idx_dct[lvl][0]] += lvl_area/city_area
    mat_areas.append(vec_areas)
    city_lst.append(str(cc))
mat_areas = np.array(mat_areas)

mat_position = np.array([df_city_summary.loc[int(e)].mean(
    axis=0).to_numpy() for e in city_lst])[:, :2]
mat_position

print(np.sum(mat_areas, axis=1))
print(mat_areas.shape, mat_position.shape)

# -------------------------------
num_cluster = 5

cluster = KMeans(num_cluster)
cluster.fit_transform(mat_areas)
labels = cluster.labels_
cluster_area_vec_label_lst = [f'c-{e}' for e in labels]

cluster = KMeans(num_cluster)
cluster.fit_transform(mat_position)
labels = cluster.labels_
cluster_position_label_lst = [f'c-{e}' for e in labels]

cluster_position_label_lst, cluster_area_vec_label_lst, city_lst, level2_lst

# %%

# %%
lst = []
for j, city in enumerate(city_lst):
    for k, lvl in enumerate(level2_lst):
        lst.append(dict(
            city=city,
            type=lvl,
            typeLvl1=''.join(land_types[int(lvl[0])].split(' ')[:2]),
            typeLvl2=land_types[int(lvl)],
            value=mat_areas[j, k],
        ))
df_city_summary = pd.DataFrame(lst)
df_city_summary

#
fig = px.box(df_city_summary, x='typeLvl1', y='value',
             color='typeLvl1', title='Land type in lvl1')
fig.update_traces(width=0.5)
fig.show()

#
fig = px.box(df_city_summary, x='typeLvl2', y='value',
             color='typeLvl1', title='Land type in lvl2')
fig.update_traces(width=0.5)
fig.show()


# %%

category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def plot_city_land_dist(results, category_names, show_flag=True, title=None, figsize=(6, 6), save_fig_flag=True):
    """
    Parameters
    ----------
    results : dict
        A mapping with values, they determine the length of the segments.
    category_names : list of str
        The category labels.
    show_flat : boolean
        Toggle whether to show the plt.
    title : str
        The title of the graph.
    figsize : (width, height)
        The figsize of the graph.
    save_fig_flag : boolean
        Toggle whether to save the fig, only valid if the title is not None.
    """

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)

    ax.legend(ncols=1, bbox_to_anchor=(1, 1),
              loc='upper left', fontsize='small')

    if title is not None:
        ax.set_title(title)
        if save_fig_flag:
            fig.savefig(f'image/{title}.jpg')

    if show_flag:
        plt.show()

    return fig, ax


# Draw
category_names = [''.join(land_types[int(lvl)].split(' ')[:2])
                  for lvl in level2_lst]

results = dict()
for j, city in enumerate(city_lst[:10]):
    results[city] = [int(e * 100) for e in mat_areas[j]]
plot_city_land_dist(results, category_names, title='example', show_flag=True)

for j in range(num_cluster):
    cname = f'c-{j}'
    print(cname)
    select = [e == cname for e in cluster_area_vec_label_lst]

    results = dict()
    for j, city in enumerate([c for c, t in zip(city_lst, cluster_area_vec_label_lst) if t == cname]):
        results[city] = [int(e * 100) for e in mat_areas[j]]
    n = len(results)

    figsize = (6, 6 * n / 30)
    plot_city_land_dist(results, category_names,
                        title=f'{cname} ({n} cities)',
                        show_flag=True,
                        figsize=figsize
                        )

# %%

# %%
df_city_summary = pd.DataFrame(city_lst, columns=['City'])
df_city_summary['ClusterArea'] = cluster_area_vec_label_lst
df_city_summary['ClusterPos'] = cluster_position_label_lst
df_city_summary['Province'] = df_city_summary['City'].map(lambda e: e[:2])
df_city_summary['Residential'] = mat_areas[:, 0]
df_city_summary['Commercial'] = np.sum(mat_areas[:, 1:3], axis=1)
df_city_summary['Industrial'] = mat_areas[:, 3]
df_city_summary['Public'] = np.sum(mat_areas[:, -5:], axis=1)
df_city_summary

# %%


def plotly_draw(kwargs):
    # -------------------------------------
    fig = px.scatter(df_city_summary,
                     x='Residential',
                     y='Public',
                     title='Residential, Public, and Commercial',
                     **kwargs
                     )
    fig.show()

    # -------------------------------------
    fig = px.scatter(df_city_summary,
                     x='Residential',
                     y='Industrial',
                     title='Residential, Industrial, and Commercial',
                     **kwargs
                     )
    fig.show()

    # -------------------------------------
    fig = px.scatter_3d(df_city_summary,
                        x='Residential',
                        y='Public',
                        z='Industrial',
                        title='Residential, Public, Industrial, and Commercial',
                        **kwargs
                        )
    fig.show()


# ----------------------------
kwargs = dict(
    color='ClusterArea',
    size='Commercial',
    size_max=15,
    hover_data=['City'],
    width=600,
    height=600,
)

plotly_draw(kwargs)

# # ----------------------------
# kwargs = dict(
#     color='ClusterPos',
#     size='Commercial',
#     size_max=15,
#     hover_data=['City'],
#     width=600,
#     height=600,
# )

# plotly_draw(kwargs)
# %%
