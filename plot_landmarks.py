"""
File: plot_landmarks.py
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
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from download.data_reader import Dataset, dataset


# %% ---- 2023-04-27 ------------------------
# Pending
table = dataset.table
land_types = dataset.land_types
table


# %%
colors = list(mcolors.TABLEAU_COLORS.values())
colors

# %% ---- 2023-04-27 ------------------------
# Pending


def plot_landmarks(df, path=None, legend_outside_flag=False, show_flag=False, land_types=land_types):
    '''
    Plot the df with its levels of land type.

    Args:
        df: the dataFrame.
        land_types: the land type dict.
        path: the path is the image output.
        legend_outside_flag: the legend_outside_flag toggles whether or not the legend is outside the box.
        show_flag: the show_flag toggles whether or not the image is shown by plt.
    '''

    xlim = (df['Lon'].min(), df['Lon'].max())
    ylim = (df['Lat'].min(), df['Lat'].max())

    levels = sorted(df['Level1'].unique())

    fig, axes = plt.subplots(
        len(levels), 1, figsize=(6, 4 * len(levels)), dpi=200)

    for lvl, ax in zip(levels, axes):
        query1 = f'Level1 == {lvl}'
        t1 = df.query(query1)

        ax.set_aspect('equal')

        for j, lvl2 in enumerate(sorted(t1['Level2'].unique())):
            query2 = f'Level2 == {lvl2}'
            t = t1.query(query2)
            c = colors[j % len(colors)]
            ax.scatter(t['Lon'], t['Lat'], label=land_types[lvl2],
                       c=c, s=t['F_AREA'], alpha=0.6)
            print(query1, query2, c)

        ax.set_title(land_types[lvl])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if legend_outside_flag:
            ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5, 0.0, 0.5))
        else:
            # ax.legend(loc='best')
            ax.legend(loc='lower left')

        # ax.grid(True)

    fig.tight_layout()

    if path is not None:
        if path.is_file():
            print(f'Warning: overwrite existing file: {path}')
        fig.savefig(path)
        print(f'Wrote image: {path}')

    if show_flag:
        plt.show()

    return path


# %% ---- 2023-04-27 ------------------------
# Pending
target_city_code = 110100
df = table.query(f'City_CODE == {target_city_code}').copy()
plot_landmarks(df, Path(f'image/{target_city_code}.jpg'), True, False)

target_city_code = 120100
df = table.query(f'City_CODE == {target_city_code}').copy()
plot_landmarks(df, Path(f'image/{target_city_code}.jpg'), True, False)

plot_landmarks(table, Path(f'image/china.jpg'), False, False)

# %%
