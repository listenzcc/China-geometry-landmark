# Downloaded dataset declarations

---
- [Downloaded dataset declarations](#downloaded-dataset-declarations)
  - [Euluc latlonnw dataset](#euluc-latlonnw-dataset)
    - [The .shp file](#the-shp-file)
    - [The .dbf file](#the-dbf-file)
    - [The level-names.xlsx file](#the-level-namesxlsx-file)
  - [The adcode dataset](#the-adcode-dataset)

---

## Euluc latlonnw dataset

The dataset is downloaded from http://data.ess.tsinghua.edu.cn/.

It requires the modules to read the dataset

### The .shp file

The shape of the areas.

```shell
# Install pyshp
pip install pyshp
```

### The .dbf file

The type of the areas.

```shell
# Install dbfread
pip install dbfread
```

### The level-names.xlsx file

The table contains the meaning of the levels in .dbf file.

---
## The adcode dataset

The dataset is downloaded from https://gitee.com/waketzheng/adcode/tree/master