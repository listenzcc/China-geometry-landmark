"""
File: adcode_reader.py
Author: Chuncheng Zhang
Date: 2023-04-28
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
     Read the adcode file for cityCode

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-04-28 ------------------------
# Pending
import json
import pandas as pd
from pathlib import Path


# %% ---- 2023-04-28 ------------------------
# Pending
adcode_describe = '''
code	bigint	国家统计局12位行政区划代码
parent	bigint	12位父级行政区划代码
name	character varying(64)	行政单位名称
level	character varying(16)	行政单位级别:国/省/市/县/乡/村
rank	integer	行政单位级别{0:国,1:省,2:市,3:区/县,4:乡/镇,5:街道/村}
adcode	integer	6位县级行政区划代码
post_code	character varying(8)	邮政编码
area_code	character varying(4)	长途区号
ur_code	character varying(4)	3位城乡属性划分代码
municipality	boolean	是否为直辖行政单位
virtual	boolean	是否为虚拟行政单位，例如市辖区、省直辖县等。
dummy	boolean	是否为模拟行政单位，例如虚拟社区、虚拟村。
longitude	double precision	地理中心经度
latitude	double precision	地理中心纬度
center	geometry	地理中心, ST_Point
province	character varying(64)	省
city	character varying(64)	市
county	character varying(64)	区/县
town	character varying(64)	乡/镇
village	character varying(64)	街道/村
'''

adcode_rank_dct = {0: '国', 1: '省', 2: '市', 3: '区/县', 4: '乡/镇', 5: '街道/村'}

data_folder = Path(__file__).parent.joinpath('adcode/data')

# %% ---- 2023-04-28 ------------------------
# Pending


class AdcodeDataset(object):
    '''
    City code dataset supported by Adcode.
    '''

    def __init__(self, root, adcode_describe, adcode_rank_dct):
        '''
        Initialize the dataset by the given root

        Args:
            param: root: pathlib: The path of the directory containing 'adcode' and 'fences' folders.
            param: adcode_describe: str: The multiple lines description of the city.
            param: adcode_rank_dct: dict: The rank table of the number in adcode_describe.
        '''
        self.root = Path(root)

        col = [e.split('\t')[0].strip()
               for e in adcode_describe.split('\n')]
        self.adc_columns = [e for e in col if e.strip()]

        self.adc_rank_dct = adcode_rank_dct

        self.full_name_columns = ['province',
                                  'city', 'county', 'town', 'village']
        pass

    def get_city(self, city_code):
        '''
        Get the data for city_code.

        Args:
            param: city_code: integer: The city code in 6 digit format.

        Returns:
            adc: The dataFrame of the city information.
            fen: The geometry json of the city.
        '''
        p_adc = self.root.joinpath(f'adcode/{city_code}.csv')
        p_fen = self.root.joinpath(f'fences/{city_code}.json')

        # ----------------------------------------------------------
        for e in [p_adc, p_fen]:
            if not e.is_file():
                print(f'Warning: File not found: {e}')

        # ----------------------------------------------------------
        if p_adc.is_file():
            adc = pd.read_csv(p_adc, header=None)
            adc.columns = self.adc_columns
            adc['rank_cn'] = adc['rank'].map(
                lambda r: self.adc_rank_dct.get(r, r))

            adc['full_name_cn'] = adc.apply(
                lambda se: '-'.join([se[e] for e in self.full_name_columns if not pd.isnull(se[e])]), axis=1)

            full_name_cn = ', '.join(adc["full_name_cn"].to_list())
            print(f"Got city: {city_code}, {full_name_cn}")
        else:
            adc = None

        # ----------------------------------------------------------
        if p_fen.is_file():
            geometry = json.load(open(p_fen, 'r'))

            properties = ''
            if adc is not None:
                properties = adc.iloc[0].to_dict()

            fen = dict(
                type='Feature',
                properties=properties,
                id=city_code,
                geometry=geometry
            )

            print(f'Got fence: {city_code}')
        else:
            fen = None

        return adc, fen


adcode_dataset = AdcodeDataset(data_folder, adcode_describe, adcode_rank_dct)

# %% ---- 2023-04-28 ------------------------
# Pending
adc, fen = adcode_dataset.get_city(110100)


# # %% ---- 2023-04-28 ------------------------
# # Pending
# adc

# # %%
fen

# %%
