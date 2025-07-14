import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def sanitize_filename(s):
    return "".join(c for c in s if c.isalnum() or c in ('_', '-')).rstrip()

def plot_typhoon_track(df, filepath):
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([110, 127, 16, 33], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.gridlines(draw_labels=True)
    ax.plot(df['LON'], df['LAT'], '-o', markersize=3, label='Track')
    ax.plot(df['LON'].iat[0], df['LAT'].iat[0], 'go', label='Start')
    ax.plot(df['LON'].iat[-1], df['LAT'].iat[-1], 'rs', label='End')
    ax.set_title(f"{df['NAME'].iat[0]} ({df['SID'].iat[0]})")
    ax.legend()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def process_and_filter(df):
    # 转换时间，并只保留 1950 年及以后
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    # df = df[df['datetime'].dt.year >= 1950]
    df = df[df['ISO_TIME'].dt.year >= 1950]

    # 排除未命名
    df = df[df['NAME'].str.upper() != 'UNNAMED']

    # 删除CMA_CAT为空的/等级0/9的
    df = df[df['CMA_CAT'].notnull() & (df['CMA_CAT']!=0) & (df['CMA_CAT']!=9)]

    # 删除 USA_WIND/PRES/RMW 为空的
    # 1) 把 '--'、'' 等非数值标记先转换成 NaN
    df[['USA_WIND','USA_PRES','USA_RMW']] = df[['USA_WIND','USA_PRES','USA_RMW']].replace(['--', ''], np.nan)

    # 2) 如果它们目前还是字符串类型，把它们强制转成浮点数
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
    df['USA_PRES'] = pd.to_numeric(df['USA_PRES'], errors='coerce')
    df['USA_RMW']  = pd.to_numeric(df['USA_RMW'],  errors='coerce')

    # 3) 现在再 dropna(subset=..., how='all') 就能正确删掉三者同时缺失的行了
    before = len(df)
    df = df.dropna(subset=['USA_WIND','USA_PRES','USA_RMW'], how='all')
    after  = len(df)
    print(f"过滤掉 {before-after} 条没有 USA_WIND/PRES/RMW 的记录。")
    # 删除距离陆地距离过远的
    df = df[df['DIST2LAND'] <= 200]

    return df

def read_and_split_csv(input_csv, output_dir, min_points=12):
    # 读取这些列，包括 R34 四象限
    usecols = [
        'SID',
        'NAME','ISO_TIME','CMA_CAT','LAT','LON',
        'USA_WIND','USA_PRES','USA_RMW','DIST2LAND',
        'USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW'
    ]
    df = pd.read_csv(input_csv, usecols=usecols)

    # 过滤与处理
    df = process_and_filter(df)

    # 按 SID + NAME 分组，只保留在区域内点数 >= min_points
    storms = []
    for (sid, name), grp in df.groupby(['SID','NAME']):
        # 统计该台风在指定区域内的点数
        mask_region = (
            (grp['LAT'] >= 20) & (grp['LAT'] <= 33) &
            (grp['LON'] >= 117) & (grp['LON'] <= 123)
        )
        if mask_region.sum() >= min_points:
            storms.append(grp.reset_index(drop=True))

    return storms

def save_storms(storms, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for df in storms:
        sid = sanitize_filename(df['SID'].iat[0])
        name = sanitize_filename(df['NAME'].iat[0].replace(' ','_'))
        base = f"{sid}_{name}"
        csv_out = os.path.join(output_dir, base + '.csv')
        png_out = os.path.join(output_dir, base + '.png')
        # 保存时会包含所有字段，包括 USA_R34_NE/.../NW
        df.to_csv(csv_out, index=False)
        plot_typhoon_track(df, png_out)

if __name__=='__main__':
    input_csv = '../../data/ibtracs.WP.list.v04r01.csv'      # 你的 CSV 文件路径
    output_dir = './typhoons_from_csv'
    storms = read_and_split_csv(input_csv, output_dir, min_points=12)
    save_storms(storms, output_dir)
    print(f"共处理并保存 {len(storms)} 个台风。")
