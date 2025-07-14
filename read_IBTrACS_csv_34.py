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
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = df[df['ISO_TIME'].dt.year >= 1950]
    df = df[df['NAME'].str.upper() != 'UNNAMED']
    df = df[df['CMA_CAT'].notnull() & (df['CMA_CAT'] != 0) & (df['CMA_CAT'] != 9)]

    # 替换 wind/pres/rmw 中的 '--', '' 为 NaN
    df[['USA_WIND','USA_PRES','USA_RMW']] = df[['USA_WIND','USA_PRES','USA_RMW']].replace(['--', ''], np.nan)
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
    df['USA_PRES'] = pd.to_numeric(df['USA_PRES'], errors='coerce')
    df['USA_RMW']  = pd.to_numeric(df['USA_RMW'],  errors='coerce')

    # 替换 R34 的无效值
    r34_cols = ['USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW']
    for col in r34_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    # 删除全部 wind/pres/rmw 缺失的记录
    before = len(df)
    df = df.dropna(subset=['USA_WIND','USA_PRES','USA_RMW'], how='all')
    after  = len(df)
    print(f"过滤掉 {before-after} 条没有 USA_WIND/PRES/RMW 的记录。")

    df = df[df['DIST2LAND'] <= 200]
    return df

def trim_r34_nan_edges(df):
    r34_cols = ['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']
    r34_data = df[r34_cols]
    all_nan = r34_data.isna().all(axis=1)

    # 找到第一个不全为NaN的位置
    first_valid_idx = all_nan[~all_nan].index[0] if (~all_nan).any() else None
    last_valid_idx  = all_nan[~all_nan].index[-1] if (~all_nan).any() else None

    if first_valid_idx is not None and last_valid_idx is not None:
        df = df.loc[first_valid_idx:last_valid_idx].reset_index(drop=True)
        return df
    else:
        return None  # 全为NaN就舍弃整个台风

def read_and_split_csv(input_csv, min_points=12):
    usecols = [
        'SID', 'NAME','ISO_TIME','CMA_CAT','LAT','LON',
        'USA_WIND','USA_PRES','USA_RMW','DIST2LAND',
        'USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW'
    ]
    df = pd.read_csv(input_csv, usecols=usecols)
    df = process_and_filter(df)

    storms = []
    for (sid, name), grp in df.groupby(['SID','NAME']):
        # 只保留含有至少一个R34值的台风
        r34_cols = ['USA_R34_NE','USA_R34_SE','USA_R34_SW','USA_R34_NW']
        if grp[r34_cols].notna().any(axis=1).sum() == 0:
            continue  # 所有时间都没有R34

        # 修剪掉开头结尾全部R34缺失的时刻
        trimmed = trim_r34_nan_edges(grp)
        if trimmed is None:
            continue

        # 检查有足够多的轨迹点在区域内
        mask_region = (
            (trimmed['LAT'] >= 20) & (trimmed['LAT'] <= 33) &
            (trimmed['LON'] >= 117) & (trimmed['LON'] <= 123)
        )
        if mask_region.sum() >= min_points:
            storms.append(trimmed)

    return storms

def save_storms(storms, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for df in storms:
        sid = sanitize_filename(df['SID'].iat[0])
        name = sanitize_filename(df['NAME'].iat[0].replace(' ','_'))
        base = f"{sid}_{name}"
        csv_out = os.path.join(output_dir, base + '.csv')
        png_out = os.path.join(output_dir, base + '.png')
        df.to_csv(csv_out, index=False)
        plot_typhoon_track(df, png_out)

if __name__ == '__main__':
    input_csv = '../../data/ibtracs.WP.list.v04r01.csv'
    output_dir = './typhoons_with_r34'  # 注意路径改了
    storms = read_and_split_csv(input_csv, min_points=12)
    save_storms(storms, output_dir)
    print(f"共处理并保存 {len(storms)} 个含 R34 数据的台风。")
