# %%
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_typhoon_track(df, filepath):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([110, 127, 16, 33], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.gridlines(draw_labels=True)

    ax.plot(df['lon'], df['lat'], marker='o', color='red', linewidth=1.5, markersize=3, label='Typhoon Track')
    ax.plot(df['lon'].iloc[0], df['lat'].iloc[0], 'go', markersize=8, label='Start')
    ax.plot(df['lon'].iloc[-1], df['lat'].iloc[-1], 'bs', markersize=8, label='End')
    ax.set_title(f"{df.iloc[0]['name']} ({df.iloc[0]['sid']})", fontsize=12)

    plt.legend()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

def save_each_storm_to_csv(storms, output_dir="./typhoons_1950on"):
    os.makedirs(output_dir, exist_ok=True)
    for storm_df in storms:
        sid = sanitize_filename(storm_df.iloc[0]['sid'])
        name = sanitize_filename(storm_df.iloc[0]['name'].replace(" ", "_"))
        base_filename = f"{sid}_{name}"
        csv_path = os.path.join(output_dir, base_filename + ".csv")
        png_path = os.path.join(output_dir, base_filename + ".png")

        storm_df.to_csv(csv_path, index=False)
        plot_typhoon_track(storm_df, png_path)


def sanitize_filename(s):
    return "".join(c for c in s if c.isalnum() or c in ('_', '-')).rstrip()

def read_IBTrACS(nc_path):
    ds = Dataset(nc_path, mode='r')
    # 查看变量名
    print(ds.variables.keys())

    sid = ds.variables['sid'][:]                     # (4187,)
    name = ds.variables['name'][:]                   # (4187,)
    time = ds.variables['time'][:]                   # (4187, 360)
    lat = ds.variables['lat'][:]                     # (4187, 360)
    lon = ds.variables['lon'][:]                     # (4187, 360)
    wind = ds.variables['usa_wind'][:]               # (4187, 360)
    pressure = ds.variables['usa_pres'][:]            # (4187, 360)
    roci = ds.variables['usa_roci'][:]               # (4187, 360)
    r34 = ds.variables['usa_r34'][:]                 # (4187, 360)
    usa_status = ds.variables['usa_status'][:]           # (4187, 360)
    cma_cat = ds.variables['cma_cat'][:]           # (4187, 360)
    usa_status = np.array([[v.tobytes().decode('utf-8').strip() for v in row] for row in usa_status])
    # cma_cat = np.array([[v.tobytes().decode('utf-8').strip() for v in row] for row in cma_cat])
    base_time = datetime(1858, 11, 17)

    storms = []

    for i in range(sid.shape[0]):
        sid_str = sid[i].tobytes().decode('utf-8').replace('\x00', '').strip()
        name_str = name[i].tobytes().decode('utf-8').replace('\x00', '').strip()

        if name_str.upper() == "UNNAMED":
            continue

        times, lats, lons, winds, press, rocis, usa_types, cma_types, r34_ne, r34_se, r34_sw, r34_nw = \
                            [], [], [], [], [], [], [], [], [], [], [], []


        for j in range(time.shape[1]):
            t = time[i, j]
            if np.isnan(t) or np.isnan(lat[i, j]) or np.isnan(lon[i, j]):
                continue

            try:
                dt = base_time + timedelta(days=float(t))
            except:
                continue

            if dt.year < 1950:
                continue

            w = wind[i, j]
            p = pressure[i, j]
            r = roci[i, j]

            # 删除 wind, pressure, roci 同时为空的点
            if np.isnan(w) and np.isnan(p) and np.isnan(r):
                continue
            # 如果 wind, pressure, roci 同时不是数值，则删除该点
            if not np.isfinite(w) and not np.isfinite(p) and not np.isfinite(r):
                continue
            # 如果 wind, pressure, roci 同时为 '--', 则删除该点
            if w == '--' and p == '--' and r == '--':
                continue
            if not np.isfinite(cma_cat[i, j]) or cma_cat[i, j] == 9 or cma_cat[i, j] == 0:
                continue

            la, lo = lat[i, j], lon[i, j]
            if la > 90 or la < -90 or lo > 360 or lo < -180:
                continue

            times.append(dt)
            lats.append(la)
            lons.append(lo)
            winds.append(w)
            press.append(p)
            rocis.append(r)
            usa_types.append(usa_status[i, j])
            cma_types.append(cma_cat[i, j])
            r34_ne.append(r34[i, j, 0])
            r34_se.append(r34[i, j, 1])
            r34_sw.append(r34[i, j, 2])
            r34_nw.append(r34[i, j, 3])


        if len(times) == 0:
            continue

        # 筛选区域内轨迹点数目
        regional_points = sum(
            (20 <= la <= 33 and 117 <= lo <= 123)
            for la, lo in zip(lats, lons)
        )
        if regional_points < 13:
            continue

        df = pd.DataFrame({
            'sid': [sid_str] * len(times),
            'name': [name_str] * len(times),
            'time': times,
            'lat': lats,
            'lon': lons,
            'wind': winds,
            'pressure': press,
            'roci': rocis,
            # 'usa_type': usa_types,
            'cma_type': cma_types,
            'r34_ne': r34[i, j, 0],
            'r34_se': r34[i, j, 1],
            'r34_sw': r34[i, j, 2],
            'r34_nw': r34[i, j, 3],
})

        storms.append(df)

    ds.close()
    return storms



if __name__ == '__main__':
    file_path = "../../data/IBTrACS.WP.v04r01.nc"
    storms = read_IBTrACS(file_path)
    save_each_storm_to_csv(storms)
    print(f"已保存 {len(storms)} 个台风的CSV文件")
