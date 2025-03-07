import os
import numpy as np
import pandas as pd
import rasterio
import pyproj
import argparse
import fiona
from shapely.geometry import Point, shape, box
from shapely.prepared import prep
from shapely.ops import transform
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def visualize_clusters_with_map(data_df, merged_clusters_map, output_map_file):
    """
    Creates a map with merged cluster bounding boxes labeled with indices
    """
    fig, ax = plt.subplots(
        figsize=(100, 160),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)

    # Plot all points in black (including noise)
    ax.scatter(
        data_df['longitude'],
        data_df['latitude'],
        s=5,
        c='black',
        alpha=0.4,
        transform=ccrs.PlateCarree()
    )

    transformer_3857_to_4326 = pyproj.Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    )

    def project_3857_to_lonlat(geom):
        return transform(transformer_3857_to_4326.transform, geom)

    # Iterate through the merged_clusters_map and plot/label bounding boxes
    index_counter = 0 # Initialize index counter
    for parent_cluster_id, nested_cluster_ids in merged_clusters_map.items():
        all_cluster_ids = [parent_cluster_id] + nested_cluster_ids
        combined_cluster_data = pd.DataFrame()
        for cluster_id in all_cluster_ids:
            cluster_data = data_df[data_df['cluster'] == cluster_id]
            combined_cluster_data = pd.concat([combined_cluster_data, cluster_data])

        if combined_cluster_data.empty:
            continue

        min_x = combined_cluster_data['x_proj'].min()
        max_x = combined_cluster_data['x_proj'].max()
        min_y = combined_cluster_data['y_proj'].min()
        max_y = combined_cluster_data['y_proj'].max()

        buffer_meters = 50000
        min_x -= buffer_meters
        max_x += buffer_meters
        min_y -= buffer_meters
        max_y += buffer_meters
        bbox_3857 = box(min_x, min_y, max_x, max_y)
        bbox_4326 = project_3857_to_lonlat(bbox_3857)

        # Plot bounding box
        ax.add_geometries(
            [bbox_4326],
            crs=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='red',
            linewidth=1.5,
            zorder=10
        )

        # Label bounding box with index
        centroid_lon, centroid_lat = bbox_4326.centroid.x, bbox_4326.centroid.y
        ax.text(
            centroid_lon,
            centroid_lat,
            str(index_counter), # Use index counter as label
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
            color='red',
            transform=ccrs.PlateCarree()
        )
        index_counter += 1 # Increment index counter

    min_lon, max_lon = data_df['longitude'].min(), data_df['longitude'].max()
    min_lat, max_lat = data_df['latitude'].min(), data_df['latitude'].max()

    lon_pad = (max_lon - min_lon) * 0.1
    lat_pad = (max_lat - min_lat) * 0.1

    ax.set_extent(
        [min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad],
        crs=ccrs.PlateCarree()
    )
    ax.set_title("DBSCAN clusters map with index labels", fontsize=14)
    plt.savefig(output_map_file, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Map saved to: {output_map_file}")


def csv_to_geotiff_dbscan(
    csv_file,
    output_dir,
    output_prefix,
    value_column_name,
    geojson_file=None,
    resolution_meters=10.0,
    target_crs_epsg=3857,
    dbscan_eps=100.0,
    dbscan_min_samples=5,
    output_map_file=None
):
    """
    DBSCAN-based CSV-to-GeoTIFF with merged nested clusters, indexed filenames,
    and labeled bounding boxes on the map
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    data_df = pd.read_csv(
        csv_file,
        na_values=['', 'NaN', 'nan', 'NULL', 'null', 'None', 'none', 'N/A', 'NA', 'na', 'n/a', '-'],
        keep_default_na=True
    )

    if value_column_name not in data_df.columns:
        print(f"Error: '{value_column_name}' not found in CSV.")
        return

    data_df[value_column_name] = pd.to_numeric(data_df[value_column_name], errors='coerce')
    data_df.dropna(subset=['longitude', 'latitude'], inplace=True)
    if data_df.empty:
        print("No valid data points after removing NaNs.")
        return

    if geojson_file:
        try:
            with fiona.open(geojson_file, 'r') as shapefile:
                gjson_geom = shape(shapefile[0]['geometry'])
                prepared_geom = prep(gjson_geom)
        except FileNotFoundError:
            print(f"Error: GeoJSON file not found at {geojson_file}")
            return
        filtered_rows = []
        for _, row in data_df.iterrows():
            if prepared_geom.contains(Point(row['longitude'], row['latitude'])):
                filtered_rows.append(row)
        data_df = pd.DataFrame(filtered_rows)
        if data_df.empty:
            print("No data inside GeoJSON geometry.")
            return
    else:
        print("No GeoJSON filter applied.")

    src_crs = rasterio.crs.CRS.from_epsg(4326)
    dst_crs = rasterio.crs.CRS.from_epsg(target_crs_epsg)
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    x_proj, y_proj = transformer.transform(
        data_df['longitude'].values,
        data_df['latitude'].values
    )
    data_df['x_proj'] = x_proj
    data_df['y_proj'] = y_proj

    coords_2d = np.column_stack((x_proj, y_proj))
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_2d)
    data_df['cluster'] = clustering.labels_

    valid_clusters = data_df[data_df['cluster'] != -1]['cluster'].unique()
    if len(valid_clusters) == 0:
        print("All points are noise (-1). No clusters to rasterize.")
        return

    cluster_bboxes_3857 = {}
    for cluster_id in valid_clusters:
        cluster_data = data_df[data_df['cluster'] == cluster_id]
        min_x = cluster_data['x_proj'].min()
        max_x = cluster_data['x_proj'].max()
        min_y = cluster_data['y_proj'].min()
        max_y = cluster_data['y_proj'].max()
        buffer_meters = 50000
        min_x -= buffer_meters
        max_x += buffer_meters
        min_y -= buffer_meters
        max_y += buffer_meters
        bbox_3857 = box(min_x, min_y, max_x, max_y)
        cluster_bboxes_3857[cluster_id] = bbox_3857

    merged_clusters_map = {}
    clusters_to_process = set(valid_clusters)

    while clusters_to_process:
        current_cluster_id = clusters_to_process.pop()
        if current_cluster_id in merged_clusters_map:
            continue

        merged_clusters_map[current_cluster_id] = []

        for other_cluster_id in list(clusters_to_process):
            if other_cluster_id == current_cluster_id:
                continue
            bbox1 = cluster_bboxes_3857[current_cluster_id]
            bbox2 = cluster_bboxes_3857[other_cluster_id]

            if bbox1.contains(bbox2):
                merged_clusters_map[current_cluster_id].append(other_cluster_id)
                clusters_to_process.remove(other_cluster_id)

    processed_clusters = set()
    index_counter = 0 # Initialize index counter for filenames

    for parent_cluster_id, nested_cluster_ids in merged_clusters_map.items():
        if parent_cluster_id in processed_clusters:
            continue

        all_cluster_ids_to_process = [parent_cluster_id] + nested_cluster_ids
        combined_cluster_data = pd.DataFrame()

        for cluster_id_to_process in all_cluster_ids_to_process:
            combined_cluster_data = pd.concat([combined_cluster_data, data_df[data_df['cluster'] == cluster_id_to_process]])
            processed_clusters.add(cluster_id_to_process)

        if combined_cluster_data.empty:
            continue

        min_x, max_x = combined_cluster_data['x_proj'].min(), combined_cluster_data['x_proj'].max()
        max_y, min_y = combined_cluster_data['y_proj'].max(), combined_cluster_data['y_proj'].min()

        buf = resolution_meters * 0.5
        min_x -= buf
        max_x += buf
        min_y -= buf
        max_y += buf

        width = int(np.ceil((max_x - min_x) / resolution_meters))
        height = int(np.ceil((max_y - min_y) / resolution_meters))

        raster_arr = np.full((height, width), np.nan, dtype=np.float32)
        transform_tile = rasterio.transform.from_origin(
            min_x, max_y, resolution_meters, resolution_meters
        )

        for _, row in combined_cluster_data.iterrows():
            val = row[value_column_name]
            if np.isnan(val):
                continue
            col, row_pix = ~transform_tile * (row['x_proj'], row['y_proj'])
            col, row_pix = int(col), int(row_pix)
            if 0 <= row_pix < height and 0 <= col < width:
                raster_arr[row_pix, col] = val

        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': raster_arr.dtype,
            'crs': dst_crs,
            'transform': transform_tile,
            'nodata': np.nan
        }

        # Use index_counter for filename
        out_tiff = os.path.join(output_dir, f"{output_prefix}_merged_tile_{index_counter:03d}.tif") # e.g., prefix_merged_tile_000.tif
        with rasterio.open(out_tiff, 'w', **profile) as dst:
            dst.write(raster_arr, 1)

        print(f"Created GeoTIFF: {out_tiff}")
        index_counter += 1

    if output_map_file:
        visualize_clusters_with_map(data_df, merged_clusters_map, output_map_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DBSCAN-based CSV-to-GeoTIFF with efficient clusters, indexed filenames, and labeled map"
    )
    parser.add_argument("csv_file", help="Path to input CSV.")
    parser.add_argument("output_dir", help="Directory for output GeoTIFF tiles.")
    parser.add_argument("output_prefix", help="Filename prefix for GeoTIFF tiles.")
    parser.add_argument("value_column", help="Name of the column to rasterize.")
    parser.add_argument("-r", "--resolution", type=float, default=10.0,
                        help="Output GeoTIFF resolution in meters (default=10).")
    parser.add_argument("-crs", "--crs_epsg", type=int, default=3857,
                        help="Target CRS EPSG code (default=3857).")
    parser.add_argument("-g", "--geojson", type=str,
                        help="Optional path to GeoJSON for spatial filtering.")
    parser.add_argument("--dbscan_eps", type=float, default=100000.0,
                        help="DBSCAN eps in meters (default=100000). e.g. 100 km")
    parser.add_argument("--dbscan_min_samples", type=int, default=1,
                        help="DBSCAN min_samples (default=1).")
    parser.add_argument("--map_output", type=str, default=None,
                        help="Optionally, saves a map (e.g. map.png).")

    args = parser.parse_args()

    csv_to_geotiff_dbscan(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        value_column_name=args.value_column,
        geojson_file=args.geojson,
        resolution_meters=args.resolution,
        target_crs_epsg=args.crs_epsg,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        output_map_file=args.map_output
    )
