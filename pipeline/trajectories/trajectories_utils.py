import os
import csv
import json
import rtree
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn import metrics
from keplergl import KeplerGl
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString, MultiPoint

# add the output dir in the parameters
def parseTrajectories(filename, rootdir, condensed):
    """
    To do
    """
    counter = 0
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if (condensed):
            writer.writerow(["Start Time", "End Time", "Origin X", "Origin Y", "Dest X", "Dest Y", "Source"])
        else:
            writer.writerow(["Time", "Speed", "Heading", "Origin X", "Origin Y", "Dest X", "Dest Y", "Source"])

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                counter += 1
                path = os.path.join(subdir, file)

                if (not (".ipynb_checkpoints" in path) and not (".DS_Store" in path)):
                    with open(path) as f:
                        data = json.load(f)

                    if (condensed):
                        start, end = data['features'][0], data['features'][-1]

                        trajectory = [start['properties']['time'], end['properties']['time'], start['geometry']['coordinates'][0][0],\
                                        start['geometry']['coordinates'][0][1], end['geometry']['coordinates'][1][0],\
                                        end['geometry']['coordinates'][1][1], os.path.basename(path).split(".")[0]]

                        writer.writerow(trajectory)
                    else:
                        for feature in data['features']:

                            trajectory = [feature['properties']['time'], feature['properties']['speed'], feature['properties']['heading'],\
                                            feature['geometry']['coordinates'][0][0], feature['geometry']['coordinates'][0][1],\
                                            feature['geometry']['coordinates'][1][0], feature['geometry']['coordinates'][1][1],\
                                          os.path.basename(path).split(".")[0]]

                            writer.writerow(trajectory)
                        # print(trajectory)

        print("All trajectory data has been parsed to {0}. {1} files total.".format(filename, counter))

def clusterByZone(trajectories, zones, merge):
    # spatial join and group by to get count of incidents in each neighborhood 
    trajectories = trajectories.loc[trajectories.is_valid]
    zones = zones.loc[zones.is_valid]

    joined = gpd.sjoin(trajectories, zones, op="within")
    
    if (not merge):
        return joined
    
    grouped = joined.groupby('CentroidID').size()
    df = grouped.to_frame().reset_index()
    df.columns = ['CentroidID', 'count']
   
    merged = zones.merge(df, on='CentroidID', how='outer')
    merged['count'].fillna(0,inplace=True)
    merged['count'] = merged['count'].astype(int)
    
    return merged

def trajectoriesFromZones(gdf_origins, gdf_dests, origin_id, dest_id):
    origins = gdf_origins.loc[gdf_origins['CentroidID'] == str(origin_id)]['Source']
    dests = gdf_dests.loc[gdf_dests['CentroidID'] == str(dest_id)]['Source']
    matches = pd.merge(origins, dests, how='inner')

    df = pd.read_csv("trajectories.csv")
    geometry = [Point(xy) for xy in zip(df['Origin X'], df['Origin Y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf = gdf.groupby(['Source'])['geometry'].apply(lambda x: LineString(x.tolist()))
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    gdf.columns = ["Source"]

    trajectories = [gdf['Source'][match] for match in matches['Source']]
        
    return gpd.GeoDataFrame(geometry=trajectories)

def showTrajectoriesFromZones(origin_id, dest_id, direct):
    int_shapefile = gpd.read_file(direct + "TAZ/InternalCentroidZones.shp") # change ./.... to rootdir for DB
    ext_shapefile = gpd.read_file(direct + "TAZ/ExternalCentroidZones.shp") # change ./.... to rootdir for DB

    df = pd.read_csv("trajectories_condensed.csv")
    gdf_origins = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Origin X'], df['Origin Y']))

    df = pd.read_csv("trajectories_condensed.csv")
    gdf_dests = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Dest X'], df['Dest Y']))

    int_trajectories_origins = clusterByZone(gdf_origins, int_shapefile, merge=False)
    int_trajectories_dests = clusterByZone(gdf_dests, int_shapefile, merge=False)
    
    matching_trajectories = trajectoriesFromZones(int_trajectories_origins, int_trajectories_dests, origin_id, dest_id)
    df = pd.read_csv("trajectories_condensed.csv")
    
    chosen_zones_map = KeplerGl(height=500)
    int_zones = clusterByZone(gdf_origins, int_shapefile, merge=True)
    chosen_zones_map.add_data(data=df, name='Trajectories')
    chosen_zones_map.add_data(data=matching_trajectories, name='Matching Trajectories')
    chosen_zones_map.add_data(data=int_zones, name='Internal Zones')
    
    chosen_zones_map.save_to_html(file_name="chosen_zones_map.html")
    return chosen_zones_map