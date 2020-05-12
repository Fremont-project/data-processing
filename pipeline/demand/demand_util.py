# array analysis
import numpy as np
import pandas as pd
import sklearn.cluster as skc
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# geo spacial data analysis
import geopandas as gpd
from shapely import wkt
from keplergl import KeplerGl
import fiona

# assorted parsing and modeling tools
import os
import math
import csv
from pytz import utc
from shutil import copyfile, copytree
from shapely.ops import nearest_points, unary_union
from shapely.geometry import Point, LineString, Polygon, MultiPoint

import requests
import random
import polyline

from pathlib import Path


# importing all the Kepler.gl configurations
import ast

def get_configs(config_file='map_configs.txt'):
    configs = open(config_file, 'r')
    config_dict = ast.literal_eval(configs.read())
    configs.close()
    
    return config_dict
    
def write_configs(config_dict, key, config, config_file='map_configs.txt'):
    config_dict[key] = config

    print(config_dict, file=open(config_file, 'w'))

config_dict = get_configs()


def to_gdf(path):
    """
    Parameters: 
        path: path of the file to read as a geodataframe (either as csv or shp file)
        
    return:
        a GeoDataFrame (with Geopandas) corresponding to the file path
    """
    if (path.endswith('.shp')):
        gdf = gpd.GeoDataFrame.from_file(path)
        gdf = gdf.to_crs('epsg:4326')
        return gdf
    elif (path.endswith('.csv')):
        # https://geopandas.readthedocs.io/en/latest/gallery/create_geopandas_from_pandas.html#from-wkt-format
        df = pd.read_csv(path)
        df['geometry'] = df['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df)
        gdf.crs = 'epsg:4326'
        return gdf
    else:
        print('Error: Incorrect file extension {}, should be .csv or .shp'.format(path))
        
def get_nodes_with_neighborhood(nodes_path, neighborhoods_shp, kepler_config=None, debug=False):
    """
    This function adds the neighborhood on the SFCTA points.
    
    Parameters: 
        nodes_path: 
            path of the SFCTA internal demand file. Should be a csv file with columns start_node_lng, start_node_lat, end_node_lng, end_node_lat
        neighborhoods_shp:
            Neighborhoods shapefile. The file has been downloaded on http://egis-cofgis.opendata.arcgis.com/datasets/044ae5a42d924d9b94dba5c42a539c78_0
        kepler_config:
            Kepler config if you want to
        debug:
            boolean variable set by default to False to debug the function
    return:
        (if debug=False) a GeoDataFrame (with Geopandas) with the demand nodes with a column with the ID of the neighborhood that are inside.
        (if debug=True) the GeoDataFrame and a Kepler map
    """
    # Loading the SFCTA points
    nodes_df = pd.read_csv(nodes_path)

    # Transforming the points in a Geo Data Frame
    nodes = gpd.GeoDataFrame(
        nodes_df, crs='epsg:4326', geometry=gpd.points_from_xy(nodes_df.start_node_lng, nodes_df.start_node_lat))
    gdf_bis = gpd.GeoDataFrame(
        nodes_df.copy(), crs='epsg:4326', geometry=gpd.points_from_xy(nodes_df.end_node_lng, nodes_df.end_node_lat))

    # Adding the end points and the start points together, only keeping some data
    nodes = nodes.append(gdf_bis)
    nodes = nodes[['leg_id', 'start_time', 'geometry']]

    # Joining the neighborhooed on the SFCTA points
    joined_nodes = gpd.sjoin(nodes, neighborhoods_shp, how='left', op='within')
    
    ## Points that are outside Fremont are associated with neighborhood 22
    joined_nodes['OBJECTID'].fillna(22, inplace=True)

    # Transforming the geopandas shapefile to numpy array
    X = np.zeros((joined_nodes.shape[0], 3))
    X[:,0],X[:,1],X[:,2] = joined_nodes.geometry.x, joined_nodes.geometry.y, joined_nodes.OBJECTID * 100

    # Visualizing the data
    if debug:
        fremont_map = KeplerGl(height=600, config=kepler_config)
        fremont_map.add_data(data=joined_nodes, name="Nodes with neighborhoods")
        fremont_map.add_data(data=neighborhoods_shp, name="Fremont neighborhoods")
        return X, fremont_map
    return X

def taz_clustering(nb_cluster, nodes_with_nbd, kepler_config_kmeans=None, debug=False):
    """
    KMeans clustering of the SFCTA nodes within neighborhoods to create TAZs.

    Parameters: 
        nb_cluster:
            number of cluster for the kmean clustering
        nodes_with_nbd: 
            a GeoDataFrame (with Geopandas) with the demand nodes with a column with the ID of the neighborhood that are inside. This is the output of get_nodes_with_neighborhood
        kepler_config_kmeans:
            Kepler config if you want to visualize the kmeans results on the SFCTA points with neighborhoods value
        debug:
            boolean variable set by default to False to debug the function
    return:
        (if debug=False) the kmean predictor
        (if debug=True) the kmean predictor and the Kepler maps for the kmeans and the SFTCA points
    """
    # Clustering the data with k-means 
    kmeans = skc.KMeans(init='k-means++', n_clusters=nb_cluster)
    kmeans = kmeans.fit(nodes_with_nbd)
    
    if debug:
        # Visualizing the result
        labels = kmeans.predict(nodes_with_nbd)
        data_cluster = gpd.GeoDataFrame(crs='epsg:4326', geometry=gpd.points_from_xy(nodes_with_nbd[:,0], nodes_with_nbd[:,1]))
        data_cluster['labels'] = labels
        fremont_map_kmeans = KeplerGl(height=600, config=kepler_config_kmeans)
        fremont_map_kmeans.add_data(data=data_cluster, name="kMeans")
        return kmeans, fremont_map_kmeans
    return kmeans

def get_taz_from_predict(data, kmeans, neighborhoods_shp, h=.0002, kepler_config_taz=None, debug=False):
    """
    Creation of TAZ shapefile using model-based prediction
    
    Parameters: 
        data:
            data to get the min and max lat and long for the mesh
        kmeans: 
            xxx
        neighborhoods_shp:
            Neighborhoods shapefile
        h:
            Step size of the mesh
        debug:
            boolean variable set by default to False to debug the function
    return:
        (if debug=False) the TAZ dataframe
        (if debug=True) the TAZ dataframe and the Kepler map for the Internal TAZs overlaid on the City of Fremont
    """
    # Creating the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    i,j = xx.shape
    xx_col = xx.reshape(i*j,)
    yy_col = yy.reshape(i*j,)

    # Getting the labels of every point in the mesh by joining the neighborhood on the points and using the trained k-mean cluster model
    data_mesh = gpd.GeoDataFrame(crs = 'epsg:4326', geometry=gpd.points_from_xy(xx_col, yy_col))
    data_mesh = gpd.sjoin(data_mesh, neighborhoods_shp, how='left', op='within')
    
    ## Mesh point that ae outside Fremont are associated with neighborhood 22
    data_mesh['OBJECTID'].fillna(22, inplace=True)
    
    data_mesh = data_mesh[['geometry', 'OBJECTID']]

    point_test = np.zeros((i*j,3))
    point_test[:,0] = xx_col
    point_test[:,1] = yy_col
    point_test[:,2] = data_mesh['OBJECTID'].astype(int) * 100

    labels_mesh = kmeans.predict(point_test)

    # Creating squares from the mesh points
    mesh_squares = data_mesh.geometry.buffer(h/1.99).envelope
    internal_taz = gpd.GeoDataFrame(labels_mesh, crs='epsg:4326', geometry=mesh_squares)

    # Merging squares with similar labels to create the internal TAZs
    internal_taz= internal_taz.rename(columns={0: "label"})
    internal_taz = internal_taz.dissolve(by='label')
    internal_taz['label'] = internal_taz.index
    if debug:
        # Visualization
        fremont_map = KeplerGl(height=600, config=kepler_config_taz)
        fremont_map.add_data(data=internal_taz, name="Internal TAZ")
        fremont_map.add_data(data=neighborhoods_shp, name="Fremont neighborhoods")
        return internal_taz, fremont_map
    return internal_taz

def get_taz(nodes_path, neighborhoods_path, ext_taz_path, output_taz, debug=False):
    """
    Generate and crop the internal TAZs with the external TAZs and return the shapefiles for each
    
    Parameters: 
        nodes_path: 
            path to the SFCTA internal demand file. Should be a csv file with columns start_node_lng, start_node_lat, end_node_lng, end_node_lat
        neighborhoods_path:
            path to the Neighborhoods shapefile
        ext_taz_path:
            path to the External TAZs shapefile
        output_taz:
            root for output path for TAZ shapefiles
        debug:
            boolean variable set by default to False to debug the function
    return:
        (if debug=False) the TAZ dataframe
        (if debug=True) the TAZ dataframe and the Kepler map for the Internal TAZs overlaid on the City of Fremont
    """
    # Loading neighborhoods
    print('Loading neighborhoods...')
    neighborhoods_shp = to_gdf(neighborhoods_path)
    
    print('Joining neighborhoods on SFCTA nodes...')
    if debug:
        nodes_with_nbd, fremont_map_nbd = get_nodes_with_neighborhood(nodes_path, neighborhoods_shp, kepler_config=config_dict['Neighborhoods'], debug=True)
    else:
        nodes_with_nbd = get_nodes_with_neighborhood(nodes_path, neighborhoods_shp)
    
    print('Clustering the SFCTA nodes...')
    # Remark:
    # Both spectral clustering and DBSCAN does not give explicit decision boundaries, therefore we did not use them to create the internal centroids.
    # One can see if there is a better way to do the clustering (with something else than KMeans).
    nb_cluster = 100
    if debug:
        kmeans, fremont_map_kmeans = taz_clustering(nb_cluster, nodes_with_nbd, kepler_config_kmeans=config_dict['Kmeans'], debug=True)
    else:
        kmeans = taz_clustering(nb_cluster, nodes_with_nbd)

    print("Creating the TAZs...")
    if debug:
        internal_taz, fremont_map_TAZs = get_taz_from_predict(nodes_with_nbd, kmeans, neighborhoods_shp, kepler_config_taz=config_dict['Internal_TAZ_mesh'], debug=True)
    else:
        internal_taz = get_taz_from_predict(nodes_with_nbd, kmeans, neighborhoods_shp)
    
    print("Cropping and writing the TAZs...")
    # Cropping the internal TAZs with the external TAZs
    ext_taz = gpd.read_file(ext_taz_path)
    new_internal_taz = gpd.overlay(internal_taz, ext_taz, how='difference').drop('label', axis=1)

    ext_taz['CentroidID'] = 'ext_' + ext_taz['CentroidID'].astype(str)
    ext_taz = ext_taz[['CentroidID', 'geometry']]
    
    new_internal_taz.to_file(output_taz + "/Internal_TAZ.shp")
    
    new_internal_taz = to_gdf(output_taz + "/Internal_TAZ.shp")
    new_internal_taz['CentroidID'] = "int_" + new_internal_taz['label'].astype(str)
    new_internal_taz = new_internal_taz[['CentroidID', 'geometry']]
    new_internal_taz.to_file(output_taz + "/Internal_TAZ.shp")
    
    ext_taz.to_file(output_taz + "/External_TAZ.shp")
    
    print("\nTAZ Generation Complete!\n")
    if debug:
        fremont_map_taz = KeplerGl(height=600)
        fremont_map_taz.add_data(data=new_internal_taz, name="Internal TAZs")
        fremont_map_taz.add_data(data=ext_taz, name="External TAZs")
        return new_internal_taz, ext_taz, fremont_map_nbd, fremont_map_kmeans, fremont_map_TAZs, fremont_map_taz
    return new_internal_taz, ext_taz

def loading_taz(aux_input_taz, nodes_path="", neighborhoods_path="", ext_taz_path="", output_taz="", processing=False):
    """
    Load TAZs into GeoDataFrames, either from existing input path or by generating them if no such path provided.
    
    Parameters: 
        aux_input_taz:
            path to directory containing shapefiles for existing External and Internal TAZs
        nodes_path: 
            path to the SFCTA internal demand file. Should be a csv file with columns start_node_lng, start_node_lat, end_node_lng, end_node_lat
        neighborhoods_path:
            path to the Neighborhoods shapefile
        ext_taz_path:
            path to the External TAZs shapefile
        output_taz:
            root for output path for TAZ shapefiles
        processing:
            boolean variable set by default to False to generate new TAZs
    return:
        (if processing=False) two GeoDataFrames containing the Internal and External TAZs
        (if processing=True) two GeoDataFrames containing the Internal and External TAZs, 
            Kepler neighborhood map, KMeans SFCTA Clustering map, KMeans TAZ map, Final TAZ Map
    """
    if processing:
        return get_taz(nodes_path, neighborhoods_path, ext_taz_path, output_taz, debug=True)
    internal_taz = to_gdf(aux_input_taz + "/Internal_TAZ.shp")
    ext_taz = to_gdf(aux_input_taz + "/External_TAZ.shp")
    return internal_taz, ext_taz

def clean_ext_cen_shapefile(ext_taz_path, output_taz):
    """
    This function simply adds the prefix 'ext_' to the external centroid IDs and save the new shapefile in output_taz folder
    
    Parameters: 
        ext_taz_path:
            path to the External TAZs shapefile
        output_taz:
            root for output path for TAZ shapefiles
            
    return:
        None. Writes directly to specified directory.
    """
    ext_shapefile = to_gdf(ext_taz_path)
    ext_shapefile = ext_shapefile.rename(columns={'Centroid_I' : "CentroidID"})[['CentroidID', 'geometry']]
    ext_shapefile['CentroidID'] = 'ext_' + ext_shapefile['CentroidID'].astype(str)
    ext_shapefile.to_file(output_taz + "/External_centroids.shp")
    
def get_internal_centroid_connection(section_path, output_taz, debug=False):
    """
    This method determines all internal centroid connections from an input road network and shapefile of Internal TAZs.
    It also merges any Internal TAZs without their own centroid connections with a specific neighbor. Returns updated
    TAZ shapefile GeoDataFrame as well as centroid connections as dictionaries. Optionally returns a visual rendering
    of the centroid connection *road sections* in KeplerGl, if debug=True.
    
    This function requires the Internal TAZs to have been pre-generated and input as a path to the 
    associated shapefile in output_taz.
    
    This function requires the Road Network to have been pre-generated and input as a path to the 
    associated shapefile in section_path.
    
    Parameters: 
        section_path:
            path to the Road Network shapefile
        output_taz:
            path to the Internal TAZs shapefile
        debug:
            boolean variable set by default to False to debug the function
    return:
        (if debug=False) the Internal TAZ GeoDataFrame, and Centroid Connection and Road Section dictionaries
        (if debug=True) the Internal TAZ GeoDataFrame, Centroid Connection and Road Section dictionaries, and Kepler map
    """
    # loading the data
    sections_gdf = to_gdf(section_path)
    int_taz_shapefile, _ = loading_taz(output_taz)
    
    #########################################################################
    #########################################################################
    ################# LINE TO CHANGE ONCE EXTERNAL TAZ DONE #################
    ## Quick fix while working on the external TAZs!!!
    int_taz_shapefile = int_taz_shapefile[int_taz_shapefile.CentroidID != 'int_68']
    int_taz_shapefile = int_taz_shapefile.reset_index(drop=True)
    #########################################################################
    #########################################################################
    #########################################################################
    
    # Find all road sections in road network with either no fnode or no tnode
    no_fnode = sections_gdf[sections_gdf['fnode'].map(math.isnan) == True]
    no_tnode = sections_gdf[sections_gdf['tnode'].map(math.isnan) == True]
    
    # initialize dictionaries for storing centroids containing each no f/t-node road section (centroid connection)
    centroid_con_nof = {}
    centroid_con_not = {}

    for i in range(len(int_taz_shapefile['geometry'])):
        centroid_con_nof[int_taz_shapefile['CentroidID'][i]] = []
        centroid_con_not[int_taz_shapefile['CentroidID'][i]] = []
        
        
    # update dictionaries with centroid:no f/t-node road section key:value pairs
    no_fnode = no_fnode.reset_index()
    for j in range(len(no_fnode['geometry'])):
        for i in range(len(int_taz_shapefile['geometry'])):
            if int_taz_shapefile['geometry'][i].contains(no_fnode['geometry'][j].centroid):
                centroid_con_nof[int_taz_shapefile['CentroidID'][i]].append(no_fnode['eid'][j])
                break

    no_tnode = no_tnode.reset_index()
    for j in range(len(no_tnode['geometry'])):
        for i in range(len(int_taz_shapefile['geometry'])):
            if int_taz_shapefile['geometry'][i].contains(no_tnode['geometry'][j].centroid):
                centroid_con_not[int_taz_shapefile['CentroidID'][i]].append(no_tnode['eid'][j])
                break
    
    # update Internal TAZs and centroid connection dictionaries by merging TAZs with no centroid connections
    int_taz_shapefile, centroid_con_nof, centroid_con_not = merge_taz_cc(int_taz_shapefile, centroid_con_nof, centroid_con_not)
          
    # if debug set to true, generate and return visual rendering of centroid connection *road sections* and Internal TAZs
    if debug:
        icz_map = KeplerGl(height=500)
        icz_map.add_data(data=no_fnode, name='No-From Road Sections')
        icz_map.add_data(data=no_tnode, name='No-To Road Sections')
        icz_map.add_data(data=sections_gdf, name='Road Sections')
        icz_map.add_data(data=int_taz_shapefile, name='Internal Centroid Zones')
        
        return int_taz_shapefile, centroid_con_nof, centroid_con_not, no_fnode, no_tnode, icz_map
                
    return int_taz_shapefile, centroid_con_nof, centroid_con_not, no_fnode, no_tnode

def merge_taz_cc(int_taz_shapefile, centroid_con_nof, centroid_con_not):
    """
    This method merrges all TAZs with no centroid connections with one of that TAZ direct neighbors
    (specifically, the neighbor with the most centroid connections of its own).
    
    Parameters: 
        int_taz_shapefile:
            the Internal TAZ GeoDataFrame
        centroid_con_nof:
            the Internal TAZ : Centroid Connection (No F-Node) Road Sections dictionary
        centroid_con_not:
            the Internal TAZ : Centroid Connection (No T-Node) Road Sections dictionary
    return:
        int_taz_shapefile:
            the updated (post-merging) Internal TAZ GeoDataFrame
        centroid_con_nof:
            the updated (post-merging) Internal TAZ : Centroid Connection (No F-Node) Road Sections dictionary
        centroid_con_not:
            the updated (post-merging) Internal TAZ : Centroid Connection (No T-Node) Road Sections dictionary
    """
    def update_neighbors(int_taz_shapefile):
        """
        This method updates the input Internal TAZ shapefile GeoDataFrame with the neighbors for each TAZ, and also
        returns them as a dictionary.
        
        Parameters: 
            int_taz_shapefile:
                the Internal TAZ GeoDataFrame
        return:
            int_taz_shapefile:
                the updated (post-neighbor-detection) Internal TAZ GeoDataFrame
            neighbors_dict:
                dictionary containing TAZ : Neighbors pairs for all Internal TAZs
        """
        # initialize dictionary for storing TAZ:Neighbors key:value pairs
        neighbors_dict = {}
        
        # add or update the NEIGHBORS column in the Internal TAZ shapefile's GeoDataFrame
        if not "Neighbors" in int_taz_shapefile.columns:
            int_taz_shapefile["Neighbors"] = None  
        else:
            int_taz_shapefile = int_taz_shapefile.drop(columns=["Neighbors"])
        
        # Find and update neighbors in neighbors_dict and Neighbors column in Internal TAZ shapefile's GeoDataFrame
        for index, centroid in int_taz_shapefile.iterrows():   
            # get 'not disjoint' TAZs
            neighbors = int_taz_shapefile[~int_taz_shapefile.geometry.disjoint(centroid.geometry)].CentroidID.tolist()
            # remove source TAZ from the list
            neighbors = [ cid for cid in neighbors if centroid.CentroidID != cid ]
            # add to neighbors dictionary for within-method reference
            neighbors_dict[str(centroid.CentroidID)] = neighbors
            # add Centroid IDs of neighbors as NEIGHBORS value
            int_taz_shapefile.at[index, "Neighbors"] = ", ".join(neighbors)
            
        return int_taz_shapefile, neighbors_dict
    
    # Update int_taz_shapefile with neigbors for each TAZ and retreive associated TAZ:Neighbors dictionary
    int_taz_shapefile, neighbors_dict = update_neighbors(int_taz_shapefile)
    
    # Initialize array to keep track of changes in int_taz_shapefile during merging process
    changed = []

    # Merge any TAZ with no centroid connections with one of its direct neighbors (the neighbor with the most connections)
    for index in int_taz_shapefile.index:
        cid = int_taz_shapefile.at[index, "CentroidID"]
        if not centroid_con_nof[cid] or not centroid_con_not[cid]:
            max_nghbrs = [(nghbr, len(centroid_con_nof.get(nghbr, [])) + len(centroid_con_not.get(nghbr, [])))\
                          for nghbr in neighbors_dict[cid]]
            max_nghbr = max(max_nghbrs, key=lambda x:x[1])[0]
            merged_polygon = int_taz_shapefile.loc[int_taz_shapefile["CentroidID"] == max_nghbr].geometry.union(int_taz_shapefile.at[index, "geometry"])
            
            int_taz_shapefile = int_taz_shapefile[int_taz_shapefile.CentroidID != cid]
            int_taz_shapefile.loc[int_taz_shapefile["CentroidID"] == max_nghbr, "geometry"] = merged_polygon
            int_taz_shapefile.loc[int_taz_shapefile["CentroidID"] == max_nghbr, "CentroidID"] = cid + max_nghbr

            centroid_con_nof[cid + max_nghbr] = centroid_con_nof[cid] + centroid_con_nof[max_nghbr]
            centroid_con_not[cid + max_nghbr] = centroid_con_not[cid] + centroid_con_not[max_nghbr]
            
            changed.append(cid)
            changed.append(max_nghbr)
            
            if changed:
                int_taz_shapefile, neighbors_dict = update_neighbors(int_taz_shapefile)
    
    # Make data structure adjustments if any merging occurred
    if changed:
        # Cleanup centroid connection dictionaries in accordance with changes made during merging process
        for change in changed:
            centroid_con_nof.pop(change)
            centroid_con_not.pop(change)
        
        # Re-index and re-label TAZs for numerical consistency after merging, update dictionaries with new labels
        int_taz_shapefile = int_taz_shapefile.reset_index(drop=True)
        old_cids = int_taz_shapefile['CentroidID'][:]
        cid_map = {}
        for i in range(len(old_cids)):
            old_cid, new_cid = int_taz_shapefile['CentroidID'][i], "int_{}".format(i)
            cid_map[old_cid] = new_cid
            centroid_con_nof[new_cid], centroid_con_not[new_cid] = centroid_con_nof.pop(old_cid), centroid_con_not.pop(old_cid)
        int_taz_shapefile['CentroidID'] = int_taz_shapefile['CentroidID'].map(cid_map)
        int_taz_shapefile, _ = update_neighbors(int_taz_shapefile)
                
    return int_taz_shapefile, centroid_con_nof, centroid_con_not

def road_section_to_external_centroid(cid_geo, sections_df):
    """
    This method determines the closest road section from an input GeoDataFrame of road sections to the input centroid.
    
    Parameters:
        cid_geo:
            A shapely object of an external centroid's geometry
        sections_df:
            An input GeoDataFrame of road sections
    return:
        The road section id closes to the input external centroid cid_geo
    """
    bestId = None
    bestVal = float('inf')
    for j in range(len(sections_df['geometry'])):
        currVal = cid_geo.distance(nearest_points(sections_df['geometry'][j], cid_geo)[0])
        if (currVal < bestVal):
            bestVal = currVal
            bestId = sections_df['eid'][j]
    return bestId

def convert_and_find_centroids(coords, int_taz_shapefile, ext_cen_shapefile, no_fnode, no_tnode):
    """
    This method converts input data structures to desired coordinate system and finds centroids for all input TAZs.
    
    Parameters: 
        coords:
            the desired coordinate system to convert the other inputs to
        int_taz_shapefile:
            the Internal TAZ GeoDataFrame
        ext_cen_shapefile:
            the External Centroid GeoDataFrame
        no_fnode:
            the Centroid Connection (No F-Node) Road Sections dataframe
        no_tnode:
            the Centroid Connection (No T-Node) Road Sections dataframe
    return:
        int_taz_shapefile:
            the Internal TAZ GeoDataFrame converted to coords
        ext_cen_shapefile:
            the External Centroid GeoDataFrame converted to coords
        no_fnode:
            the Centroid Connection (No F-Node) Road Sections dataframe converted to coords
        no_tnode:
            the Centroid Connection (No T-Node) Road Sections dataframe converted to coords
        centroid_gravity:
            dictionary of centroids for all Internal/External TAZs
    """
    ext_cen_shapefile = ext_cen_shapefile.to_crs(coords)
    int_taz_shapefile = int_taz_shapefile.to_crs(coords)
    no_fnode = no_fnode.to_crs(coords)
    no_tnode = no_tnode.to_crs(coords)
    
    centroid_gravity = {}
    int_taz_shapefile = int_taz_shapefile.reset_index(drop=True)
    for i in range(len(int_taz_shapefile['geometry'])):
        centroid_gravity[int_taz_shapefile['CentroidID'][i]] = int_taz_shapefile['geometry'][i].centroid
        
    return int_taz_shapefile, ext_cen_shapefile, no_fnode, no_tnode, centroid_gravity
    

def write_centroid_connections(output_path, section_path, output_taz):
    """
    This method generates and writes the centroid connections for the Internal TAZs stored at path given by output_taz 
    and the Road Network stored at output_path to a CSV at output_path, and creates and returns a visual KeplerGl 
    rendering of the centroid connections among Internal TAZs, full road network, and internal/external centroids.

    Parameters: 
        output_path:
            path to directory where centroid connections CSV is to be written
        section_path:
            path to the Road Network shapefile
        output_taz:
            path to the Internal TAZs shapefile
    return:
        icz_map:
            KeplerGl map of Internal TAZs and no f/t-node road sections
        connections_map:
            KeplerGl map of the centroid connections among Internal TAZs, road network, and internal/external centroids
    """
    # loading the data
    sections_gdf = to_gdf(section_path)
#     int_taz_shapefile, _ = loading_taz(output_taz)
    ext_cen_shapefile = to_gdf(output_taz + "/External_centroids.shp")
    
    #get internal centroid connections
    int_taz_shapefile, centroid_con_nof, centroid_con_not, no_fnode, no_tnode, icz_map = get_internal_centroid_connection(section_path, output_taz, debug=True)

    # Visual rendering of centroid connections among Internal TAZs, full road network, and internal/external centroids
    centroid_gravity = convert_and_find_centroids("EPSG:4326", int_taz_shapefile, ext_cen_shapefile, no_fnode, no_tnode)[4]
    connections = []
    centroids_gdf = gpd.GeoDataFrame(geometry=list(centroid_gravity.values()))
    centroids_gdf["cid"] = list(centroid_gravity.keys())
    centroids_gdf = centroids_gdf[["cid", "geometry"]]
    for cid in centroid_gravity.keys():
        for centroids_list in [centroid_con_nof, centroid_con_not]:
            for eid in centroids_list[cid]:
                cid_center = centroid_gravity[cid]
                connection_center = sections_gdf.loc[sections_gdf['eid'] == eid, 'geometry'].iloc[0].centroid
                connections.append(LineString([cid_center, connection_center]))
    connects_gdf = gpd.GeoDataFrame(geometry=connections)
    connections_map = KeplerGl(height=500, config=config_dict['Centroid Connections'])
    connections_map.add_data(data=centroids_gdf, name='Internal TAZ Centroids')
    connections_map.add_data(data=ext_cen_shapefile, name='External Centroids')
    connections_map.add_data(data=connects_gdf, name='Centroid Connections')
    connections_map.add_data(data=sections_gdf, name='Road Network')
    connections_map.add_data(data=int_taz_shapefile, name='Internal TAZs')
    
    # converting the coordinate system to the one use by Aimsun software     
    int_taz_shapefile, ext_cen_shapefile, no_fnode, no_tnode, centroid_gravity = convert_and_find_centroids("EPSG:32610", int_taz_shapefile, ext_cen_shapefile, no_fnode, no_tnode)
    
    # Writing the new TAZs
    int_taz_shapefile.to_file(output_taz + "/Internal_TAZ.shp")
    
    # Writing centroid connections data to CSV file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["CentroidID", "Centroid Type", "Center Lon", "Center Lat", "From Connection IDs", "To Connection IDs"])


        for cid in int_taz_shapefile['CentroidID']:
            cid = str(cid)
            ccnof = ", ".join(str(i) for i in centroid_con_nof[cid])
            ccnot = ", ".join(str(i) for i in centroid_con_not[cid])
            connection = [cid, "Internal", centroid_gravity[cid].x, centroid_gravity[cid].y, ccnof, ccnot]

            writer.writerow(connection)

        for i in range(len(ext_cen_shapefile['CentroidID'])):
            cid = str(ext_cen_shapefile['CentroidID'][i])
            cid_geo = ext_cen_shapefile['geometry'][i]
            nearest_no_f = road_section_to_external_centroid(cid_geo, no_fnode)
            nearest_no_t = road_section_to_external_centroid(cid_geo, no_tnode)
            connection = [cid, "External", cid_geo.x, cid_geo.y, nearest_no_f, nearest_no_t]

            writer.writerow(connection)
    
    # return KeplerGl visual rendering maps
    return icz_map, connections_map

def get_sfcta_dataframe(int_int_path, int_ext_path, ext_int_path):
    """
    This method reads and appropriately merges the files of the SFCTA demand data.
    """
    int_int_trips = pd.read_csv(int_int_path)
    int_ext_trips = pd.read_csv(int_ext_path)
    ext_int_trips = pd.read_csv(ext_int_path)
    internal_trips = pd.DataFrame.merge(int_int_trips, int_ext_trips, 'outer')
    internal_trips = pd.DataFrame.merge(internal_trips, ext_int_trips, 'outer')
    return internal_trips

def join_taz_on_sfcta_data(internal_trips, tazs):
    """
    This method joins the TAZs on the SFCTA trip legs.
    """
    
    internal_trips_start = gpd.GeoDataFrame(internal_trips, crs='epsg:4326', geometry=gpd.points_from_xy(internal_trips.start_node_lng, internal_trips.start_node_lat))

    internal_trips_start = gpd.sjoin(tazs, internal_trips_start, how='right', op='contains')
    internal_trips_start.rename(columns={"CentroidID": "CentroidID_O"},inplace=True)

    internal_trips_start = internal_trips_start[['leg_id', 'start_time', 'start_node_lat', 'start_node_lng', 'end_node_lat', 'end_node_lng', 'CentroidID_O']]

    internal_trips_end = gpd.GeoDataFrame(internal_trips_start, crs='epsg:4326', geometry=gpd.points_from_xy(internal_trips_start.end_node_lng, internal_trips_start.end_node_lat))

    internal_trips_end = gpd.sjoin(tazs, internal_trips_end, how='right', op='contains')
    internal_trips_end.rename(columns={"CentroidID": "CentroidID_D"},inplace=True)

    internal_trips_end = internal_trips_end[['leg_id', 'start_time', 'start_node_lat', 'start_node_lng', 'end_node_lat', 'end_node_lng', 'CentroidID_O', 'CentroidID_D']]
    
    return internal_trips_end

def shift_time(demand_df, column, time_to_shift):
    demand_df[column] = demand_df[column].apply(lambda x: pd.Timestamp(x))
    demand_df[column] += pd.to_timedelta(time_to_shift, unit='h')
    demand_df[column] = demand_df[column].apply(lambda x: x.time())

def cluster_demand_15min(df):
    """
    Return an origin-destination matrix as pandas dataframe.
    
    -----------------------------------------------
    | CentroidID_O | CentroidID_D | dt_15 | count |
    -----------------------------------------------
    
    Parameters
    ----------
    df : DataFrame
        DataFrame representing OD matrix
    """
    demand_df = df.copy()
    demand_df['start_time'] = demand_df['start_time'].apply(lambda x: str(x.replace(minute=int(x.minute/15)*15,second = 0)))
    demand_df = demand_df.groupby(['CentroidID_D', 'CentroidID_O', 'start_time']).size().reset_index(name='counts')
    demand_df.rename(columns={"start_time": "dt_15"},inplace=True)
    return demand_df


def export_all_demand_between_centroids(concatenated_matrices, output):
    """
    This method exports datasets with demand between centroids to CSV in directory specified by `output` parameter.
    """
    # Create folder if it doesn't exist
#     Path(output).mkdir(parents=True, exist_ok=True)

#     for timestamp in concatenated_matrices['dt_15'].unique():
#         demand_matrices = concatenated_matrices.groupby('dt_15').get_group(timestamp)
#         demand_matrices = pd.pivot_table(demand_matrices, values='counts', index='CentroidID_O', columns='CentroidID_D', aggfunc=np.sum)
#         demand_matrices = demand_matrices.fillna(0)
#         demand_matrices.insert(0, '', value=demand_matrices.index)
#         demand_matrices.to_csv(output+'/'+ timestamp.replace(':', '-')+'.csv', index=False)
    concatenated_matrices.to_csv(output + ".csv")
    print('Datasets with demand between centroids exported.')
    
def process_SFCTA_data(int_int_path, int_ext_path, ext_int_path, output_taz, output_int_demand_path):
    """
    This method performs all processing steps on SFCTA data. Loads trips (given by int_int_path, int_ext_path, 
    ext_int_path) into a dataframe, merges appropriate TAZs, and exports grouped demand in an Aimsun-friednly format at 
    output_int_demand_path.
    
    @param int_int_path:          To do
    """
    print("Loading SFCTA trips...")
    internal_trips = get_sfcta_dataframe(int_int_path, int_ext_path, ext_int_path)
    print(str(internal_trips.leg_id.count()) + " trips")
    
    ## Depending on the data, the time might need to be shifted
    print("Shifting time...")
    shift_time(internal_trips, 'start_time', 0)

    print("Loading TAZs...")
    # Merging internal and external TAZs
    internal_taz, external_taz = loading_taz(output_taz)
    tazs = pd.DataFrame.merge(internal_taz, external_taz, 'outer')

    print("Joining SFCTA trips on TAZs...")
    internal_trips_end = join_taz_on_sfcta_data(internal_trips, tazs)
    print(str(internal_trips_end.leg_id.count()) + " trips")

    print("Grouping demand per 15 minutes time step...")
    grouped_od_demand_15min = cluster_demand_15min(internal_trips_end)
    print(str(grouped_od_demand_15min.counts.sum()) + " trips")
    
    print("Exporting SFCTA demand with format for Aimsun...")
    export_all_demand_between_centroids(grouped_od_demand_15min, output_int_demand_path)
    print("Processing of SFCTA data finished. The output files are located in " + output_int_demand_path)

def get_external_demand_streetlight(streetlight_path, flow_path, output_ext_demand_path, pems_nb_id = 414016, pems_sb_id = 403226):
    """
    This method performs External OD Demand inference from input Streetlight data.
    pems_nb_id and pems_sb_id are the id of the chosen pems detector used to estimate the distribution for inference.
    """
    # loading streetlight data
    ext_ext_OD_AM_PM = pd.read_csv(streetlight_path)

    # loading flow data
    flow_pems = pd.read_csv(flow_path)

    # extract flow for the pems detector and compute the distribution
    flow_nb = flow_pems[flow_pems['Detector_Id']==pems_nb_id].loc[:, '14:00': '20:00']
    flow_sb = flow_pems[flow_pems['Detector_Id']==pems_sb_id].loc[:, '14:00': '20:00']
    flow_nb_percent = flow_nb / int(flow_nb.sum(axis=1))
    flow_sb_percent = flow_sb / int(flow_sb.sum(axis=1))

    # create new columns
    for col in flow_nb.columns.tolist():
        ext_ext_OD_AM_PM[col]=0

    # demand that has external centriod 13 as their origin
    start_with_13 = (ext_ext_OD_AM_PM['Centroid_O']==13)
    # demand that has external centriod 13 as their destination
    end_with_13 = (ext_ext_OD_AM_PM['Centroid_D']==13)

    for col in flow_nb.columns.tolist():
        ext_ext_OD_AM_PM.loc[start_with_13, col]=float(flow_sb_percent[col])*ext_ext_OD_AM_PM[start_with_13]['PM']
        ext_ext_OD_AM_PM.loc[end_with_13, col]=float(flow_nb_percent[col])*ext_ext_OD_AM_PM[end_with_13]['PM']

    ext_ext_OD_AM_PM['Centroid_O'] = 'ext_' + ext_ext_OD_AM_PM['Centroid_O'].astype(str)
    ext_ext_OD_AM_PM['Centroid_D'] = 'ext_' + ext_ext_OD_AM_PM['Centroid_D'].astype(str)

    ext_ext_OD = ext_ext_OD_AM_PM.melt(["Centroid_O", "Centroid_D"])
    ext_ext_OD = ext_ext_OD.drop(ext_ext_OD[(ext_ext_OD['variable']=='AM') | (ext_ext_OD['variable']=='PM')].index)
    ext_ext_OD = ext_ext_OD.rename(columns = {"Centroid_O": "CentroidID_O", "Centroid_D": "CentroidID_D", 'variable': 'dt_15', 'value': 'counts'})
    ext_ext_OD['counts'] = ext_ext_OD['counts'].astype(int)

    export_all_demand_between_centroids(ext_ext_OD, output_ext_demand_path)

def create_shp_file_by_changing_speeds_of_roads(dir_sections, dir_output):
    """
    @todo This is not done since Google Maps API needs a Asset Tracking license for access
    Creates new shp file by copying the contents of sections.shp file contained in dir_sections.
     The new shp file has new speed limits obtained from Google Maps API. We only keep major roads,
     highways and on/off ramps. Major roads are found by roads having names in sections.shp and highways
     and on/off-ramps are found by having a speed limit >= 100 km/hr

     @param dir_sections    folder containing sections.shp
     @param dir_output      folder writing the new shp file
    """
    api_key = "AIzaSyAfHjYZKsMAdqpvvdIAPjaWcwEJU-3b77E"
    shape_file = os.path.join(dir_sections, 'sections.shp')
    output_file = os.path.join(dir_output, 'sections_updated_speeds.shp')
    with fiona.collection(shape_file, 'r') as input:
        schema = input.schema.copy()
        with fiona.collection(output_file, 'w', 'ESRI Shapefile', schema, crs=input.crs) as output:
            for row in input:
                # create copy of row to avoid segmentation error (we will write to copy and save it)
                row_copy = row.copy()
                # get name and speed of road,
                # we will use to determine if road is a Major road or Highway and on/off ramp
                road_name = row['properties']['name']
                speed = row['properties']['speed']
                # print('Name: ', road_name, 'speed', speed)

                # use Google API to get speed limits on desired roads and save them to output
                if road_name:  # if its a Major road
                    pass
                elif speed >= 100:  # highway or on/off ramp
                    pass

                # row_copy['properties']['Id'] = str(match[DETECTOR_ID_NAME])
                # output.write(row_copy)
    pass

project_delimitation = []
project_delimitation.append((-121.94277062699996, 37.55273259000006))
project_delimitation.append((-121.94099807399999, 37.554268507000074))
project_delimitation.append((-121.91790942699998, 37.549823434000075))
project_delimitation.append((-121.89348666299998, 37.52770136500004, ))
project_delimitation.append((-121.90056572499998, 37.52292299800007))
project_delimitation.append((-121.90817571699995, 37.52416183400004))
project_delimitation.append((-121.91252749099999, 37.51845069500007))
project_delimitation.append((-121.91349347899995, 37.513972023000065))
project_delimitation.append((-121.90855417099999, 37.503837324000074))
project_delimitation.append((-121.91358547299996, 37.50097863000008))
project_delimitation.append((-121.90798018999999, 37.49080413200005))
project_delimitation.append((-121.91894942199997, 37.48791568200005))
project_delimitation.append((-121.92029048799998, 37.488706567000065))
project_delimitation.append((-121.93070953799997, 37.48509600500006))
project_delimitation.append((-121.93254686299997, 37.48864173700008))
project_delimitation.append((-121.94079404499996, 37.50416395900004))
project_delimitation.append((-121.94569804899999, 37.51332606200003))
project_delimitation.append((-121.94918207899997, 37.520371545000046))
project_delimitation.append((-121.95305006999996, 37.52804520800004))
project_delimitation.append((-121.953966735, 37.53272020000003))
project_delimitation.append((-121.95428756799998, 37.53817435800005))
project_delimitation.append((-121.95506236799997, 37.54107322100003))
project_delimitation.append((-121.95676186899999, 37.54656695700004))
project_delimitation.append((-121.95529950799994, 37.54980786700003))
project_delimitation.append((-121.95261192399994, 37.550479763000055))
project_delimitation.append((-121.94988481799999, 37.55277211300006))
project_delimitation.append((-121.94613010599994, 37.55466923100005))
project_delimitation.append((-121.94277062699996, 37.55273259000006))


def create_external_taz(dir_taz, sections_df):
    """
    3 Steps for Create external TAZs
    1. Create a external demand delimitation:
    - load SFCTA data as Geopandas point (one point = one origin or one destination)
    - Get convex hull of the point
    - Use the convex hull (+ buffer) as the external demand delimitation
    2. create external centroids:
    - select road with no fnode and capacity above 800 from sections_df
    - create a point at the end of all selected road
    - plot the points, get a list of points to remove visually
    3. create external TAZs:
    - create a mesh a points inside the external demand delimitation and outside the internal demand delimitation (project delimitation)
    - use a Direction API (maybe Here direction):
    for every mesh point:
        Query path from mesh point to center of the project area
        Find the closest external centroid to the path. Test that all paths are not to far from existing
            external centroid --> if not, we might be missing one external centroid.
        Associate the external centroid to the mesh point.
        create external TAZ from mesh of points (if you reach point, Theo has already done it for internal TAZs)

    @param dir_taz:         folder containing prefix_fremont_legs.csv where prefix=ending, internal and starting
    @param sections_df:     geo pandas data frame of the aimsun sections
    """
    # 1. Create a external demand delimitation:
    # load the 3 csv files
    ending_csv = pd.read_csv(os.path.join(dir_taz, "ending_fremont_legs.csv"))
    internal_csv = pd.read_csv(os.path.join(dir_taz, "internal_fremont_legs.csv"))
    starting_csv = pd.read_csv(os.path.join(dir_taz, "starting_fremont_legs.csv"))

    # get the points from the csv's (start and end points)
    def get_points(csv_df):
        all_points = []
        node_types = ['start', 'end']
        for node_type in node_types:
            points = list(zip(csv_df[node_type + '_node_lng'], csv_df[node_type + '_node_lat']))
            all_points.extend(points)
        return all_points

    points = []
    points.extend(get_points(ending_csv))
    points.extend(get_points(internal_csv))
    points.extend(get_points(starting_csv))
    points = np.array(points)

    # get convex hull of points
    hull = ConvexHull(points)
    hull_points = points[hull.vertices, :]

    # add buffer to convex hull
    def normalize(point):
        norm = np.linalg.norm(point)
        return point / norm if norm > 0 else point

    # for each point calculate the direction to expand for buffer
    buffer_directions = []
    for i in range(len(hull_points)):
        point = hull_points[i]
        left_neighbor = hull_points[(i-1) % len(hull_points)]
        right_neighbor = hull_points[(i+1) % len(hull_points)]
        left_arrow = point - left_neighbor
        right_arrow = point - right_neighbor
        left_arrow = normalize(left_arrow)
        right_arrow = normalize(right_arrow)
        buffer_directions.append(normalize(left_arrow + right_arrow))
    buffer_directions = np.array(buffer_directions)

    # calculate the new (expanded) hull points with buffer
    buffer_coefficient = .05
    expanded_hull_points = hull_points + buffer_coefficient * buffer_directions

    ## to visualize the points and convex hulls
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.plot(hull_points[:, 0], hull_points[:, 1], 'bo--', lw=2)
    # plt.plot(expanded_hull_points[:, 0], expanded_hull_points[:, 1], 'ro--', lw=2)
    # plt.show()

    # 2. create external centroids:
    # select roads with no fnode and capacity above 800 from sections_df
    sections_df = sections_df[pd.isnull(sections_df['fnode']) & (sections_df['capacity'] > 800)]
    sections_df = sections_df[['eid', 'geometry']]

    # filter out roads that are visually erroneous -> a road not entering the project area (Fremont)
    # sections_df.to_csv('selected_roads.csv')   # roads to obtained visually
    roads_to_remove = [56744, 30676, 35572, 56534]
    sections_df = sections_df.astype({'eid': 'int32'})
    sections_df = sections_df[~sections_df['eid'].isin(roads_to_remove)]

    # create external centroid nodes -> create a point at the terminal end of these roads
    # that is, for each road find the end of the road that is closer to the external delimitation (convex hull)
    external_centroid_nodes = []
    internal_centroid_nodes = []  # need later to compute center point of project area
    circle = np.concatenate((expanded_hull_points, expanded_hull_points[0][None, :]), axis=0)
    external_delimitation = LineString(circle)
    for road in sections_df['geometry']:
        start_point = Point(road.coords[0])
        end_point = Point(road.coords[-1])

        if external_delimitation.distance(start_point) < external_delimitation.distance(end_point):
            # start is external centroid
            external_centroid_nodes.append(start_point)
            internal_centroid_nodes.append(end_point)
        else:
            # end is external centroid
            external_centroid_nodes.append(end_point)
            internal_centroid_nodes.append(start_point)

    # 3. create external TAZs:
    # create mesh of points
    mesh_density = 0.001  # should be 0.001 (creates 2 million points)
    x_min, x_max = np.min(expanded_hull_points[:, 0]), np.max(expanded_hull_points[:, 0])
    y_min, y_max = np.min(expanded_hull_points[:, 1]), np.max(expanded_hull_points[:, 1])
    x, y = np.meshgrid(np.arange(x_min, x_max, mesh_density), np.arange(y_min, y_max, mesh_density))
    x = x.reshape(x.shape[0] * x.shape[1])
    y = y.reshape(y.shape[0] * y.shape[1])
    mesh_points = list(zip(x, y))
    print('created {} mesh points'.format(len(mesh_points)))
    x = y = points = None  # free up memory

    # keep those inside external delimitation and outside project delimitation
    external_delimitation_poly = Polygon(expanded_hull_points)
    project_delimitation_poly = Polygon(project_delimitation)
    external_minus_project = external_delimitation_poly.difference(project_delimitation_poly)
    # bottleneck (iterating over points and using contains method is slow)
    mesh_points = list(filter(lambda p: external_minus_project.contains(p), MultiPoint(mesh_points)))
    print('kept {} mesh points'.format(len(mesh_points)))

    # compute center of project area
    internal_centroid_nodes = np.array([(p.x, p.y) for p in internal_centroid_nodes])
    project_center = np.mean(internal_centroid_nodes, axis=0)
    project_center = Point(project_center[0], project_center[1])

    # for each mesh point find closest external centroid to its query path
    project_delimitation_line = LineString(project_delimitation + [project_delimitation[0]])

    testing = True
    sample_size = 500
    info_point_to_center = []  # desired result
    intersection_to_centroid_paths = []

    # for testing sample mesh points at random and run them
    if testing:
        mesh_points = random.sample(mesh_points, sample_size)

    for point in mesh_points:
        path = get_path_by_here_api(point, project_center, stop_on_error=False)
        if not path:
            # not path found, ie. start is body of water hence no car path to destination
            # from sample testing, Google API takes this into account but not Here API
            continue  # next mesh point

        # find intersection (point) of path and project delimitation
        path = LineString(path)
        intersect_point = project_delimitation_line.intersection(path)
        if not isinstance(intersect_point, Point):
            # usually API error, ie. start is body of water, path includes a segment that jumps from water
            # to fremont intersecting project delimitation multiple times
            continue  # next mesh point

        # find closest centroid to intersection point
        min_distance = 999999
        closest_centroid = None
        for centroid in external_centroid_nodes:
            dist = intersect_point.distance(centroid)
            if dist < min_distance:
                min_distance = dist
                closest_centroid = centroid

        # path intersection to centroid
        intersection_to_centroid = [(intersect_point.x, intersect_point.y), (closest_centroid.x, closest_centroid.y)]
        intersection_to_centroid_paths.append(LineString(intersection_to_centroid))

        # write result to csv
        info_point_to_center.append([point, project_center, closest_centroid, min_distance, path])

    if testing:
        # can start a quick kepler demo on their website using these csv files
        pd.DataFrame({'geometry': mesh_points}).to_csv('test/mesh_points.csv')
        pd.DataFrame({'geometry': [l[-1] for l in info_point_to_center]}).to_csv('test/paths.csv')
        pd.DataFrame({'geometry': intersection_to_centroid_paths}).to_csv('test/intersect_to_centroids.csv')

    def to_csv(file_name, header, lines):
        def add_quotes(val):
            return "\"" + str(val) + "\"" if ',' in str(val) else str(val)

        csv = open(file_name, 'w')
        csv.write(header + '\n')
        for line in lines:
            csv.write(','.join(map(add_quotes, line)) + '\n')

    # write results to csv
    to_csv('test/mesh_point_to_centroid.csv',
           'origin_mesh_point,destination,closest_external_centroid,distance_to_centroid,path',
           info_point_to_center)


def get_path_by_here_api(start, end, stop_on_error=False):
    """
    Using Here API to get smooth path from start to end location
    where start and end are Point objects
    """
    here_url = 'https://route.ls.hereapi.com/routing/7.2/calculateroute.json?'
    api_key = 'QxZ6oy17gjoSHXxI9smn8THEzA1KA5pe7cveWt_k4hM'

    # convention on .shp files and kepler are lng, lat
    # Here API convention is lat, lng
    start_pos = 'geo!{},{}'.format(start.y, start.x)
    end_pos = 'geo!{},{}'.format(end.y, end.x)
    params = {'apiKey': api_key,
              'mode': 'fastest;car;traffic:disabled',
              'representation': 'display',
              'waypoint0': start_pos,
              'waypoint1': end_pos}

    response = requests.get(here_url, params=params)
    if response.ok:
        body = response.json()
        route = body['response']['route']
        if route:
            points = []
            points.append((start.x, start.y))
            maneuver = route[0]['leg'][0]['maneuver']
            for m in maneuver:
                path = m['shape']
                for p in path:
                    lat_lng = p.split(',')
                    lat = float(lat_lng[0])
                    lng = float(lat_lng[1])
                    # keep .shp file convention, lng, lat
                    points.append((lng, lat))
            points.append((end.x, end.y))
            return points
        else:
            if stop_on_error:
                stop_code('no routes found for start={}, destination={}'.format(start, end))
            return None
    else:
        if stop_on_error:
            response.raise_for_status()
        # Here API throws an error on some bodies of water, keep going if stop on error is false
        return None
        # print('response={}'.format(response))
        # print('start={}, destination={}'.format(start_pos, end_pos))

# legacy code method, used to get path from start to end via Google API
def get_path_by_google_api(start_point, end_point, stop_on_error=False):
    api_key = None  # removing api key from pushing to github
    google_url = "https://maps.googleapis.com/maps/api/directions/json"

    # convention on shp files for a geo-point is lng,lat
    # google api takes lat,lng
    origin = "{},{}".format(start_point.y, start_point.x)
    destination = "{},{}".format(end_point.y, end_point.x)
    params = {
        'origin': origin,
        'destination': destination,
        'key': api_key
    }
    response = requests.get(google_url, params=params)
    if response.ok:
        body = response.json()
        # print(body)
        routes = body['routes']
        if routes:
            # reminder, don't use legs, is not a smooth path from origin to destination
            # legs = routes[0]['legs'][0]['steps']
            encoded_points = routes[0]['overview_polyline']['points']
            points = polyline.decode(encoded_points)
            # keep lng,lat convention
            for i in range(len(points)):
                points[i] = points[i][::-1]
            return points
        else:
            if stop_on_error:
                stop_code('no routes found for start={}, destination={}'.format(origin, destination))
            return None  # no routes found
    else:
        # request produced an error, we should stop code completely, likely that other requests will fail
        print('response={}'.format(response))
        print('start={}, destination={}'.format(origin, destination))
        response.raise_for_status()



def test():
    json = {"response":{"metaInfo":{"timestamp":"2020-05-12T22:00:15Z","mapVersion":"8.30.108.153","moduleVersion":"7.2.202018-7021","interfaceVersion":"2.6.76","availableMapVersion":["8.30.108.153"]},"route":[{"waypoint":[{"linkId":"-23818577","mappedPosition":{"latitude":38.3334201,"longitude":-122.6585667},"originalPosition":{"latitude":38.333979,"longitude":-122.6585699},"type":"stopOver","spot":0.7261905,"sideOfStreet":"right","mappedRoadName":"Curtis Dr","label":"Curtis Dr","shapeIndex":0,"source":"user"},{"linkId":"+23742171","mappedPosition":{"latitude":37.5156533,"longitude":-121.9345286},"originalPosition":{"latitude":37.5152057,"longitude":-121.9349225},"type":"stopOver","spot":0.9008264,"sideOfStreet":"left","mappedRoadName":"Laurel Canyon Way","label":"Laurel Canyon Way","shapeIndex":897,"source":"user"}],"mode":{"type":"fastest","transportModes":["car"],"trafficMode":"disabled","feature":[]},"shape":["38.3334201,-122.6585667","38.3334017,-122.6636946","38.3333695,-122.6667845","38.332243,-122.6667845","38.3265781,-122.6667094","38.3228016,-122.666688","38.3223081,-122.6667202","38.3216,-122.6667953","38.3203125,-122.6668167","38.3193147,-122.6667845","38.314755,-122.6667202","38.3099914,-122.6665807","38.3070624,-122.6665163","38.3031678,-122.6664948","38.3008289,-122.6664627","38.2983398,-122.6664519","38.2960224,-122.6664627","38.2958829,-122.6664734","38.2957435,-122.6665056","38.2956469,-122.6665699","38.2955503,-122.666688","38.2954645,-122.6668489","38.2952929,-122.6672995","38.2945526,-122.6669884","38.2937586,-122.666688","38.2931793,-122.6664948","38.2929754,-122.6664627","38.292675,-122.6664305","38.2923317,-122.6664305","38.2894135,-122.6664948","38.288759,-122.6665485","38.2875252,-122.6667309","38.2836092,-122.667557","38.2832766,-122.6676428","38.2805407,-122.6682329","38.2789958,-122.6685441","38.2779551,-122.6687264","38.2756162,-122.6693058","38.2741892,-122.6695311","38.2722795,-122.6699817","38.2714963,-122.6701319","38.2710564,-122.6703036","38.2709813,-122.6703572","38.2709277,-122.6704645","38.2709169,-122.6705611","38.2710028,-122.6707435","38.2710886,-122.6708615","38.2712817,-122.671001","38.271507,-122.6710653","38.2718182,-122.6710224","38.2719147,-122.6709688","38.2719791,-122.6709044","38.2720327,-122.6708078","38.2720757,-122.6707006","38.2720864,-122.6705718","38.2720757,-122.6704645","38.2720542,-122.670368","38.2720113,-122.6702714","38.2719362,-122.6701641","38.2714963,-122.6693809","38.2701552,-122.6676321","38.2693505,-122.6666558","38.2684386,-122.6656151","38.2668507,-122.663877","38.2661533,-122.6630831","38.2657456,-122.6625574","38.2653809,-122.6620209","38.2650268,-122.661463","38.2627416,-122.6573753","38.2569051,-122.6470327","38.2549095,-122.64346","38.250103,-122.6341903","38.2464552,-122.6271093","38.2457578,-122.6258862","38.2455754,-122.6255965","38.2451785,-122.6250494","38.2447815,-122.624588","38.2440841,-122.6238799","38.2435691,-122.62344","38.2432902,-122.6232255","38.242861,-122.6229358","38.2412517,-122.6220882","38.2404149,-122.6216698","38.2381296,-122.6204789","38.2346213,-122.6187301","38.2339025,-122.6183331","38.2335591,-122.6181614","38.2333231,-122.6180649","38.232851,-122.6177859","38.2327974,-122.6177752","38.2315958,-122.6171637","38.2302547,-122.6164556","38.2289457,-122.6158011","38.2267249,-122.6146424","38.2261992,-122.6143527","38.2259309,-122.614181","38.2256949,-122.6140201","38.2251263,-122.6135695","38.2247937,-122.6132584","38.224225,-122.6126361","38.2236028,-122.6118207","38.2210708,-122.6080763","38.2203197,-122.6069927","38.2196116,-122.606113","38.2191932,-122.6056623","38.2188606,-122.6053298","38.2184422,-122.6049435","38.2178414,-122.60445","38.2174015,-122.6041174","38.2170153,-122.6038599","38.216393,-122.6034951","38.2158458,-122.6032162","38.2123911,-122.601639","38.2109427,-122.6009953","38.2092369,-122.6002014","38.2076061,-122.5994933","38.2071018,-122.5993001","38.2065976,-122.5991392","38.2056534,-122.5989032","38.2052886,-122.5988281","38.2040656,-122.5986564","38.2036042,-122.598635","38.202585,-122.598635","38.2020915,-122.5986671","38.2005572,-122.5988495","38.1983149,-122.5992465","38.1967163,-122.5996113","38.1950748,-122.6001048","38.1924891,-122.6008451","38.1914914,-122.6011562","38.1898284,-122.601639","38.1893349,-122.6017463","38.1884766,-122.6018965","38.1879187,-122.6019394","38.1871462,-122.6019609","38.1864274,-122.6019073","38.1859767,-122.6018536","38.1850541,-122.6016605","38.1846249,-122.6015425","38.1840563,-122.6013494","38.1836164,-122.601167","38.1828761,-122.6008236","38.1824148,-122.6005554","38.1817174,-122.6001048","38.1808376,-122.5994289","38.1805801,-122.5992036","38.1797647,-122.5983667","38.1793034,-122.597841","38.1787026,-122.5970685","38.1765783,-122.5942576","38.1744325,-122.5913286","38.1704307,-122.5859427","38.1700444,-122.5853741","38.1694329,-122.5843871","38.1683815,-122.5827563","38.1663644,-122.5795484","38.1659567,-122.5789905","38.1655383,-122.5784647","38.1640792,-122.5767159","38.1632853,-122.5757182","38.1602597,-122.5715339","38.1596589,-122.5708151","38.1587684,-122.5700104","38.1583071,-122.5696671","38.157835,-122.5693667","38.1573522,-122.5691092","38.1563544,-122.568723","38.1519449,-122.5672317","38.1466448,-122.5653541","38.1455183,-122.5649357","38.1446707,-122.5646996","38.144381,-122.5646353","38.1438017,-122.5645709","38.134886,-122.5639164","38.1343496,-122.563895","38.1333089,-122.5638843","38.1324291,-122.5638843","38.1318927,-122.5639057","38.1313348,-122.5639808","38.1307983,-122.5640774","38.1302726,-122.5642061","38.1297255,-122.5643778","38.1293285,-122.564528","38.1269252,-122.5656545","38.1263995,-122.5658476","38.1258738,-122.5660193","38.1253266,-122.5661266","38.1247795,-122.5662017","38.124243,-122.5662231","38.1236959,-122.5662017","38.1231487,-122.5661373","38.1177521,-122.5653005","38.1127417,-122.5644958","38.1116474,-122.5643349","38.1102204,-122.5640774","38.1088793,-122.5637448","38.1078601,-122.5634444","38.1049955,-122.562393","38.1038582,-122.5620174","38.1033218,-122.5618994","38.1029141,-122.5618351","38.1023669,-122.5617814","38.1020558,-122.5617599","38.1015944,-122.5617599","38.1009185,-122.5618029","38.0989337,-122.5619853","38.0984187,-122.5620174","38.0979252,-122.5620067","38.0974317,-122.5619745","38.0964446,-122.5618136","38.0954468,-122.5614703","38.0949426,-122.5612557","38.0944705,-122.5610304","38.0939555,-122.5607407","38.093065,-122.560097","38.0926037,-122.5597215","38.0918419,-122.5589597","38.0907047,-122.5577259","38.0900395,-122.5570714","38.0894494,-122.5564063","38.0885053,-122.555387","38.077594,-122.5438213","38.0768108,-122.5430274","38.0764997,-122.5427592","38.0761135,-122.5424588","38.0756199,-122.5421369","38.0721438,-122.5402486","38.0701697,-122.5392187","38.0694079,-122.5389183","38.0677986,-122.5384247","38.0673695,-122.5382638","38.0668759,-122.538017","38.0661893,-122.537545","38.0655885,-122.53703","38.0648911,-122.5363648","38.0640113,-122.5355709","38.0613077,-122.5331676","38.060385,-122.5323844","38.060149,-122.5322127","38.0599022,-122.5320411","38.0591619,-122.5316012","38.0586255,-122.5313544","38.058089,-122.5311399","38.0575526,-122.5309682","38.0569947,-122.5308502","38.0561686,-122.5307429","38.0553317,-122.5307322","38.0547738,-122.5307751","38.0531001,-122.5309896","38.051995,-122.5311506","38.0514586,-122.5312579","38.0503964,-122.5315368","38.0495381,-122.5318372","38.0486584,-122.5322127","38.0480576,-122.5325024","38.0476391,-122.5327277","38.0469847,-122.533114","38.046062,-122.5337148","38.0436051,-122.5354314","38.0419743,-122.5365472","38.0414915,-122.5368154","38.041234,-122.5369227","38.0407083,-122.5370622","38.0404401,-122.5371051","38.0401719,-122.5371265","38.0394959,-122.5370944","38.0377364,-122.5367188","38.0373716,-122.5366759","38.036921,-122.5366652","38.0366421,-122.5366867","38.0363739,-122.5367296","38.0345607,-122.5371051","38.0335522,-122.5372553","38.0326939,-122.5373518","38.0303442,-122.5375772","38.0275118,-122.5379097","38.0269647,-122.5379956","38.0226517,-122.5388861","38.0216861,-122.5391114","38.0212569,-122.5392401","38.0208814,-122.5393581","38.0198193,-122.5397658","38.0161607,-122.541225","38.0154634,-122.5415254","38.0147874,-122.5417829","38.0142832,-122.5419438","38.0137146,-122.5420725","38.0134571,-122.5421047","38.0128992,-122.5421262","38.0090904,-122.5421047","38.0086291,-122.5420618","38.0078781,-122.5419116","38.0073953,-122.5417614","38.0070841,-122.5416327","38.0064297,-122.5413108","38.0059147,-122.5409889","38.0051851,-122.5404632","38.0036831,-122.5393152","37.9965484,-122.5340796","37.9934049,-122.5317407","37.9926324,-122.5311935","37.9920852,-122.5307751","37.9916883,-122.5304103","37.9913557,-122.5301421","37.9895639,-122.5283289","37.9883516,-122.5271809","37.9855299,-122.5244451","37.9850793,-122.5240481","37.9845965,-122.5236726","37.9838991,-122.5232327","37.9810023,-122.5216448","37.9804981,-122.5214088","37.9800045,-122.5212049","37.9797041,-122.5210977","37.9791892,-122.5209475","37.9788888,-122.5208724","37.9783201,-122.5207651","37.9778051,-122.5207007","37.9772687,-122.5206792","37.9764748,-122.5207007","37.9758739,-122.5207651","37.9754019,-122.5208616","37.9739642,-122.5212371","37.9731274,-122.5215054","37.9723549,-122.5216985","37.9717648,-122.5218809","37.9713893,-122.5219452","37.9711211,-122.5219667","37.9708099,-122.5219667","37.9705417,-122.5219452","37.9702628,-122.5218916","37.9700053,-122.5218058","37.96978,-122.5217092","37.9692972,-122.5214517","37.9690504,-122.5212693","37.96875,-122.5210011","37.9684389,-122.5206578","37.9680419,-122.5200891","37.9670763,-122.5183725","37.9668081,-122.5178683","37.9648876,-122.5145316","37.9644048,-122.5138772","37.9632032,-122.511946","37.9629457,-122.5116026","37.9627848,-122.5112915","37.9623556,-122.5105405","37.9621518,-122.5100148","37.9616046,-122.5082874","37.9614329,-122.5079226","37.9613149,-122.5077188","37.9609931,-122.5071287","37.9607463,-122.5067639","37.9604244,-122.5062382","37.9599953,-122.5056159","37.9596841,-122.5052083","37.9593408,-122.504822","37.9574418,-122.5028157","37.9467773,-122.4916148","37.9461765,-122.491014","37.9456079,-122.4902201","37.9454577,-122.4899626","37.945329,-122.4896944","37.9451787,-122.4893403","37.9450393,-122.4889326","37.944932,-122.4885142","37.9448676,-122.4882245","37.9447818,-122.4875271","37.9447603,-122.4871624","37.9447603,-122.4867976","37.9448247,-122.4857354","37.9448676,-122.4843621","37.9448247,-122.4837291","37.9447174,-122.4830961","37.9445457,-122.482506","37.9443204,-122.4819374","37.9441917,-122.4816692","37.9434514,-122.4804568","37.9431832,-122.4799848","37.9429901,-122.4795878","37.9428613,-122.4792552","37.9427326,-122.4788582","37.942636,-122.4785149","37.9419816,-122.4748027","37.9417777,-122.4737084","37.9414558,-122.4721313","37.9410267,-122.4701464","37.9406297,-122.4684298","37.9401684,-122.4663162","37.939117,-122.4627435","37.9384089,-122.4604154","37.9367137,-122.4551153","37.9364669,-122.4543214","37.9361987,-122.4533451","37.9359627,-122.4522936","37.9357266,-122.4511135","37.9356086,-122.4503946","37.9354799,-122.4493325","37.9353189,-122.4477661","37.9351366,-122.4456525","37.9348898,-122.442348","37.9346538,-122.4386251","37.9342997,-122.4325311","37.9327548,-122.4093246","37.9325938,-122.4077904","37.9324651,-122.4069428","37.9322934,-122.405473","37.9320681,-122.4037993","37.9319286,-122.4024367","37.9318964,-122.4017394","37.9319286,-122.399776","37.9319608,-122.3994648","37.9320788,-122.3988318","37.9322398,-122.3982739","37.9324436,-122.39748","37.9325402,-122.3968363","37.9325724,-122.3963213","37.9325724,-122.3959351","37.9325509,-122.395581","37.932508,-122.3952484","37.9324543,-122.3949051","37.9322827,-122.3941863","37.9321003,-122.3937249","37.931757,-122.3929632","37.9315102,-122.3923516","37.9302979,-122.3897123","37.9297614,-122.3884571","37.9294825,-122.3876953","37.9290855,-122.3864722","37.9285705,-122.3847449","37.9277658,-122.3821914","37.9274976,-122.3812258","37.9273474,-122.3805714","37.9271972,-122.379638","37.9271114,-122.3787153","37.9270899,-122.378211","37.9270685,-122.3744237","37.9270685,-122.369585","37.927047,-122.3689413","37.9269934,-122.3682868","37.9268968,-122.3675573","37.9267573,-122.3668063","37.9265964,-122.3661411","37.9264033,-122.3654759","37.9262853,-122.365154","37.9260385,-122.364521","37.9257596,-122.3638988","37.9252446,-122.3629546","37.9234529,-122.3600578","37.9230344,-122.3592854","37.9227984,-122.3587918","37.9225731,-122.3582339","37.9223049,-122.3575044","37.9219937,-122.3563349","37.9218972,-122.3558736","37.9218006,-122.3552835","37.9217148,-122.3546076","37.9216719,-122.3539746","37.9216504,-122.3533523","37.921747,-122.3475051","37.9217362,-122.3466575","37.9216182,-122.3430848","37.9215109,-122.3407459","37.9214036,-122.3378921","37.9212749,-122.3368084","37.9212213,-122.3364437","37.921114,-122.3358965","37.920953,-122.3352635","37.9207599,-122.3346305","37.9206634,-122.3343408","37.9201591,-122.3331928","37.9196227,-122.3322487","37.9189897,-122.3313153","37.9183245,-122.3303711","37.9172945,-122.3290408","37.9163504,-122.3279464","37.9160821,-122.3276567","37.9151487,-122.3266912","37.9143119,-122.3259079","37.9128206,-122.3246419","37.9121447,-122.3240948","37.9120588,-122.3240089","37.9091406,-122.321595","37.9087651,-122.3213053","37.902832,-122.3163915","37.899431,-122.3136234","37.8988838,-122.3131406","37.8953648,-122.3103297","37.8950536,-122.3101044","37.8947103,-122.3099005","37.8938735,-122.3094499","37.8929615,-122.3091066","37.8925216,-122.3089778","37.8922212,-122.3089135","37.8919208,-122.3088706","37.8912342,-122.3088491","37.8907728,-122.308892","37.8892064,-122.309171","37.8885734,-122.3092568","37.8878546,-122.3092782","37.887125,-122.3093426","37.8865242,-122.3092997","37.8836811,-122.3086774","37.8827047,-122.3084414","37.8817821,-122.308259","37.8801513,-122.307905","37.8793895,-122.3077226","37.8776622,-122.3072827","37.8765249,-122.3069072","37.8734028,-122.3059738","37.8710961,-122.3052549","37.8667188,-122.3038495","37.8642726,-122.3031843","37.8633928,-122.302916","37.862159,-122.3025835","37.854445,-122.3003948","37.8510332,-122.2994614","37.8488767,-122.2989786","37.8477716,-122.2987533","37.8445745,-122.298013","37.8426754,-122.2976053","37.8422248,-122.2975302","37.8399181,-122.2970688","37.8385556,-122.2968328","37.838062,-122.2967148","37.8379655,-122.2967041","37.8370428,-122.2964144","37.8347147,-122.2953093","37.8328478,-122.2944617","37.832644,-122.2943223","37.8324831,-122.2942364","37.8321826,-122.2941291","37.8316677,-122.2939682","37.8312814,-122.2938716","37.830863,-122.2937965","37.8305089,-122.2937858","37.8301013,-122.293818","37.829597,-122.2938824","37.8294146,-122.2938824","37.829082,-122.2938502","37.8288352,-122.293818","37.8284705,-122.2936678","37.8282773,-122.2935712","37.8279448,-122.2933245","37.8275907,-122.2929704","37.8270543,-122.2922409","37.8268182,-122.2919834","37.8264749,-122.2916615","37.8261209,-122.2914469","37.8258848,-122.2913289","37.8257132,-122.2912753","37.8251874,-122.2911787","37.8247154,-122.2912002","37.8244579,-122.2912538","37.8242862,-122.2913074","37.8240502,-122.2914147","37.8234494,-122.2917366","37.8228915,-122.2920585","37.8219473,-122.2926486","37.8214753,-122.2929811","37.8201663,-122.2941077","37.8196406,-122.2946548","37.8191471,-122.2952449","37.8135145,-122.3023367","37.8131497,-122.3027766","37.8127956,-122.3031521","37.8123558,-122.3035491","37.8119051,-122.3039138","37.8114116,-122.304225","37.8106713,-122.304579","37.810446,-122.3046649","37.8101563,-122.3047507","37.8093946,-122.3048794","37.808826,-122.3049331","37.8079891,-122.3048794","37.8075171,-122.3048151","37.8073883,-122.3047936","37.8069592,-122.3046649","37.8064227,-122.3044825","37.8059292,-122.3042357","37.8056073,-122.3040533","37.8053391,-122.3038709","37.8050816,-122.3036778","37.8044379,-122.3031306","37.8041482,-122.3028302","37.8037405,-122.3023367","37.8034079,-122.3018539","37.803129,-122.3014247","37.8026354,-122.3004162","37.8023887,-122.2998047","37.802099,-122.2987854","37.8020346,-122.298485","37.8018951,-122.2976053","37.8018522,-122.2970152","37.8018415,-122.2961247","37.8019381,-122.2950304","37.8020239,-122.2943652","37.802453,-122.2918975","37.8026676,-122.2904921","37.8027534,-122.2896659","37.8027749,-122.2885716","37.8026569,-122.2870159","37.8025711,-122.2863936","37.8024209,-122.2856104","37.8022385,-122.2849131","37.8019273,-122.2839689","37.8016591,-122.2832394","37.8011763,-122.2820163","37.8003824,-122.2798598","37.8002751,-122.2796452","37.8001034,-122.2792161","37.8000176,-122.2790444","37.7990305,-122.2765768","37.7978933,-122.2735083","37.7974856,-122.272532","37.7969706,-122.2711265","37.7962303,-122.2692382","37.7958977,-122.2684336","37.7956295,-122.2677255","37.7949321,-122.2661054","37.7944922,-122.2653115","37.7940202,-122.2646248","37.7934194,-122.2638953","37.7930868,-122.263552","37.7923036,-122.2626078","37.7919173,-122.2621894","37.7913809,-122.2615457","37.7910697,-122.2611165","37.7907264,-122.2605908","37.7904153,-122.2599685","37.7902973,-122.2596788","37.7901149,-122.2591853","37.7898896,-122.2583163","37.7896857,-122.2572219","37.7895033,-122.2559667","37.7891386,-122.2538638","37.7889025,-122.2527695","37.7886236,-122.2517073","37.7882159,-122.2503984","37.7879584,-122.2496367","37.7872396,-122.2476411","37.7869713,-122.2469866","37.7867568,-122.2464931","37.7858019,-122.2446048","37.7848256,-122.2428453","37.7843106,-122.2418582","37.7837849,-122.2409356","37.7834201,-122.2404206","37.7828836,-122.2397768","37.7824116,-122.2393692","37.7818537,-122.2390473","37.7815318,-122.2388971","37.7812529,-122.2387898","37.7810061,-122.2387147","37.7797723,-122.2382748","37.7787745,-122.2378671","37.7784204,-122.2376955","37.7779055,-122.237395","37.777648,-122.2372127","37.7773583,-122.2369552","37.7768648,-122.2364616","37.7762425,-122.2357857","37.7758884,-122.2353029","37.77547,-122.2345841","37.7751803,-122.2340155","37.7749658,-122.2335112","37.7740753,-122.2308612","37.7739251,-122.2304535","37.7733672,-122.2287154","37.7731526,-122.2279966","37.7720475,-122.2247458","37.7717364,-122.2240913","37.7713716,-122.2234476","37.7710819,-122.2230184","37.7707171,-122.2225463","37.7705348,-122.2223747","37.7703846,-122.2221923","37.7702129,-122.2220314","37.769419,-122.2213769","37.7689469,-122.221055","37.7681315,-122.2204328","37.7669513,-122.2195852","37.7657068,-122.2186089","37.764945,-122.2179115","37.7644086,-122.2173858","37.7639365,-122.2169888","37.7607822,-122.21421","37.7589476,-122.2126329","37.7576709,-122.2114956","37.7564263,-122.2104228","37.7540123,-122.2082663","37.7532184,-122.2075903","37.7496135,-122.2044039","37.7439809,-122.1994901","37.742275,-122.1979666","37.7408481,-122.196722","37.7403975,-122.1963143","37.7403331,-122.1962392","37.7382946,-122.1945441","37.7382731,-122.1945119","37.7366102,-122.193085","37.7348614,-122.1915507","37.7322865,-122.1892548","37.7298295,-122.187109","37.7281451,-122.1855855","37.7273941,-122.1849632","37.7253127,-122.1831393","37.7250659,-122.182914","37.7249479,-122.1827853","37.7247655,-122.1826351","37.7241111,-122.1819913","37.71909,-122.1768951","37.7173197,-122.1750712","37.7149165,-122.172668","37.7138972,-122.1715951","37.7130497,-122.1705329","37.7124059,-122.1696532","37.710464,-122.1667778","37.708447,-122.1638596","37.707063,-122.1617675","37.7039731,-122.1573687","37.6997244,-122.1511996","37.6982224,-122.149086","37.6944566,-122.1435821","37.6926863,-122.1410286","37.691313,-122.1389687","37.6882768,-122.1346986","37.6874828,-122.1334648","37.6854014,-122.1304929","37.6848435,-122.1296668","37.6844466,-122.1291196","37.6826978,-122.1265876","37.6806486,-122.1237016","37.6794684,-122.1220064","37.6790714,-122.1213841","37.6779985,-122.1198392","37.6765501,-122.1175647","37.676003,-122.1167707","37.6748228,-122.1152365","37.6742113,-122.1143889","37.6736641,-122.1135843","37.6732242,-122.1129942","37.6727092,-122.1123612","37.6723659,-122.1119857","37.6720119,-122.1116209","37.6715183,-122.1111488","37.6711214,-122.110827","37.6705205,-122.1104085","37.6689649,-122.1094","37.6647913,-122.1068251","37.6642656,-122.1065247","37.6633644,-122.1059775","37.660135,-122.1039176","37.6543951,-122.1003556","37.6513052,-122.0984674","37.6487625,-122.0968366","37.6480114,-122.0962465","37.6448357,-122.0940149","37.6444817,-122.0937252","37.6432049,-122.0928562","37.6410377,-122.0913112","37.6387739,-122.0896697","37.6317143,-122.0846164","37.6239789,-122.0791233","37.6177025,-122.0746279","37.617327,-122.0743811","37.6164579,-122.0737267","37.6129603,-122.0712268","37.6086688,-122.068212","37.6086044,-122.0681477","37.6071453,-122.0670855","37.6036048,-122.064575","37.6002145,-122.0621395","37.5989377,-122.061249","37.5987554,-122.0611417","37.5967705,-122.0596933","37.5964165,-122.0594144","37.5952148,-122.0585883","37.5925541,-122.0566785","37.5900221,-122.054919","37.5852048,-122.0514858","37.5841856,-122.0508099","37.5836813,-122.050488","37.5821364,-122.0496619","37.5812995,-122.0493293","37.5783062,-122.0482564","37.572459,-122.0462608","37.5719011,-122.0460463","37.5714827,-122.0458531","37.5710535,-122.0456064","37.5705814,-122.0453918","37.5695729,-122.0447266","37.5692403,-122.0444798","37.5685751,-122.0439327","37.5680709,-122.0434928","37.5676847,-122.0431066","37.5638652,-122.0390081","37.5633931,-122.0384824","37.5618589,-122.0368731","37.5608397,-122.0357788","37.5602067,-122.035135","37.5556147,-122.0301569","37.5544238,-122.0289445","37.5540912,-122.0286441","37.5533187,-122.0279896","37.5511837,-122.026391","37.5496495,-122.0252645","37.548008,-122.0239878","37.5465596,-122.0229578","37.5438452,-122.0208871","37.5433731,-122.0205438","37.5429869,-122.0202863","37.5423646,-122.0198035","37.539264,-122.0174754","37.5384593,-122.0167994","37.537955,-122.0163167","37.5375152,-122.0158875","37.5369358,-122.0152223","37.536217,-122.0141923","37.5355947,-122.0131409","37.532419,-122.0073152","37.5296938,-122.0022082","37.5291896,-122.0013607","37.5278699,-121.998893","37.5257349,-121.9944942","37.5242007,-121.9914043","37.5232136,-121.9893551","37.5211,-121.9851279","37.5205421,-121.9841516","37.5198662,-121.9831431","37.519598,-121.9827998","37.5182676,-121.9811261","37.5137186,-121.9755042","37.5102317,-121.971159","37.5098991,-121.9709229","37.5096416,-121.9706762","37.5088155,-121.9698071","37.5084507,-121.9694746","37.5083542,-121.9694209","37.5082254,-121.9693565","37.5080216,-121.9693244","37.5074959,-121.9693458","37.5073242,-121.969378","37.5071526,-121.9693887","37.5068414,-121.969378","37.5067127,-121.9692814","37.5066161,-121.9690669","37.5068629,-121.9685197","37.5085688,-121.9643676","37.5093949,-121.962415","37.5094914,-121.962136","37.5096738,-121.9614923","37.5097167,-121.9611919","37.5102639,-121.95822","37.5103176,-121.957866","37.5110471,-121.9540358","37.5112832,-121.9533706","37.5117123,-121.9523728","37.5117981,-121.9520295","37.5120234,-121.9509351","37.51212,-121.9503343","37.5121951,-121.949991","37.5122917,-121.9491971","37.5122273,-121.9476843","37.5122702,-121.9471586","37.5123775,-121.9463754","37.5124633,-121.9460106","37.5126779,-121.9452918","37.5128067,-121.944927","37.5129569,-121.94453","37.5132895,-121.9437468","37.5134611,-121.9434464","37.5136757,-121.94291","37.5140941,-121.9414937","37.5142658,-121.9408178","37.5148129,-121.938951","37.5149846,-121.9382751","37.5149846,-121.9377923","37.5149632,-121.9375026","37.5149202,-121.9372344","37.5147593,-121.9366872","37.5147378,-121.9365585","37.5146091,-121.9362473","37.5141156,-121.9351745","37.5139546,-121.9347453","37.5138795,-121.934402","37.5138581,-121.9340587","37.5138688,-121.9337583","37.5139439,-121.9333076","37.5140727,-121.9326854","37.514888,-121.9328785","37.5150275,-121.9329643","37.5151348,-121.9330609","37.5152636,-121.9332218","37.5153494,-121.9333613","37.5154245,-121.9336188","37.5155747,-121.9343376","37.5156069,-121.9344449","37.5156533,-121.9345286"],"leg":[{"maneuver":[{"position":{"latitude":38.3334201,"longitude":-122.6585667},"instruction":"Head toward <span class=\"toward_street\">Johnies Way</span> on <span class=\"street\">Curtis Dr</span>. <span class=\"distance-description\">Go for <span class=\"length\">718 m</span>.</span>","travelTime":127,"length":718,"shape":["38.3334201,-122.6585667","38.3334017,-122.6636946","38.3333695,-122.6667845"],"note":[],"id":"M1","_type":"PrivateTransportManeuverType"},{"position":{"latitude":38.3333695,"longitude":-122.6667845},"instruction":"Turn <span class=\"direction\">left</span> onto <span class=\"next-street\">Petaluma Hill Rd</span>. <span class=\"distance-description\">Go for <span class=\"length\">4.3 km</span>.</span>","travelTime":300,"length":4269,"shape":["38.3333695,-122.6667845","38.332243,-122.6667845","38.3265781,-122.6667094","38.3228016,-122.666688","38.3223081,-122.6667202","38.3216,-122.6667953","38.3203125,-122.6668167","38.3193147,-122.6667845","38.314755,-122.6667202","38.3099914,-122.6665807","38.3070624,-122.6665163","38.3031678,-122.6664948","38.3008289,-122.6664627","38.2983398,-122.6664519","38.2960224,-122.6664627","38.2958829,-122.6664734","38.2957435,-122.6665056","38.2956469,-122.6665699","38.2955503,-122.666688","38.2954645,-122.6668489","38.2952929,-122.6672995"],"note":[],"id":"M2","_type":"PrivateTransportManeuverType"},{"position":{"latitude":38.2952929,"longitude":-122.6672995},"instruction":"Turn <span class=\"direction\">left</span> onto <span class=\"next-street\">Old Redwood Hwy</span>. <span class=\"distance-description\">Go for <span class=\"length\">2.7 km</span>.</span>","travelTime":195,"length":2679,"shape":["38.2952929,-122.6672995","38.2945526,-122.6669884","38.2937586,-122.666688","38.2931793,-122.6664948","38.2929754,-122.6664627","38.292675,-122.6664305","38.2923317,-122.6664305","38.2894135,-122.6664948","38.288759,-122.6665485","38.2875252,-122.6667309","38.2836092,-122.667557","38.2832766,-122.6676428","38.2805407,-122.6682329","38.2789958,-122.6685441","38.2779551,-122.6687264","38.2756162,-122.6693058","38.2741892,-122.6695311","38.2722795,-122.6699817","38.2714963,-122.6701319"],"note":[],"id":"M3","_type":"PrivateTransportManeuverType"},{"position":{"latitude":38.2714963,"longitude":-122.6701319},"instruction":"Take ramp onto <span class=\"number\">US-101 S</span> <span class=\"next-street\">(Redwood Hwy)</span> toward <span class=\"sign\"><span lang=\"en\">San Francisco</span></span>. <span class=\"distance-description\">Go for <span class=\"length\">40.2 km</span>.</span>","travelTime":1510,"length":40205,"shape":["38.2714963,-122.6701319","38.2710564,-122.6703036","38.2709813,-122.6703572","38.2709277,-122.6704645","38.2709169,-122.6705611","38.2710028,-122.6707435","38.2710886,-122.6708615","38.2712817,-122.671001","38.271507,-122.6710653","38.2718182,-122.6710224","38.2719147,-122.6709688","38.2719791,-122.6709044","38.2720327,-122.6708078","38.2720757,-122.6707006","38.2720864,-122.6705718","38.2720757,-122.6704645","38.2720542,-122.670368","38.2720113,-122.6702714","38.2719362,-122.6701641","38.2714963,-122.6693809","38.2701552,-122.6676321","38.2693505,-122.6666558","38.2684386,-122.6656151","38.2668507,-122.663877","38.2661533,-122.6630831","38.2657456,-122.6625574","38.2653809,-122.6620209","38.2650268,-122.661463","38.2627416,-122.6573753","38.2569051,-122.6470327","38.2549095,-122.64346","38.250103,-122.6341903","38.2464552,-122.6271093","38.2457578,-122.6258862","38.2455754,-122.6255965","38.2451785,-122.6250494","38.2447815,-122.624588","38.2440841,-122.6238799","38.2435691,-122.62344","38.2432902,-122.6232255","38.242861,-122.6229358","38.2412517,-122.6220882","38.2404149,-122.6216698","38.2381296,-122.6204789","38.2346213,-122.6187301","38.2339025,-122.6183331","38.2335591,-122.6181614","38.2333231,-122.6180649","38.232851,-122.6177859","38.2327974,-122.6177752","38.2315958,-122.6171637","38.2302547,-122.6164556","38.2289457,-122.6158011","38.2267249,-122.6146424","38.2261992,-122.6143527","38.2259309,-122.614181","38.2256949,-122.6140201","38.2251263,-122.6135695","38.2247937,-122.6132584","38.224225,-122.6126361","38.2236028,-122.6118207","38.2210708,-122.6080763","38.2203197,-122.6069927","38.2196116,-122.606113","38.2191932,-122.6056623","38.2188606,-122.6053298","38.2184422,-122.6049435","38.2178414,-122.60445","38.2174015,-122.6041174","38.2170153,-122.6038599","38.216393,-122.6034951","38.2158458,-122.6032162","38.2123911,-122.601639","38.2109427,-122.6009953","38.2092369,-122.6002014","38.2076061,-122.5994933","38.2071018,-122.5993001","38.2065976,-122.5991392","38.2056534,-122.5989032","38.2052886,-122.5988281","38.2040656,-122.5986564","38.2036042,-122.598635","38.202585,-122.598635","38.2020915,-122.5986671","38.2005572,-122.5988495","38.1983149,-122.5992465","38.1967163,-122.5996113","38.1950748,-122.6001048","38.1924891,-122.6008451","38.1914914,-122.6011562","38.1898284,-122.601639","38.1893349,-122.6017463","38.1884766,-122.6018965","38.1879187,-122.6019394","38.1871462,-122.6019609","38.1864274,-122.6019073","38.1859767,-122.6018536","38.1850541,-122.6016605","38.1846249,-122.6015425","38.1840563,-122.6013494","38.1836164,-122.601167","38.1828761,-122.6008236","38.1824148,-122.6005554","38.1817174,-122.6001048","38.1808376,-122.5994289","38.1805801,-122.5992036","38.1797647,-122.5983667","38.1793034,-122.597841","38.1787026,-122.5970685","38.1765783,-122.5942576","38.1744325,-122.5913286","38.1704307,-122.5859427","38.1700444,-122.5853741","38.1694329,-122.5843871","38.1683815,-122.5827563","38.1663644,-122.5795484","38.1659567,-122.5789905","38.1655383,-122.5784647","38.1640792,-122.5767159","38.1632853,-122.5757182","38.1602597,-122.5715339","38.1596589,-122.5708151","38.1587684,-122.5700104","38.1583071,-122.5696671","38.157835,-122.5693667","38.1573522,-122.5691092","38.1563544,-122.568723","38.1519449,-122.5672317","38.1466448,-122.5653541","38.1455183,-122.5649357","38.1446707,-122.5646996","38.144381,-122.5646353","38.1438017,-122.5645709","38.134886,-122.5639164","38.1343496,-122.563895","38.1333089,-122.5638843","38.1324291,-122.5638843","38.1318927,-122.5639057","38.1313348,-122.5639808","38.1307983,-122.5640774","38.1302726,-122.5642061","38.1297255,-122.5643778","38.1293285,-122.564528","38.1269252,-122.5656545","38.1263995,-122.5658476","38.1258738,-122.5660193","38.1253266,-122.5661266","38.1247795,-122.5662017","38.124243,-122.5662231","38.1236959,-122.5662017","38.1231487,-122.5661373","38.1177521,-122.5653005","38.1127417,-122.5644958","38.1116474,-122.5643349","38.1102204,-122.5640774","38.1088793,-122.5637448","38.1078601,-122.5634444","38.1049955,-122.562393","38.1038582,-122.5620174","38.1033218,-122.5618994","38.1029141,-122.5618351","38.1023669,-122.5617814","38.1020558,-122.5617599","38.1015944,-122.5617599","38.1009185,-122.5618029","38.0989337,-122.5619853","38.0984187,-122.5620174","38.0979252,-122.5620067","38.0974317,-122.5619745","38.0964446,-122.5618136","38.0954468,-122.5614703","38.0949426,-122.5612557","38.0944705,-122.5610304","38.0939555,-122.5607407","38.093065,-122.560097","38.0926037,-122.5597215","38.0918419,-122.5589597","38.0907047,-122.5577259","38.0900395,-122.5570714","38.0894494,-122.5564063","38.0885053,-122.555387","38.077594,-122.5438213","38.0768108,-122.5430274","38.0764997,-122.5427592","38.0761135,-122.5424588","38.0756199,-122.5421369","38.0721438,-122.5402486","38.0701697,-122.5392187","38.0694079,-122.5389183","38.0677986,-122.5384247","38.0673695,-122.5382638","38.0668759,-122.538017","38.0661893,-122.537545","38.0655885,-122.53703","38.0648911,-122.5363648","38.0640113,-122.5355709","38.0613077,-122.5331676","38.060385,-122.5323844","38.060149,-122.5322127","38.0599022,-122.5320411","38.0591619,-122.5316012","38.0586255,-122.5313544","38.058089,-122.5311399","38.0575526,-122.5309682","38.0569947,-122.5308502","38.0561686,-122.5307429","38.0553317,-122.5307322","38.0547738,-122.5307751","38.0531001,-122.5309896","38.051995,-122.5311506","38.0514586,-122.5312579","38.0503964,-122.5315368","38.0495381,-122.5318372","38.0486584,-122.5322127","38.0480576,-122.5325024","38.0476391,-122.5327277","38.0469847,-122.533114","38.046062,-122.5337148","38.0436051,-122.5354314","38.0419743,-122.5365472","38.0414915,-122.5368154","38.041234,-122.5369227","38.0407083,-122.5370622","38.0404401,-122.5371051","38.0401719,-122.5371265","38.0394959,-122.5370944","38.0377364,-122.5367188","38.0373716,-122.5366759","38.036921,-122.5366652","38.0366421,-122.5366867","38.0363739,-122.5367296","38.0345607,-122.5371051","38.0335522,-122.5372553","38.0326939,-122.5373518","38.0303442,-122.5375772","38.0275118,-122.5379097","38.0269647,-122.5379956","38.0226517,-122.5388861","38.0216861,-122.5391114","38.0212569,-122.5392401","38.0208814,-122.5393581","38.0198193,-122.5397658","38.0161607,-122.541225","38.0154634,-122.5415254","38.0147874,-122.5417829","38.0142832,-122.5419438","38.0137146,-122.5420725","38.0134571,-122.5421047","38.0128992,-122.5421262","38.0090904,-122.5421047","38.0086291,-122.5420618","38.0078781,-122.5419116","38.0073953,-122.5417614","38.0070841,-122.5416327","38.0064297,-122.5413108","38.0059147,-122.5409889","38.0051851,-122.5404632","38.0036831,-122.5393152","37.9965484,-122.5340796","37.9934049,-122.5317407","37.9926324,-122.5311935","37.9920852,-122.5307751","37.9916883,-122.5304103","37.9913557,-122.5301421","37.9895639,-122.5283289","37.9883516,-122.5271809","37.9855299,-122.5244451","37.9850793,-122.5240481","37.9845965,-122.5236726","37.9838991,-122.5232327","37.9810023,-122.5216448","37.9804981,-122.5214088","37.9800045,-122.5212049","37.9797041,-122.5210977","37.9791892,-122.5209475","37.9788888,-122.5208724","37.9783201,-122.5207651","37.9778051,-122.5207007","37.9772687,-122.5206792","37.9764748,-122.5207007","37.9758739,-122.5207651","37.9754019,-122.5208616","37.9739642,-122.5212371","37.9731274,-122.5215054","37.9723549,-122.5216985","37.9717648,-122.5218809","37.9713893,-122.5219452","37.9711211,-122.5219667","37.9708099,-122.5219667","37.9705417,-122.5219452","37.9702628,-122.5218916","37.9700053,-122.5218058","37.96978,-122.5217092","37.9692972,-122.5214517","37.9690504,-122.5212693","37.96875,-122.5210011","37.9684389,-122.5206578","37.9680419,-122.5200891","37.9670763,-122.5183725","37.9668081,-122.5178683","37.9648876,-122.5145316"],"note":[],"id":"M4","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.9648876,"longitude":-122.5145316},"instruction":"Take exit <span class=\"exit\">451B</span> toward <span class=\"sign\"><span lang=\"en\">I-580</span>/<span lang=\"en\">Richmond Br</span>/<span lang=\"en\">Oakland</span></span>. <span class=\"distance-description\">Go for <span class=\"length\">723 m</span>.</span>","travelTime":39,"length":723,"shape":["37.9648876,-122.5145316","37.9644048,-122.5138772","37.9632032,-122.511946","37.9629457,-122.5116026","37.9627848,-122.5112915","37.9623556,-122.5105405","37.9621518,-122.5100148","37.9616046,-122.5082874","37.9614329,-122.5079226","37.9613149,-122.5077188"],"note":[],"id":"M5","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.9613149,"longitude":-122.5077188},"instruction":"Keep <span class=\"direction\">left</span> onto <span class=\"number\">I-580 E</span> toward <span class=\"sign\"><span lang=\"en\">Richmond Bridge</span>/<span lang=\"en\">Oakland</span></span>. <span class=\"distance-description\">Go for <span class=\"length\">27.2 km</span>.</span>","travelTime":1146,"length":27232,"shape":["37.9613149,-122.5077188","37.9609931,-122.5071287","37.9607463,-122.5067639","37.9604244,-122.5062382","37.9599953,-122.5056159","37.9596841,-122.5052083","37.9593408,-122.504822","37.9574418,-122.5028157","37.9467773,-122.4916148","37.9461765,-122.491014","37.9456079,-122.4902201","37.9454577,-122.4899626","37.945329,-122.4896944","37.9451787,-122.4893403","37.9450393,-122.4889326","37.944932,-122.4885142","37.9448676,-122.4882245","37.9447818,-122.4875271","37.9447603,-122.4871624","37.9447603,-122.4867976","37.9448247,-122.4857354","37.9448676,-122.4843621","37.9448247,-122.4837291","37.9447174,-122.4830961","37.9445457,-122.482506","37.9443204,-122.4819374","37.9441917,-122.4816692","37.9434514,-122.4804568","37.9431832,-122.4799848","37.9429901,-122.4795878","37.9428613,-122.4792552","37.9427326,-122.4788582","37.942636,-122.4785149","37.9419816,-122.4748027","37.9417777,-122.4737084","37.9414558,-122.4721313","37.9410267,-122.4701464","37.9406297,-122.4684298","37.9401684,-122.4663162","37.939117,-122.4627435","37.9384089,-122.4604154","37.9367137,-122.4551153","37.9364669,-122.4543214","37.9361987,-122.4533451","37.9359627,-122.4522936","37.9357266,-122.4511135","37.9356086,-122.4503946","37.9354799,-122.4493325","37.9353189,-122.4477661","37.9351366,-122.4456525","37.9348898,-122.442348","37.9346538,-122.4386251","37.9342997,-122.4325311","37.9327548,-122.4093246","37.9325938,-122.4077904","37.9324651,-122.4069428","37.9322934,-122.405473","37.9320681,-122.4037993","37.9319286,-122.4024367","37.9318964,-122.4017394","37.9319286,-122.399776","37.9319608,-122.3994648","37.9320788,-122.3988318","37.9322398,-122.3982739","37.9324436,-122.39748","37.9325402,-122.3968363","37.9325724,-122.3963213","37.9325724,-122.3959351","37.9325509,-122.395581","37.932508,-122.3952484","37.9324543,-122.3949051","37.9322827,-122.3941863","37.9321003,-122.3937249","37.931757,-122.3929632","37.9315102,-122.3923516","37.9302979,-122.3897123","37.9297614,-122.3884571","37.9294825,-122.3876953","37.9290855,-122.3864722","37.9285705,-122.3847449","37.9277658,-122.3821914","37.9274976,-122.3812258","37.9273474,-122.3805714","37.9271972,-122.379638","37.9271114,-122.3787153","37.9270899,-122.378211","37.9270685,-122.3744237","37.9270685,-122.369585","37.927047,-122.3689413","37.9269934,-122.3682868","37.9268968,-122.3675573","37.9267573,-122.3668063","37.9265964,-122.3661411","37.9264033,-122.3654759","37.9262853,-122.365154","37.9260385,-122.364521","37.9257596,-122.3638988","37.9252446,-122.3629546","37.9234529,-122.3600578","37.9230344,-122.3592854","37.9227984,-122.3587918","37.9225731,-122.3582339","37.9223049,-122.3575044","37.9219937,-122.3563349","37.9218972,-122.3558736","37.9218006,-122.3552835","37.9217148,-122.3546076","37.9216719,-122.3539746","37.9216504,-122.3533523","37.921747,-122.3475051","37.9217362,-122.3466575","37.9216182,-122.3430848","37.9215109,-122.3407459","37.9214036,-122.3378921","37.9212749,-122.3368084","37.9212213,-122.3364437","37.921114,-122.3358965","37.920953,-122.3352635","37.9207599,-122.3346305","37.9206634,-122.3343408","37.9201591,-122.3331928","37.9196227,-122.3322487","37.9189897,-122.3313153","37.9183245,-122.3303711","37.9172945,-122.3290408","37.9163504,-122.3279464","37.9160821,-122.3276567","37.9151487,-122.3266912","37.9143119,-122.3259079","37.9128206,-122.3246419","37.9121447,-122.3240948","37.9120588,-122.3240089","37.9091406,-122.321595","37.9087651,-122.3213053","37.902832,-122.3163915","37.899431,-122.3136234","37.8988838,-122.3131406","37.8953648,-122.3103297","37.8950536,-122.3101044","37.8947103,-122.3099005","37.8938735,-122.3094499","37.8929615,-122.3091066","37.8925216,-122.3089778","37.8922212,-122.3089135","37.8919208,-122.3088706","37.8912342,-122.3088491","37.8907728,-122.308892","37.8892064,-122.309171","37.8885734,-122.3092568","37.8878546,-122.3092782","37.887125,-122.3093426","37.8865242,-122.3092997","37.8836811,-122.3086774","37.8827047,-122.3084414","37.8817821,-122.308259","37.8801513,-122.307905","37.8793895,-122.3077226","37.8776622,-122.3072827","37.8765249,-122.3069072","37.8734028,-122.3059738","37.8710961,-122.3052549","37.8667188,-122.3038495","37.8642726,-122.3031843","37.8633928,-122.302916","37.862159,-122.3025835","37.854445,-122.3003948","37.8510332,-122.2994614","37.8488767,-122.2989786","37.8477716,-122.2987533","37.8445745,-122.298013","37.8426754,-122.2976053","37.8422248,-122.2975302","37.8399181,-122.2970688","37.8385556,-122.2968328","37.838062,-122.2967148","37.8379655,-122.2967041","37.8370428,-122.2964144","37.8347147,-122.2953093","37.8328478,-122.2944617"],"note":[],"id":"M6","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.8328478,"longitude":-122.2944617},"instruction":"Continue on <span class=\"number\">I-580</span> toward <span class=\"sign\"><span lang=\"en\">Downtown Oakland</span>/<span lang=\"en\">Hayward</span>/<span lang=\"en\">Stockton</span>/<span lang=\"en\">I-880</span>/<span lang=\"en\">Alameda</span>/<span lang=\"en\">San Jose</span>/<span lang=\"en\">Airport</span></span>. <span class=\"distance-description\">Go for <span class=\"length\">428 m</span>.</span>","travelTime":20,"length":428,"shape":["37.8328478,-122.2944617","37.832644,-122.2943223","37.8324831,-122.2942364","37.8321826,-122.2941291","37.8316677,-122.2939682","37.8312814,-122.2938716","37.830863,-122.2937965","37.8305089,-122.2937858","37.8301013,-122.293818","37.829597,-122.2938824","37.8294146,-122.2938824","37.829082,-122.2938502"],"note":[],"id":"M7","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.829082,"longitude":-122.2938502},"instruction":"Keep <span class=\"direction\">right</span> onto <span class=\"number\">I-880</span> toward <span class=\"sign\"><span lang=\"en\">West Grand Avenue</span>/<span lang=\"en\">Alameda</span>/<span lang=\"en\">San Jose</span>/<span lang=\"en\">Airport</span></span>. <span class=\"distance-description\">Go for <span class=\"length\">48.8 km</span>.</span>","travelTime":1952,"length":48761,"shape":["37.829082,-122.2938502","37.8288352,-122.293818","37.8284705,-122.2936678","37.8282773,-122.2935712","37.8279448,-122.2933245","37.8275907,-122.2929704","37.8270543,-122.2922409","37.8268182,-122.2919834","37.8264749,-122.2916615","37.8261209,-122.2914469","37.8258848,-122.2913289","37.8257132,-122.2912753","37.8251874,-122.2911787","37.8247154,-122.2912002","37.8244579,-122.2912538","37.8242862,-122.2913074","37.8240502,-122.2914147","37.8234494,-122.2917366","37.8228915,-122.2920585","37.8219473,-122.2926486","37.8214753,-122.2929811","37.8201663,-122.2941077","37.8196406,-122.2946548","37.8191471,-122.2952449","37.8135145,-122.3023367","37.8131497,-122.3027766","37.8127956,-122.3031521","37.8123558,-122.3035491","37.8119051,-122.3039138","37.8114116,-122.304225","37.8106713,-122.304579","37.810446,-122.3046649","37.8101563,-122.3047507","37.8093946,-122.3048794","37.808826,-122.3049331","37.8079891,-122.3048794","37.8075171,-122.3048151","37.8073883,-122.3047936","37.8069592,-122.3046649","37.8064227,-122.3044825","37.8059292,-122.3042357","37.8056073,-122.3040533","37.8053391,-122.3038709","37.8050816,-122.3036778","37.8044379,-122.3031306","37.8041482,-122.3028302","37.8037405,-122.3023367","37.8034079,-122.3018539","37.803129,-122.3014247","37.8026354,-122.3004162","37.8023887,-122.2998047","37.802099,-122.2987854","37.8020346,-122.298485","37.8018951,-122.2976053","37.8018522,-122.2970152","37.8018415,-122.2961247","37.8019381,-122.2950304","37.8020239,-122.2943652","37.802453,-122.2918975","37.8026676,-122.2904921","37.8027534,-122.2896659","37.8027749,-122.2885716","37.8026569,-122.2870159","37.8025711,-122.2863936","37.8024209,-122.2856104","37.8022385,-122.2849131","37.8019273,-122.2839689","37.8016591,-122.2832394","37.8011763,-122.2820163","37.8003824,-122.2798598","37.8002751,-122.2796452","37.8001034,-122.2792161","37.8000176,-122.2790444","37.7990305,-122.2765768","37.7978933,-122.2735083","37.7974856,-122.272532","37.7969706,-122.2711265","37.7962303,-122.2692382","37.7958977,-122.2684336","37.7956295,-122.2677255","37.7949321,-122.2661054","37.7944922,-122.2653115","37.7940202,-122.2646248","37.7934194,-122.2638953","37.7930868,-122.263552","37.7923036,-122.2626078","37.7919173,-122.2621894","37.7913809,-122.2615457","37.7910697,-122.2611165","37.7907264,-122.2605908","37.7904153,-122.2599685","37.7902973,-122.2596788","37.7901149,-122.2591853","37.7898896,-122.2583163","37.7896857,-122.2572219","37.7895033,-122.2559667","37.7891386,-122.2538638","37.7889025,-122.2527695","37.7886236,-122.2517073","37.7882159,-122.2503984","37.7879584,-122.2496367","37.7872396,-122.2476411","37.7869713,-122.2469866","37.7867568,-122.2464931","37.7858019,-122.2446048","37.7848256,-122.2428453","37.7843106,-122.2418582","37.7837849,-122.2409356","37.7834201,-122.2404206","37.7828836,-122.2397768","37.7824116,-122.2393692","37.7818537,-122.2390473","37.7815318,-122.2388971","37.7812529,-122.2387898","37.7810061,-122.2387147","37.7797723,-122.2382748","37.7787745,-122.2378671","37.7784204,-122.2376955","37.7779055,-122.237395","37.777648,-122.2372127","37.7773583,-122.2369552","37.7768648,-122.2364616","37.7762425,-122.2357857","37.7758884,-122.2353029","37.77547,-122.2345841","37.7751803,-122.2340155","37.7749658,-122.2335112","37.7740753,-122.2308612","37.7739251,-122.2304535","37.7733672,-122.2287154","37.7731526,-122.2279966","37.7720475,-122.2247458","37.7717364,-122.2240913","37.7713716,-122.2234476","37.7710819,-122.2230184","37.7707171,-122.2225463","37.7705348,-122.2223747","37.7703846,-122.2221923","37.7702129,-122.2220314","37.769419,-122.2213769","37.7689469,-122.221055","37.7681315,-122.2204328","37.7669513,-122.2195852","37.7657068,-122.2186089","37.764945,-122.2179115","37.7644086,-122.2173858","37.7639365,-122.2169888","37.7607822,-122.21421","37.7589476,-122.2126329","37.7576709,-122.2114956","37.7564263,-122.2104228","37.7540123,-122.2082663","37.7532184,-122.2075903","37.7496135,-122.2044039","37.7439809,-122.1994901","37.742275,-122.1979666","37.7408481,-122.196722","37.7403975,-122.1963143","37.7403331,-122.1962392","37.7382946,-122.1945441","37.7382731,-122.1945119","37.7366102,-122.193085","37.7348614,-122.1915507","37.7322865,-122.1892548","37.7298295,-122.187109","37.7281451,-122.1855855","37.7273941,-122.1849632","37.7253127,-122.1831393","37.7250659,-122.182914","37.7249479,-122.1827853","37.7247655,-122.1826351","37.7241111,-122.1819913","37.71909,-122.1768951","37.7173197,-122.1750712","37.7149165,-122.172668","37.7138972,-122.1715951","37.7130497,-122.1705329","37.7124059,-122.1696532","37.710464,-122.1667778","37.708447,-122.1638596","37.707063,-122.1617675","37.7039731,-122.1573687","37.6997244,-122.1511996","37.6982224,-122.149086","37.6944566,-122.1435821","37.6926863,-122.1410286","37.691313,-122.1389687","37.6882768,-122.1346986","37.6874828,-122.1334648","37.6854014,-122.1304929","37.6848435,-122.1296668","37.6844466,-122.1291196","37.6826978,-122.1265876","37.6806486,-122.1237016","37.6794684,-122.1220064","37.6790714,-122.1213841","37.6779985,-122.1198392","37.6765501,-122.1175647","37.676003,-122.1167707","37.6748228,-122.1152365","37.6742113,-122.1143889","37.6736641,-122.1135843","37.6732242,-122.1129942","37.6727092,-122.1123612","37.6723659,-122.1119857","37.6720119,-122.1116209","37.6715183,-122.1111488","37.6711214,-122.110827","37.6705205,-122.1104085","37.6689649,-122.1094","37.6647913,-122.1068251","37.6642656,-122.1065247","37.6633644,-122.1059775","37.660135,-122.1039176","37.6543951,-122.1003556","37.6513052,-122.0984674","37.6487625,-122.0968366","37.6480114,-122.0962465","37.6448357,-122.0940149","37.6444817,-122.0937252","37.6432049,-122.0928562","37.6410377,-122.0913112","37.6387739,-122.0896697","37.6317143,-122.0846164","37.6239789,-122.0791233","37.6177025,-122.0746279","37.617327,-122.0743811","37.6164579,-122.0737267","37.6129603,-122.0712268","37.6086688,-122.068212","37.6086044,-122.0681477","37.6071453,-122.0670855","37.6036048,-122.064575","37.6002145,-122.0621395","37.5989377,-122.061249","37.5987554,-122.0611417","37.5967705,-122.0596933","37.5964165,-122.0594144","37.5952148,-122.0585883","37.5925541,-122.0566785","37.5900221,-122.054919","37.5852048,-122.0514858","37.5841856,-122.0508099","37.5836813,-122.050488","37.5821364,-122.0496619","37.5812995,-122.0493293","37.5783062,-122.0482564","37.572459,-122.0462608","37.5719011,-122.0460463","37.5714827,-122.0458531","37.5710535,-122.0456064","37.5705814,-122.0453918","37.5695729,-122.0447266","37.5692403,-122.0444798","37.5685751,-122.0439327","37.5680709,-122.0434928","37.5676847,-122.0431066","37.5638652,-122.0390081","37.5633931,-122.0384824","37.5618589,-122.0368731","37.5608397,-122.0357788","37.5602067,-122.035135","37.5556147,-122.0301569","37.5544238,-122.0289445","37.5540912,-122.0286441","37.5533187,-122.0279896","37.5511837,-122.026391","37.5496495,-122.0252645","37.548008,-122.0239878","37.5465596,-122.0229578","37.5438452,-122.0208871","37.5433731,-122.0205438","37.5429869,-122.0202863","37.5423646,-122.0198035","37.539264,-122.0174754","37.5384593,-122.0167994","37.537955,-122.0163167","37.5375152,-122.0158875","37.5369358,-122.0152223","37.536217,-122.0141923","37.5355947,-122.0131409","37.532419,-122.0073152","37.5296938,-122.0022082","37.5291896,-122.0013607","37.5278699,-121.998893","37.5257349,-121.9944942","37.5242007,-121.9914043","37.5232136,-121.9893551","37.5211,-121.9851279","37.5205421,-121.9841516","37.5198662,-121.9831431","37.519598,-121.9827998","37.5182676,-121.9811261","37.5137186,-121.9755042","37.5102317,-121.971159"],"note":[],"id":"M8","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.5102317,"longitude":-121.971159},"instruction":"Take exit <span class=\"exit\">15</span> toward <span class=\"sign\"><span lang=\"en\">Auto Mall Parkway</span>/<span lang=\"en\">Unitek College</span></span>. <span class=\"distance-description\">Go for <span class=\"length\">445 m</span>.</span>","travelTime":45,"length":445,"shape":["37.5102317,-121.971159","37.5098991,-121.9709229","37.5096416,-121.9706762","37.5088155,-121.9698071","37.5084507,-121.9694746","37.5083542,-121.9694209","37.5082254,-121.9693565","37.5080216,-121.9693244","37.5074959,-121.9693458","37.5073242,-121.969378","37.5071526,-121.9693887","37.5068414,-121.969378","37.5067127,-121.9692814"],"note":[],"id":"M9","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.5067127,"longitude":-121.9692814},"instruction":"Turn <span class=\"direction\">left</span> onto <span class=\"next-street\">Auto Mall Pkwy</span> toward <span class=\"sign\"><span lang=\"en\">Unitek College</span></span>. <span class=\"distance-description\">Go for <span class=\"length\">3.5 km</span>.</span>","travelTime":340,"length":3450,"shape":["37.5067127,-121.9692814","37.5066161,-121.9690669","37.5068629,-121.9685197","37.5085688,-121.9643676","37.5093949,-121.962415","37.5094914,-121.962136","37.5096738,-121.9614923","37.5097167,-121.9611919","37.5102639,-121.95822","37.5103176,-121.957866","37.5110471,-121.9540358","37.5112832,-121.9533706","37.5117123,-121.9523728","37.5117981,-121.9520295","37.5120234,-121.9509351","37.51212,-121.9503343","37.5121951,-121.949991","37.5122917,-121.9491971","37.5122273,-121.9476843","37.5122702,-121.9471586","37.5123775,-121.9463754","37.5124633,-121.9460106","37.5126779,-121.9452918","37.5128067,-121.944927","37.5129569,-121.94453","37.5132895,-121.9437468","37.5134611,-121.9434464","37.5136757,-121.94291","37.5140941,-121.9414937","37.5142658,-121.9408178","37.5148129,-121.938951","37.5149846,-121.9382751","37.5149846,-121.9377923","37.5149632,-121.9375026","37.5149202,-121.9372344","37.5147593,-121.9366872","37.5147378,-121.9365585","37.5146091,-121.9362473","37.5141156,-121.9351745","37.5139546,-121.9347453","37.5138795,-121.934402","37.5138581,-121.9340587","37.5138688,-121.9337583","37.5139439,-121.9333076","37.5140727,-121.9326854"],"note":[],"id":"M10","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.5140727,"longitude":-121.9326854},"instruction":"Turn <span class=\"direction\">left</span> onto <span class=\"next-street\">Laurel Canyon Way</span>. <span class=\"distance-description\">Go for <span class=\"length\">270 m</span>.</span>","travelTime":43,"length":270,"shape":["37.5140727,-121.9326854","37.514888,-121.9328785","37.5150275,-121.9329643","37.5151348,-121.9330609","37.5152636,-121.9332218","37.5153494,-121.9333613","37.5154245,-121.9336188","37.5155747,-121.9343376","37.5156069,-121.9344449","37.5156533,-121.9345286"],"note":[],"id":"M11","_type":"PrivateTransportManeuverType"},{"position":{"latitude":37.5156533,"longitude":-121.9345286},"instruction":"Arrive at <span class=\"street\">Laurel Canyon Way</span>. Your destination is on the left.","travelTime":0,"length":0,"shape":["37.5156533,-121.9345286"],"note":[{"type":"info","code":"previousIntersection","text":"The last intersection is <span class=\"street\">Laurel Canyon Ct</span>"}],"id":"M12","_type":"PrivateTransportManeuverType"}]}]}],"language":"en-us"}}
    maneuver = json['response']['route'][0]['leg'][0]['maneuver']
    points = []
    for m in maneuver:
        shape = m['shape']
        for s in shape:
            st = s.split(',')
            points.append((float(st[1]), float(st[0])))
    line = LineString(points)
    print(line)



def test_create_external_taz():
    # test()
    # stop_code()
    data_path = "/Users/edson/Dropbox/Private Structured data collection"
    sfcta_folder = os.path.join(data_path, "Data processing", "Raw", "Demand", "OD demand", "SFCTA demand data")
    sections_path = os.path.join(data_path, "Aimsun", "Inputs", "sections.shp")
    sections_df = gpd.GeoDataFrame.from_file(sections_path)
    sections_df = sections_df.to_crs(epsg=4326)
    sections_df = sections_df.set_geometry('geometry')
    create_external_taz(sfcta_folder, sections_df)
    pass


def stop_code(msg=None):
    if msg:
        raise (ValueError(msg))
    else:
        raise(ValueError('User stopped code'))


if __name__ == '__main__':
    test_create_external_taz()