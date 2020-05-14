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
    api_key = None
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

                # todo use Google API to get speed limits on desired roads and save them to output

                # if its a Major road or # highway or on/off ramp
                if road_name or speed >= 100:
                    row_copy['properties']['speed'] = None  # speed by google api speed
                    output.write(row_copy)


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


def create_external_taz(dir_taz, sections_df, output_dir=None):
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

    distance_to_centroid_threshold = 0.005
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

        if min_distance < distance_to_centroid_threshold:
            # path intersection to centroid
            intersection_to_centroid = [(intersect_point.x, intersect_point.y), (closest_centroid.x, closest_centroid.y)]
            intersection_to_centroid_paths.append(LineString(intersection_to_centroid))

            # write result to csv
            info_point_to_center.append([point, project_center, closest_centroid, min_distance, path])

    if testing:
        kepler_map = KeplerGl(height=600)
        kepler_map.add_data(data=gpd.GeoDataFrame({'geometry': [project_center]}, crs='epsg:4326'), name='project_center')
        kepler_map.add_data(data=gpd.GeoDataFrame({'geometry': external_centroid_nodes}, crs='epsg:4326'), name='external_centroids')
        kepler_map.add_data(data=gpd.GeoDataFrame({'geometry': [project_delimitation_line]}, crs='epsg:4326'), name='project_delimitation')
        kepler_map.add_data(data=gpd.GeoDataFrame({'geometry': [external_delimitation]}, crs='epsg:4326'), name='external_delimitation')
        kepler_map.add_data(data=gpd.GeoDataFrame({'geometry': mesh_points}, crs='epsg:4326'), name='mesh_points')
        kepler_map.add_data(data=gpd.GeoDataFrame({'geometry': [l[-1] for l in info_point_to_center]}, crs='epsg:4326'), name='paths')
        kepler_map.add_data(data=gpd.GeoDataFrame({'geometry': intersection_to_centroid_paths}, crs='epsg:4326'), name='intersection_to_centroid_paths')
        file_path = 'mesh_points_to_external_centroids.html'
        if output_dir:
            file_path = os.path.join(output_dir, file_path)
        kepler_map.save_to_html(file_name=file_path)


    def to_csv(file_name, header, lines):
        def add_quotes(val):
            return "\"" + str(val) + "\"" if ',' in str(val) else str(val)

        csv = open(file_name, 'w')
        csv.write(header + '\n')
        for line in lines:
            csv.write(','.join(map(add_quotes, line)) + '\n')

    # write results to csv
    mesh_points_to_centroid_file_path = 'mesh_point_to_centroid.csv'
    if output_dir:
        mesh_points_to_centroid_file_path = os.path.join(output_dir, mesh_points_to_centroid_file_path)
    to_csv(mesh_points_to_centroid_file_path,
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


def test_create_external_taz():
    data_path = "/Users/edson/Dropbox/Private Structured data collection"
    sfcta_folder = os.path.join(data_path, "Data processing", "Raw", "Demand", "OD demand", "SFCTA demand data")
    sections_path = os.path.join(data_path, "Aimsun", "Inputs", "sections.shp")
    sections_df = gpd.GeoDataFrame.from_file(sections_path)
    sections_df = sections_df.to_crs(epsg=4326)
    sections_df = sections_df.set_geometry('geometry')
    output_dir = os.path.join(data_path, 'Data processing', 'Kepler maps', 'HereAPI')
    create_external_taz(sfcta_folder, sections_df, output_dir)
    pass


def stop_code(msg=None):
    if msg:
        raise (ValueError(msg))
    else:
        raise(ValueError('User stopped code'))


if __name__ == '__main__':
    test_create_external_taz()