# array analysis
import numpy as np
import pandas as pd
import sklearn.cluster as skc

# geo spacial data analysis
import geopandas as gpd
from shapely import wkt
from keplergl import KeplerGl

# assorted parsing and modeling tools
import math
import csv
from pytz import utc
from shutil import copyfile, copytree
from shapely.ops import nearest_points, unary_union
from shapely.geometry import Point, LineString

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
        nodes_df, crs='epsg:4326', geometry=gpd.points_from_xy(nodes_df.end_node_lng, nodes_df.end_node_lat))

    # Adding the end points and the start points together, only keeping some data
    nodes = nodes.append(gdf_bis)
    nodes = nodes[['leg_id', 'start_time', 'geometry']]

    # Joining the neighborhooed on the SFCTA points
    joined_nodes = gpd.sjoin(nodes, neighborhoods_shp, how='left', op='within')
    
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
    int_taz_shapefile, _ = loading_taz(output_taz)
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
    
    print("Shifting time...")
    shift_time(internal_trips, 'start_time', -8)

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
    flow_nb = flow_pems[flow_pems['Detector_Id']==pems_nb_id].loc[:, '14:0': '20:0']
    flow_sb = flow_pems[flow_pems['Detector_Id']==pems_sb_id].loc[:, '14:0': '20:0']
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