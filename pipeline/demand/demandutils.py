import numpy as np
from geopandas import GeoDataFrame, sjoin
from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area


def add_point_geometry(input_path, output_path, lng_column='lng', lat_column='lat', geometry_column='geometry', crs='epsg:4326'):
    """
    Add a new Point geometry column
    Parameters
    ----------
    path : String
        Path to DataFrame
    lat_column : string
        Name of the column representing lattitude
    lng_column : string
        Name of the column representing longitude
    geometry_column : string
        Name of the column that will represent geometry column

    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame representing demand legs (internal|starting|ending) with added point geometry.
    """
    # Process XY coordinates data as Point geometry
    import pandas as pd
    from shapely.geometry import Point

    df = pd.read_csv(input_path)
    points = [Point(xy) for xy in zip(df[lng_column], df[lat_column])]

    gdf = GeoDataFrame(df, crs=crs, geometry=points)
    gdf = gdf.rename(columns={'geometry': geometry_column}
                     ).set_geometry(geometry_column)
    gdf = gdf[['leg_id', 'start_time', geometry_column]]
    gdf.to_file(output_path)


def neighborhoods_touching_internal_legs(neighborhood_path, points_gdf_path, output_path):
    import geopandas as gpd

    neighborhoods_shp = GeoDataFrame.from_file(neighborhood_path)
    neighborhoods_shp = neighborhoods_shp.to_crs('epsg:4326')

    assert(neighborhoods_shp.crs == 'epsg:4326'),'Neighborhoods shapefile Coordinate Reference System missmatch! CRS is not EPSG:4326.'

    int_neighborhood_names = _get_internal_neighborhoods(points_gdf_path, neighborhood_path)

    points_gdf = gpd.GeoDataFrame.from_file(points_gdf_path)
    points_gdf = points_gdf.to_crs('epsg:4326')

    int_neighborhoods = gpd.GeoDataFrame()

    for neighborhood_name in int_neighborhood_names['NAME'].unique():
        particular_neighborhood = neighborhoods_shp[neighborhoods_shp.NAME == neighborhood_name]
        int_neighborhoods = int_neighborhoods.append(particular_neighborhood, ignore_index=True)

    print('Filtered {x} neighborhoods.'.format(x=repr(len(int_neighborhoods['NAME'].unique()))))
    int_neighborhoods.to_file(output_path)


def geovor_in_neighborhoods_random_points(neighborhood_path, points_gdf_path, output_path):
    import numpy as np
    import geopandas as gpd

    np.random.seed(123)

    neighborhoods_shp = GeoDataFrame.from_file(neighborhood_path)
    neighborhoods_shp = neighborhoods_shp.to_crs('epsg:4326')

    assert(neighborhoods_shp.crs == 'epsg:4326'),'Neighborhoods shapefile Coordinate Reference System missmatch! CRS is not EPSG:4326.'

    int_neighborhoods = _get_internal_neighborhoods(points_gdf_path, neighborhood_path)

    partitioned_neighborhood_centroids = gpd.GeoDataFrame()

    init_centroid_index = 100

    for neighborhood_index, neighborhood_name in enumerate(int_neighborhoods['NAME'].unique()):
        neighborhood_id = neighborhoods_shp.loc[neighborhood_index]['OBJECTID']
        poly_shapes, pts, poly_to_pt_assignments = _voronoi_within_neighborhood_from_random_points(neighborhoods_shp, neighborhood_name, n_points=15)

        for index, pp in enumerate(poly_shapes):
            init_centroid_index += 1
            new_row = {'CentroidID':int(init_centroid_index), 'centroid_lng':pp.centroid.x, 'centroid_lat':pp.centroid.y, 'NeighborhoodID':neighborhood_id, 'neighborhood_name':neighborhood_name, 'geometry':pp}
            partitioned_neighborhood_centroids = partitioned_neighborhood_centroids.append(new_row, ignore_index=True)

    print('Created {x} centroids within {y} neighborhoods.'.format(x=repr(len(partitioned_neighborhood_centroids)), y=repr(len(int_neighborhoods['NAME'].unique()))))
    partitioned_neighborhood_centroids.to_csv(output_path)


def geovor_in_neighborhoods_real_points(neighborhood_path, points_gdf_path, output_path):
    import geopandas as gpd

    neighborhoods_shp = GeoDataFrame.from_file(neighborhood_path)
    neighborhoods_shp = neighborhoods_shp.to_crs('epsg:4326')

    assert(neighborhoods_shp.crs == 'epsg:4326'),'Neighborhoods shapefile Coordinate Reference System missmatch! CRS is not EPSG:4326.'

    int_neighborhoods = _get_internal_neighborhoods(points_gdf_path, neighborhood_path)

    points_gdf = gpd.GeoDataFrame.from_file(points_gdf_path)

    partitioned_neighborhood_centroids = gpd.GeoDataFrame()

    init_centroid_index = 100

    for neighborhood_index, neighborhood_name in enumerate(int_neighborhoods['NAME'].unique()):
        neighborhood_id = neighborhoods_shp.loc[neighborhood_index]['OBJECTID']
        poly_shapes, pts, poly_to_pt_assignments = _voronoi_within_neighborhood_from_real_points(
            neighborhoods_shp, neighborhood_name, points_gdf.geometry, n_points=15)

        for index, pp in enumerate(poly_shapes):
            init_centroid_index += 1
            new_row = {'CentroidID': int(init_centroid_index), 'centroid_lng': pp.centroid.x, 'centroid_lat': pp.centroid.y,
                       'NeighborhoodID': neighborhood_id, 'neighborhood_name': neighborhood_name, 'geometry': pp}
            partitioned_neighborhood_centroids = partitioned_neighborhood_centroids.append(
                new_row, ignore_index=True)

    print('Created {x} centroids within {y} neighborhoods.'.format(x=repr(len(partitioned_neighborhood_centroids)), y=repr(len(int_neighborhoods['NAME'].unique()))))
    partitioned_neighborhood_centroids.to_csv(output_path)


def visual_check_internal_taz(neighborhoods_path, points_path):
    from keplergl import KeplerGl
    import pandas as pd
    import geopandas as gpd

    nodes = gpd.GeoDataFrame.from_file(points_path)
    nodes = nodes.to_crs('epsg:4326')

    internal_neighborhoods = pd.read_csv(neighborhoods_path)

    fremont_map = KeplerGl(height=600)
    fremont_map.add_data(data=nodes, name="Nodes")
    fremont_map.add_data(data=internal_neighborhoods, name="Fremont neighborhoods")
    return fremont_map

#
# Internal functions
#


def _select_gdf_by_geometry(gdf, position='origin', start_node_geometry_column='start_node_geometry', end_node_geometry_column='end_node_geometry'):
    if position == 'origin':
        gdf_to_join = gdf.set_geometry(start_node_geometry_column)
    elif position == 'destination':
        gdf_to_join = gdf.set_geometry(end_node_geometry_column)

    if position not in ['origin', 'destination']:
        raise ValueError(
            '{type} argument is incorrect, use "origin" or "destination"'.format(type=repr(position)))

    return gdf_to_join


def _spatial_join_nodes_with_centroids(gdf, centroid_zones, how='left', op='within', position='origin', start_node_geometry_column='start_node_geometry', end_node_geometry_column='end_node_geometry'):
    gdf_to_join = _select_gdf_by_geometry(
        gdf, position, start_node_geometry_column, end_node_geometry_column)

    centroid_id_column = "CentroidID_O" if type == 'origin' else "CentroidID_D"

    gdf_to_join = sjoin(gdf_to_join, centroid_zones, how='left', op='within')
    gdf_to_join.rename(
        columns={
            "CentroidID": centroid_id_column
        },
        inplace=True
    )

    for column in ['index_left', 'index_right', 'OBJECTID']:
        try:
            gdf_to_join.drop(column, axis=1, inplace=True)
        except KeyError:
            # ignore if there are no index columns
            pass

    return gdf_to_join


def _spatial_join_nodes_with_neighborhoods(gdf, neighborhoods, how='left', op='within'):
    gdf_to_join = sjoin(gdf, neighborhoods, how=how, op=op)

    for column in ['index_left', 'index_right']:
        try:
            gdf_to_join.drop(column, axis=1, inplace=True)
        except KeyError:
            # ignore if there are no index columns
            pass

    return gdf_to_join

def _voronoi_within_neighborhood_from_real_points(gdf, neighborhood_name, real_points, n_points=10, epsg=4326):
    area = gdf[gdf.NAME == neighborhood_name]

    area = area.to_crs(epsg=epsg)
    area_shape = area.iloc[0].geometry

    pts_count = len(real_points)

    if (n_points < pts_count):
        filter_count = n_points
    else:
        filter_count = pts_count

    print(real_points[:filter_count].head())
    point_coords = points_to_coords(real_points[:filter_count])

    # calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(
        point_coords, area_shape)

    return poly_shapes, pts, poly_to_pt_assignments


def _voronoi_within_neighborhood_from_random_points(gdf, neighborhood_name, n_points=10, epsg=4326):
    import numpy as np
    np.random.seed(123)

    area = gdf[gdf.NAME == neighborhood_name]

    area = area.to_crs(epsg=epsg)
    area_shape = area.iloc[0].geometry

    # generate some random points within the bounds
    minx, miny, maxx, maxy = area_shape.bounds

    randx = np.random.uniform(minx, maxx, n_points)
    randy = np.random.uniform(miny, maxy, n_points)
    point_coords = np.vstack((randx, randy)).T

    # use only the points inside the geographic area
    pts = [p for p in coords_to_points(point_coords) if p.within(
        area_shape)]  # converts to shapely Point

    # convert back to simple NumPy coordinate array
    point_coords = points_to_coords(pts)

    del pts
    # calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(
        point_coords, area_shape)

    return poly_shapes, pts, poly_to_pt_assignments


def _get_internal_neighborhoods(leg_path, neighborhood_path):
    import geopandas as gpd

    nodes = gpd.GeoDataFrame.from_file(leg_path)
    nodes = nodes.to_crs('epsg:4326')

    neighborhoods = gpd.GeoDataFrame.from_file(neighborhood_path)
    neighborhoods = neighborhoods.to_crs('epsg:4326')

    assert(neighborhoods.crs == 'epsg:4326'),'Neighborhoods shapefile Coordinate Reference System missmatch! CRS is not EPSG:4326.'

    internal_neighborhoods = _spatial_join_nodes_with_neighborhoods(nodes, neighborhoods, how='inner', op='intersects')
    internal_neighborhoods = internal_neighborhoods[['REGION','NAME','OBJECTID']]
    return internal_neighborhoods
