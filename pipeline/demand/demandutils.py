import numpy as np
from geopandas import GeoDataFrame, sjoin
from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area


def add_point_geometry(df, lng_column='start_node_lng', lat_column='start_node_lat', geometry_column='geometry', crs_degree={'init': 'epsg:4326'}):
    """
    Add a new Point geometry column
    Parameters
    ----------
    df : DataFrame
        DataFrame representing demand legs (internal|starting|ending)
    lat_column : string
        Name of the column representing lattitude
    lng_column : string
        Name of the column representing longitude
    geometry_column : string
        Name of the column that will represent geometry column

    Returns
    -------
    df_with_geometry : GeoDataFrame
        GeoDataFrame representing demand legs (internal|starting|ending) with added point geometry.
    """
    # Process XY coordinates data as Point geometry
    from shapely.geometry import Point
    points = [Point(xy) for xy in zip(df[lng_column], df[lat_column])]

    gdf = GeoDataFrame(df, crs=crs_degree, geometry=points)
    gdf = gdf.rename(columns={'geometry': geometry_column}
                     ).set_geometry(geometry_column)
    return gdf


def spatial_join_nodes_with_centroids(gdf, centroid_zones, how='left', op='within', position='origin', start_node_geometry_column='start_node_geometry', end_node_geometry_column='end_node_geometry'):
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


def spatial_join_nodes_with_neighborhoods(gdf, neighborhoods, how='left', op='within'):
    gdf_to_join = sjoin(gdf, neighborhoods, how='left', op='within')

    for column in ['index_left', 'index_right', 'OBJECTID']:
        try:
            gdf_to_join.drop(column, axis=1, inplace=True)
        except KeyError:
            # ignore if there are no index columns
            pass

    return gdf_to_join


def voronoi_within_neighborhood(gdf, neighborhood_name, n_points=10, epsg=4326):
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


def voronoi_within_neighborhood_from_real_points(gdf, neighborhood_name, real_points, n_points=10, epsg=4326):
    area = gdf[gdf.NAME == neighborhood_name]

    area = area.to_crs(epsg=epsg)
    area_shape = area.iloc[0].geometry

    pts_count = len(real_points)

    if (n_points < pts_count):
        filter_count = n_points
    else:
        filter_count = pts_count

    point_coords = points_to_coords(real_points[:filter_count])

    # calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(
        point_coords, area_shape)

    return poly_shapes, pts, poly_to_pt_assignments


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
