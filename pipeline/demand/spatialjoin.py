from geopandas import sjoin


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


def _select_gdf_by_geometry(gdf, position='origin', start_node_geometry_column='start_node_geometry', end_node_geometry_column='end_node_geometry'):
    if position == 'origin':
        gdf_to_join = gdf.set_geometry(start_node_geometry_column)
    elif position == 'destination':
        gdf_to_join = gdf.set_geometry(end_node_geometry_column)

    if position not in ['origin', 'destination']:
        raise ValueError(
            '{type} argument is incorrect, use "origin" or "destination"'.format(type=repr(position)))

    return gdf_to_join
