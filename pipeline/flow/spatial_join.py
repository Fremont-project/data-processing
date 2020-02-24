import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plot
from shapely import wkt
import keplergl as kp

def detectors_to_road_segments(year, dir_section, dir_detectors):
    print('\ncreating detectors_to_road_segments_%s.csv' % str(year))
    # aimsum (has linestring)
    section_path = dir_section + "/sections.shp"
    sections_df = gpd.GeoDataFrame.from_file(section_path)
    sections_df = sections_df.to_crs(epsg=4326)
    sections_df = sections_df[['id', 'eid', 'geometry']]
    print('number of road segments', sections_df.shape[0])
    # sections_df.to_csv('sections_df.csv')

    detector_path = dir_detectors + "/location_%s_detector.shp" % str(year)
    detectors_df = gpd.GeoDataFrame.from_file(detector_path)
    detectors_df = detectors_df.to_crs(epsg=4326)
    detectors_df = detectors_df[['OBJECTID', 'geometry', 'Year', 'Name']]
    # detectors.to_csv('detectors_%s.csv' % str(year))
    print('number of detectors', detectors_df.shape[0])

    detectors_to_roads_df = sjoin_nearest_polygon(detectors_df, sections_df, report_dist=True)

    print('Columns of Result data frame')
    print(detectors_to_roads_df.columns)
    detectors_to_roads_df.to_csv('detectors_to_road_segments_%s.csv' % str(year))


def sjoin_nearest_polygon(detectors_df, sections_df, search_dist=0.03, report_dist=False):
    distance_column_name = 'distance'
    if report_dist:
        if distance_column_name in detectors_df.columns:
            raise(ValueError("'dist' column exists in the left DataFrame. Remove it, or set 'report_dist' to False."))

    # geo data for accumulating results
    detectors_to_roads_df = gpd.GeoDataFrame()

    # Iterate over points and find closest polygon to point
    for _, point in detectors_df.iterrows():
        # point is a pandas series data type

        # Get lines (road segments) within search distance
        candidates = sections_df.loc[sections_df.intersects(point['geometry'].buffer(search_dist))]
        # candidates is a geo pandas data frame

        if len(candidates) == 0:
            print('\nDetector without road segment')
            print(point)
            detectors_to_roads_df = detectors_to_roads_df.append(point)
            continue
            # raise(Exception('No road segment found for point detector %s' % point))

        closest_line_distance = None

        geom_type = candidates['geometry'].iloc[0].geom_type
        if geom_type == 'LineString':
            min_dist = 9999999999
            min_idx = None
            for idx, row in candidates.iterrows():
                # row is a pandas series data type
                line = row['geometry']
                dist = line.distance(point['geometry'])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx

            # closest_line is a pandas series data type
            closest_line = sections_df.iloc[min_idx] # closes line (road) segment
            if report_dist:
                closest_line_distance = min_dist
        else:
            raise(Exception("Non supported input geometry type: %s" % geom_type))

        # Drop geometry from closest polygon
        closest_line = closest_line.drop('geometry')

        # join values
        join = point.append(closest_line)

        # Add information about distance to closest geometry if requested
        if report_dist:
            join[distance_column_name] = closest_line_distance

        detectors_to_roads_df = detectors_to_roads_df.append(join, ignore_index=True, sort=False)

    # rename some columns and rearrange
    detectors_to_roads_df = detectors_to_roads_df.rename(columns={'id': 'section_id', 'eid': 'section_eid'})
    detectors_to_roads_df = detectors_to_roads_df[['OBJECTID', 'Name', 'Year', 'geometry', 'section_id', 'section_eid', distance_column_name]]
    return detectors_to_roads_df


def find_duplicates(year, dir_sections):
    print('\nFinding detector to road duplicates for %s' % str(year))
    # load sections
    section_path = dir_sections + "/sections.shp"
    sections_df = gpd.GeoDataFrame.from_file(section_path)
    sections_df = sections_df.to_crs(epsg=4326)
    sections_df = sections_df[['id', 'eid', 'geometry']]

    # load year detectors_to_roads
    file_path = "detectors_to_road_segments_%s.csv" % str(year)
    detectors_to_roads_df = pd.read_csv(file_path)
    detectors_to_roads_df['geometry'] = detectors_to_roads_df['geometry'].apply(wkt.loads)
    detectors_to_roads_df = gpd.GeoDataFrame(detectors_to_roads_df, geometry='geometry')

    # write duplicates found to csv
    duplicates_df = find_duplicates_helper(detectors_to_roads_df, sections_df)
    if duplicates_df.shape[0] > 0:
        print('\nDuplicates written to duplicates_%s' % str(year))
        duplicates_df.to_csv('duplicates_%s.csv' % str(year))

        # plot
        axes = sections_df.plot(color='blue')
        duplicates_df.plot(ax=axes, color='red')
        plot.show()
    else:
        print('No duplicates found')


def find_duplicates_helper(detectors_to_roads_df, sections_df):
    road_id_to_detector_ids = {}
    for i, row in detectors_to_roads_df.iterrows():
        # row is pandas series data type
        # section_id = road_id and OBJECTID = detector id
        section_id = row['section_id'] # float
        if section_id not in road_id_to_detector_ids:
            road_id_to_detector_ids[section_id] = []
        road_id_to_detector_ids[section_id].append(row)

    ret_df = gpd.GeoDataFrame()
    for section_id, rows in road_id_to_detector_ids.items():
        # if multiple detectors to one road section
        if len(rows) > 1:
            # need geometry of points and geometry of line segment
            # k = road id, rows = points ids
            # geometry of points
            print("\nfound multiple detectors for road id %s" % int(section_id))
            for i, row in enumerate(rows):
                print("\ndetector %d info:" % i)
                print(row)
                ret_df = ret_df.append(row)

            # geometry of road id
            road_section = sections_df.loc[sections_df['id'] == int(section_id)]
            ret_df = ret_df.append(road_section)

    return ret_df


def create_kepler_map(dir_sections, dir_detectors):
    section_path = dir_sections + "/sections.shp"
    sections_df = gpd.GeoDataFrame.from_file(section_path)
    sections_df = sections_df.to_crs(epsg=4326)
    sections_df = sections_df[['id', 'eid', 'geometry']]
    print('number of road segments', sections_df.shape[0])

    detectors_2013_df = load_detector_data(dir_detectors + "/location_%s_detector.shp" % str(2013))
    detectors_2015_df = load_detector_data(dir_detectors + "/location_%s_detector.shp" % str(2015))
    detectors_2017_df = load_detector_data(dir_detectors + "/location_%s_detector.shp" % str(2017))
    detectors_2019_df = load_detector_data(dir_detectors + "/location_%s_detector.shp" % str(2019))

    map = kp.KeplerGl(height=600)
    map.add_data(data=sections_df, name='Road Sections')
    map.add_data(data=detectors_2013_df, name='Detectors 2013')
    map.add_data(data=detectors_2015_df, name='Detectors 2015')
    map.add_data(data=detectors_2017_df, name='Detectors 2017')
    map.add_data(data=detectors_2019_df, name='Detectors 2019')

    return map


def load_detector_data(file_path):
    detectors_df = gpd.GeoDataFrame.from_file(file_path)
    detectors_df = detectors_df.to_crs(epsg=4326)
    detectors_df = detectors_df[['OBJECTID', 'geometry', 'Year', 'Name']]
    return detectors_df


def main():
    # # find road segments for detectors
    # dir_detectors = 'detectors'
    # dir_sections = 'aimsum'
    # detectors_to_road_segments(2013, dir_sections, dir_detectors)
    # detectors_to_road_segments(2015, dir_sections, dir_detectors)
    # detectors_to_road_segments(2017, dir_sections, dir_detectors)
    # detectors_to_road_segments(2019, dir_sections, dir_detectors)
    #
    # # find duplicates, that is multiple detectors for one road, if any
    # find_duplicates(2013, dir_sections)
    # find_duplicates(2015, dir_sections)
    # find_duplicates(2017, dir_sections)
    # find_duplicates(2019, dir_sections)

    # visualize
    # map = create_kepler_map(dir_sections)
    pass

# local testing only
if __name__ == '__main__':
    main()