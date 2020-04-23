import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plot
from shapely import wkt
import keplergl as kp
import numpy as np

# Aimsum sections to detectors
# Name of columns in all output csv files
ROAD_ID_NAME = 'Road_Id'  # aimsum eid for a road, renamed to Road_Id for user clarity
DETECTOR_ID_NAME = 'Detector_Id'
DISTANCE_NAME = 'Distance'


def streetlines_to_detectors(dir_streetline, dir_detectors, dir_output):
    """
    Create lines_to_detectors.csv file where each row is a street line with a corresponding closest 
     detector for each year and pems. Desired columns for output file are: 
     OBJECTID, Shape_Length, Name, Direction, 2013, 2015, 2017, 2019, PeMS.
     Where Name is the streetline name and years/pems are the detector ids.
    
    :param dir_streetline: directory of Streetline.shp file
    :param dir_detectors: directory of detector shape files, location_year_detector.shp and pems_detectors.shp, 
        where year takes values (2013, 2015, 2017, 2019)
    :param dir_output: directory for outputting lines_to_detectors.csv file
    
    :return: writes lines_to_detectors.csv as described above
    """
    print('\nCreating lines_to_detectors.csv file')

    # load streetline
    streetline_df = load_streetline_data(dir_streetline + 'Streetline.shp')

    # load detectors
    dic_detectors_df = {}
    years = [2013, 2015, 2017, 2019, 'pems']
    for year in years:
        dic_detectors_df[str(year)] = load_detector_data(dir_detectors + get_shape_file_name(year))

    # main idea: iterate through all detectors for all years
    streetline_to_all_detectors_dic = {}
    object_id_to_pems_distance = {}
    for year, detectors_df in dic_detectors_df.items():
        detector_year_to_streetline_df, _ = join_detector_to_nearest_road(detectors_df, streetline_df,
                                                              search_dist=1e-4, report_dist=True)

        for _, row_for_year in detector_year_to_streetline_df.iterrows():
            object_id = row_for_year['OBJECTID'] # used as identifier of a streetline

            if np.isnan(object_id):
                continue # if detector is not assigned to a streetline, skip it

            # create copy and get desired data
            row_for_year = pd.Series(row_for_year)
            row_for_year = row_for_year.rename({DETECTOR_ID_NAME: year})
            row_for_year[year] = str(int(row_for_year[year])) # detector id
            # drop undesired columns
            # row_for_year = row_for_year.drop(['geometry', DISTANCE_NAME])
            row_for_year = row_for_year.drop(['geometry'])

            # if streetline has been seen before append detector id else add it as a new row
            if object_id in streetline_to_all_detectors_dic:
                row_all_years = streetline_to_all_detectors_dic[object_id]
                if year in row_all_years:
                    # year seen before, multiple detectors to one streetline for this year

                    # handle pems manually, multiple pems detectors assigned to one streetline
                    # with near zero distance <1e-6, taking the detector with smallest distance is the right one
                    # if interested, multiple pems detectors assigned to one road can be found in duplicates_pems.csv
                    # in the output folder
                    if year == 'pems':
                        if row_for_year[DISTANCE_NAME] < object_id_to_pems_distance[object_id]:
                            row_all_years[year] = row_for_year[year]
                            object_id_to_pems_distance[object_id] = row_for_year[DISTANCE_NAME]
                    else:
                        # year seen before, multiple detectors to one streetline for this year
                        row_all_years[year] += ' - ' + row_for_year[year]
                else:
                    # new year not seen before
                    row_all_years[year] = row_for_year[year]
            else:
                if year == 'pems':
                    object_id_to_pems_distance[object_id] = row_for_year[DISTANCE_NAME]
                streetline_to_all_detectors_dic[object_id] = row_for_year

    # creat dataframe to write to csv
    streetline_to_all_detectors_df = pd.DataFrame(streetline_to_all_detectors_dic.values())
    streetline_to_all_detectors_df = streetline_to_all_detectors_df[
        ['OBJECTID', 'Shape_Length', 'Name', 'Direction', '2013', '2015','2017', '2019', 'pems']]
    streetline_to_all_detectors_df.to_csv(dir_output + 'lines_to_detectors.csv', index=False)


def detectors_to_road_segments(year, dir_section, dir_detectors, dir_output):
    """
    Writes detectors_to_road_segments_year.csv file where each row is a detector with a corresponding closest
    road section. Note that 'year' can takes values as given in the input.
    
    :param year: year of detector data to load from .shp file location_year_detector.shp' 
    :param dir_section: directory of section .shp file 'sections.shp' to load road section data (aimsum network)
    :param dir_detectors: directory of detector .shp file 'location_year_detector.shp' to load detector data
    :param dir_output: directory for outputting detectors_to_road_segments_year.csv file
    
    :return: Writes a detectors_to_road_segments_year.csv file as described above.  
    """
    print('\nCreating detectors_to_road_segments_%s.csv' % str(year))
    # aimsum (has linestring geometry)
    sections_df = load_section_data(dir_section + "sections.shp")
    print('number of road segments', sections_df.shape[0])

    # detectors (has point geometry)
    detectors_df = load_detector_data(dir_detectors + get_shape_file_name(year))
    print('number of detectors', detectors_df.shape[0])

    report_distance = False
    search_dist = 1e-4
    detectors_to_roads_df, detectors_wout_roads_df = join_detector_to_nearest_road(detectors_df, sections_df,
                                                          search_dist=search_dist, report_dist=report_distance)
    # drop columns and rearrange as desired
    detectors_to_roads_df = detectors_to_roads_df.drop(columns=['geometry'])
    if report_distance:
        detectors_to_roads_df = detectors_to_roads_df[[DETECTOR_ID_NAME, ROAD_ID_NAME, DISTANCE_NAME]]
    else:
        detectors_to_roads_df = detectors_to_roads_df[[DETECTOR_ID_NAME, ROAD_ID_NAME]]

    # write to csv
    detectors_to_roads_df = detectors_to_roads_df.sort_values(by=[DETECTOR_ID_NAME])
    detectors_to_roads_df.to_csv(dir_output + 'detectors_to_road_segments_%s.csv' % str(year), index=False)

    # write detectors wout road assignment to csv
    if not detectors_wout_roads_df.empty:
        print('Detectors w/out road assignment found, with search radius of ' + str(search_dist))
        print('Writing detectors_without_road_segments_%s.csv' % str(year))
        detectors_wout_roads_df.to_csv(dir_output + 'detectors_without_road_segments_%s.csv' % str(year), index=False)


def join_detector_to_nearest_road(detectors_df, sections_df, search_dist=1e-4, report_dist=False):
    """
    Iterates over all detectors and finds nearest road to it spatially.
    The detectors have point geometry and road sections have line geometry, thus we for each point
    we find closest line to it.
    
    Parameters
    :param detectors_df:  geodataframe of detectors
    :param sections_df: geodataframe of road sections
    :param search_dist: search distance for a detector, that is we only consider 
        roads within a search_dist radius 
    :param report_dist: if True, adds a distance column of detector to its closest road section
     
    Output
    :return: geodataframe where rows are detectors with a corresponding closest road section
    """
    if report_dist:
        if DISTANCE_NAME in detectors_df.columns:
            raise(ValueError("'dist' column exists in the left DataFrame. Remove it, or set 'report_dist' to False."))

    # geo data for accumulating results
    detectors_to_roads_df = gpd.GeoDataFrame()
    detectors_without_roads_df = gpd.GeoDataFrame()

    # Iterate over points and find closest line to point
    for _, point in detectors_df.iterrows():
        # point is a pandas series data type

        # Get lines (road segments) within search distance
        candidates = sections_df.loc[sections_df.intersects(point['geometry'].buffer(search_dist))]
        # candidates is a list of geo pandas data frame

        # if no roads found, add detector to detectors without roads
        if len(candidates) == 0:
            detectors_without_roads_df = detectors_without_roads_df.append(point)
            continue

        closest_line_distance = None

        # find closest road from candidates
        geom_type = candidates['geometry'].iloc[0].geom_type
        if geom_type == 'LineString':
            min_dist = 9999999999
            min_idx = None
            for idx, row in candidates.iterrows():
                # row is a pandas series data type representing the road
                line = row['geometry']
                dist = line.distance(point['geometry'])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx

            # closest_line is a pandas series data type
            # note: min_idx is an index with reference to sections_df not candidates
            closest_line = sections_df.iloc[min_idx]
            if report_dist:
                closest_line_distance = min_dist
        else:
            raise(Exception("Non supported input geometry type: %s" % geom_type))

        # Drop geometry of line
        closest_line = closest_line.drop('geometry')

        # join values of detector and road
        join = point.append(closest_line)

        # Add information about distance to closest geometry if requested
        if report_dist:
            join[DISTANCE_NAME] = closest_line_distance

        # append to result dataframe
        detectors_to_roads_df = detectors_to_roads_df.append(join, ignore_index=True, sort=False)

    return detectors_to_roads_df, detectors_without_roads_df


def find_duplicates(year, dir_sections, dir_detectors, dir_output, show_plot=False):
    """
    Finds duplicates, that is multiple detectors assigned to one road section, if any. 
    
    Parameters
    :param year: year of the detectors_to_road_segments_year.csv data
    :param dir_sections: directory of sections .shp file (aimsum network)
    
    Output
    :return: Writes duplicates_year.csv file only if any duplicates were found. 
    """
    print('\nFinding detector to road duplicates for %s' % str(year))
    # load sections
    sections_df = load_section_data(dir_sections + "sections.shp")

    # load detectors
    detectors_df = load_detector_data(dir_detectors + get_shape_file_name(year))

    # load year detectors_to_roads
    detectors_to_roads_df = pd.read_csv(dir_output + "detectors_to_road_segments_%s.csv" % str(year))
    # need to parse and tell it we have geometries
    # detectors_to_roads_df['geometry'] = detectors_to_roads_df['geometry'].apply(wkt.loads)
    # detectors_to_roads_df = gpd.GeoDataFrame(detectors_to_roads_df, geometry='geometry')

    # write duplicates found to csv
    duplicates_df = find_duplicates_helper(detectors_to_roads_df, sections_df, detectors_df)
    if duplicates_df.shape[0] > 0:
        print('\nDuplicates written to duplicates_%s.csv' % str(year))
        duplicates_df.to_csv(dir_output + 'duplicates_%s.csv' % str(year), index=False)

        # plot duplicates
        if show_plot:
            axes = sections_df.plot(color='blue')
            duplicates_df.plot(ax=axes, color='red')
            plot.show()
    else:
        print('No duplicates found')


def find_duplicates_helper(detectors_to_roads_df, sections_df, detectors_df, verbose=False):
    # group detectors by road ids
    road_id_to_detectors = {}
    for i, detector in detectors_to_roads_df.iterrows():
        # row is pandas series data type
        # section_id = road id and OBJECTID = detector id
        road_id = detector[ROAD_ID_NAME]  # float
        if road_id not in road_id_to_detectors:
            road_id_to_detectors[road_id] = []
        road_id_to_detectors[road_id].append(detector)

    # find roads with multiple detectors
    # iterate over the groups and if group size > 1 write it to ret_df
    ret_df = gpd.GeoDataFrame()
    for road_id, detectors in road_id_to_detectors.items():
        if len(detectors) > 1:
            # need geometry of points and geometry of line segment
            if verbose:
                print("\n found multiple detectors for road id %s" % int(road_id))

            # add the detectors
            for i, detector in enumerate(detectors):
                if verbose:
                    print("\n detector %d info:" % i)
                    print(detector)
                detector_info = detectors_df.loc[detectors_df[DETECTOR_ID_NAME] == int(detector[DETECTOR_ID_NAME])]
                detector['geometry'] = detector_info.iloc[0]['geometry']  # there should only be one match
                ret_df = ret_df.append(detector)


            # then add the road
            road_section = sections_df.loc[sections_df[ROAD_ID_NAME] == int(road_id)]
            ret_df = ret_df.append(road_section)

    return ret_df


# def create_kepler_map_streetline(dir_sections, dir_detectors, years):
#     """
#     Creates kepler map to be visualized in a notebook

#     Parameters
#     :param dir_sections: directory of sections.shp file
#     :param dir_detectors: directory of location_year_detector.shp files for all years
#     """
#     print("\nCreating Kepler Map")
#     sections_df = load_section_data(dir_sections + "Streetline.shp")
#     print('number of road segments', sections_df.shape[0])

#     detectors_dic = {i: load_detector_data(dir_detectors + get_shape_file_name(i)) for i in [2013, 2015, 2017, 2019, 'PeMS']}

#     map = kp.KeplerGl(height=600)
#     map.add_data(data=sections_df, name='Road Sections')
#     for year in years:
#         map.add_data(data=detectors_dic[year], name='Detectors %s' % str(year))

#     return map


def load_section_data(file_path):
    sections_df = gpd.GeoDataFrame.from_file(file_path)
    sections_df = sections_df.to_crs(epsg=4326)
    sections_df = sections_df[['eid', 'geometry']]
    # sections_df['eid'] = sections_df['eid'].astype('int32')
    sections_df = sections_df.rename(columns={'eid': ROAD_ID_NAME})
    return sections_df


def load_detector_data(file_path):
    detectors_df = gpd.GeoDataFrame.from_file(file_path)
    detectors_df = detectors_df.to_crs(epsg=4326)

    id_name = None
    if 'Id' in detectors_df.columns:
        id_name = 'Id'
    elif 'ID' in detectors_df.columns:
        id_name = 'ID'
    else:
        raise(ValueError("Can't find detector Id column in file: " + file_path))

    detectors_df = detectors_df[[id_name, 'geometry']]
    detectors_df = detectors_df.rename(columns={id_name: DETECTOR_ID_NAME})
    return detectors_df


def get_shape_file_name(val):
    if isinstance(val, int):
        return "location_%s_detector.shp" % str(val)
    elif isinstance(val, str):
        return "%s_detectors.shp" % val
    else:
        raise(ValueError("Expected year or pems, got " + str(val)))


def load_streetline_data(file_path):
    streetline_df = gpd.GeoDataFrame.from_file(file_path)
    streetline_df = streetline_df.to_crs(epsg=4326)
    # OBJECTID, Shape_Length, Name, Direction, 2013, 2015, 2017, 2019, PeMS.
    streetline_df = streetline_df[['OBJECTID', 'Shape_Leng', 'Name', 'Direction', 'geometry']]
    streetline_df = streetline_df.rename(columns={'Shape_Leng': 'Shape_Length'})
    return streetline_df

"""
codes below are for local testing only
"""
def run_detectors_to_aimsum():
    # # find road segments for detectors
    # detectors_folder = 'detectors/'
    # sections_folder = 'aimsum/'
    # output_folder = 'test_output/'
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    detectors_folder = data_process_folder + "Raw/Demand/Flow_speed/detectors/"
    sections_folder = data_process_folder + "Raw/Network/Aimsun/"
    output_folder = data_process_folder + "Temporary exports to be copied to processed data/Network/Infrastructure/Detectors/"

    detectors_to_road_segments(2013, sections_folder, detectors_folder, output_folder)
    detectors_to_road_segments(2015, sections_folder, detectors_folder, output_folder)
    detectors_to_road_segments(2017, sections_folder, detectors_folder, output_folder)
    detectors_to_road_segments(2019, sections_folder, detectors_folder, output_folder)
    detectors_to_road_segments('pems', sections_folder, detectors_folder, output_folder)

    #
    # # find duplicates, that is multiple detectors for one road, if any
    find_duplicates(2013, sections_folder, detectors_folder, output_folder)
    find_duplicates(2015, sections_folder, detectors_folder, output_folder)
    find_duplicates(2017, sections_folder, detectors_folder, output_folder)
    find_duplicates(2019, sections_folder, detectors_folder, output_folder)
    find_duplicates('pems', sections_folder, detectors_folder, output_folder)

    # visualize
    # map = create_kepler_map(sections_folder, detectors_folder)

def run_detectors_to_streetline():
    # detectors_folder = 'detectors/'
    # streetline_folder = 'streetline/'
    # output_folder = 'test_output/'
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    streetline_folder = data_process_folder + "Raw/Demand/Flow_speed/Road section/"
    detectors_folder = data_process_folder + "Raw/Demand/Flow_speed/detectors/"
    output_folder = data_process_folder + "Temporary exports to be copied to processed data/Network/Infrastructure/Detectors/"

    streetlines_to_detectors(streetline_folder, detectors_folder, output_folder)

def test_write_detector_shp_to_csv():
    # dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    # data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    # detectors_folder = data_process_folder + "Raw/Demand/Flow_speed/detectors/"
    df = load_detector_data('location_2019_detector.shp')
    df.to_csv('2019_shp_detectors.csv')

def raise_exception():
    raise(Exception('stop here'))

if __name__ == '__main__':
    # run_detectors_to_streetline()
    # run_detectors_to_aimsum()
    # test_write_detector_shp_to_csv()
    pass