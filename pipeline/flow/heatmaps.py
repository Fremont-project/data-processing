import geopandas as gpd
import pandas as pd
import numpy as np
import keplergl as kp
import json

heatmap_24hr_config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': 'wfpgbp', 'type': 'geojson', 'config': {'dataId': 'label_name', 'label': 'label_name', 'color': [183, 136, 94], 'columns': {'geojson': 'geometry'}, 'isVisible': True, 'visConfig': {'opacity': 0.8, 'thickness': 2, 'strokeColor': None, 'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 'strokeColorRange': {'name': 'Custom Palette', 'type': 'custom', 'category': 'Custom', 'colors': ['#1EF013', '#E49309', '#1DC9DD', '#F364F3', '#4C0369']}, 'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True, 'filled': False, 'enable3d': False, 'wireframe': False}, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear', 'strokeColorField': {'name': 'Avg 2-Way 24hr Counts', 'type': 'real'}, 'strokeColorScale': 'quantile', 'heightField': None, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}], 'interactionConfig': {'tooltip': {'fieldsToShow': {'label_name': ['Name', 'Avg 2-Way 24hr Counts']}, 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': [], 'animationConfig': {'currentTime': None, 'speed': 1}}, 'mapState': {'bearing': 0, 'dragRotate': False, 'latitude': 37.51747575092308, 'longitude': -121.93332419092869, 'pitch': 0, 'zoom': 12.223268867164116, 'isSplit': False}, 'mapStyle': {'styleType': 'dark', 'topLayerGroups': {}, 'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 'threeDBuildingColor': [9.665468314072013, 17.18305478057247, 31.1442867897876], 'mapStyles': {}}}}

def heatmap_24hr_counts(streetline_dir, flow_section_dir, export_dir):

    # load road sections and their line geometries
    streetline_df = gpd.GeoDataFrame.from_file(streetline_dir + 'Streetline.shp')
    streetline_df = streetline_df[['Name', 'Direction', 'OBJECTID', 'geometry']]
    streetline_df = streetline_df.set_geometry('geometry')
    streetline_df = streetline_df.to_crs(epsg=4326)
    streetline_df = streetline_df.set_index(['Name', 'Direction'])
    streetline_df.to_csv('streetline.csv')

    # read 2017 flow data for all 3 days
    flow_df = pd.read_csv(flow_section_dir + '/flow_processed_section.csv')
    flow_df = flow_df.set_index(['Name', 'Direction'])

    # keep only 2017 data timesteps of day 1
    years = ['2013', '2015', '2017', '2019']
    map_year_to_data = {}
    kepler_maps = []
    for year in years:
        # get only time step data for given year
        timestep_name = 'Day 1 - %s' % year
        timestep_cols = [col for col in flow_df.columns if timestep_name in col]
        year_flow_df = flow_df[timestep_cols]

        # sum over column timesteps
        # year_flow_df.loc[:, 'Day 1 Sum'] = np.nansum(year_flow_df.values, axis=1)
        # year_flow_df = year_flow_df.assign({'Day 1 Sum': np.nansum(year_flow_df.values, axis=1)})
        year_flow_df['Day 1 Sum'] = np.nansum(year_flow_df.values, axis=1)
        year_flow_df = year_flow_df[['Day 1 Sum']]  # drop everything else
        year_flow_df = year_flow_df.join(streetline_df)  # get geometries

        # for each road name take the avg over directions, accumulate values in list first
        year_flow_df = year_flow_df.reset_index()
        map_name_to_sum = {}
        map_name_to_geo = {}
        for i, row in year_flow_df.iterrows():
            name = row['Name'].replace('SB', '').replace('NB', '').replace('WB', '').replace('EB', '').strip()
            if name not in map_name_to_sum:
                map_name_to_sum[name] = []
            map_name_to_sum[name].append(row['Day 1 Sum'])
            map_name_to_geo[name] = row['geometry']  # keep any one geometry

        # do the average and keep only those greater than 0
        delete_keys = []  # avoid concurrent dic modification error
        for name, sums in map_name_to_sum.items():
            if sums:
                avg = np.nanmean(np.array(sums))
                if avg > 0:
                    map_name_to_sum[name] = avg
                else:
                    delete_keys.append(name)

        for name in delete_keys:
            del map_name_to_geo[name]
            del map_name_to_sum[name]

        # output file is road name and 24hr 2-way avg flow
        data_dic = {'Name': [], 'Avg 2-Way 24hr Counts': [], 'geometry': []}
        for name, avg in map_name_to_sum.items():
            data_dic['Name'].append(name)
            data_dic['Avg 2-Way 24hr Counts'].append(avg)
            data_dic['geometry'].append(map_name_to_geo[name])

        map_year_to_data[year] = data_dic

    # calc min and max values for scaling
    values = []
    for _, data_dic in map_year_to_data.items():
        values.extend(data_dic['Avg 2-Way 24hr Counts'])
    values = np.array(values)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    # scale values in data, we do this so that heat map colors are comparable across years
    for _, data_dic in map_year_to_data.items():
        values = np.array(data_dic['Avg 2-Way 24hr Counts'])
        values = (values - min_value) / (max_value - min_value)
        data_dic['Avg 2-Way 24hr Counts'] = list(values)

    # create kepler maps
    for year, data_dic in map_year_to_data.items():
        data_df = gpd.GeoDataFrame(data=data_dic, crs='epsg:4326', geometry='geometry')

        # get config
        map_config = json.dumps(heatmap_24hr_config)
        map_config = map_config.replace('label_name', '%s_24HrCounts' % year)
        map_config = json.loads(map_config)

        # create map
        kepler_map = kp.KeplerGl(height=600, data={year + '_24HrCounts': data_df}, config=map_config)
        kepler_maps.append(kepler_map)

        # save heatmap as html
        kepler_map.save_to_html(file_name=export_dir + "%s_24HrCounts_heatmap.html" % year)

    return kepler_maps

if __name__ == '__main__':
    # set paths
    dropbox_dir = '/Users/edson/Dropbox/'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    streetline_dir = data_process_folder + "Raw/Demand/Flow_speed/Road section/"
    flow_section_dir = data_process_folder + "Auxiliary files/Network/Infrastructure/Detectors/"
    export_dir = data_process_folder + "Temporary exports to be copied to processed data/Network/Infrastructure/"
    heatmap_24hr_counts(streetline_dir, flow_section_dir, export_dir)