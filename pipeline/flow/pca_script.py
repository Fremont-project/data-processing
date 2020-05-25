import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
import geopandas as gpd
import keplergl as kp
import json
import warnings

def run_pca(dropbox_dir):
    """
       Runs pca on 'flow_processed_section.csv' located in input_dir folder. Goal of pca is to
       find principal components (PCs) that explain the variance in flow between years.
       Data in flow_processed_section.csv is a matrix where rows are road sections and columns are timesteps.
       We transpose the matrix and apply pca, hence observations are timesteps and features are road sections.

       :param input_dir: folder that contains flow_processed_section.csv
       :param desired_variance_explained: threshold variance to obtain number of pcs that explain the variance
           in the data.
       :return: plots of analysis, PCs, and projections of observations on PCs
       """

    # surpress runtime warnings
    warnings.filterwarnings('ignore')

    # preset heatmap configurations
    heatmap_pca_config, heatmap_4pmconfig, heatmap_24hr_config = get_keplermap_configs()

    # folders
    data_process_folder = dropbox_dir + "Private Structured data collection/Data processing/"
    detector_folder = data_process_folder + "Auxiliary files/Network/Infrastructure/Detectors/"
    streetline_folder = data_process_folder + "Raw/Demand/Flow_speed/Road section/"
    pca_folder = data_process_folder + "Kepler maps/Demand/Flow_speed/PCA/"

    # read data from flow_processed_section.csv
    desired_variance_explained = 96
    flow_df_raw = pd.read_csv(detector_folder + '/flow_processed_section.csv')
    flow_df_raw = flow_df_raw.set_index(['Name', 'Direction'])
    flow_df_raw = flow_df_raw.drop(['OBJECTID', 'Day 1 2013',
                                    'Day 1 2015', 'Day 1 2017', 'Day 1 2019'], axis=1)

    # remove 2015 data also
    cols_2015 = [col for col in flow_df_raw.columns if '2015' in col]
    flow_df = flow_df_raw.drop(cols_2015, axis=1)

    # check how many roads sections we lost
    print('number of road sections', flow_df.shape[0])
    print('number of road sections after nan drop', flow_df.dropna().shape[0])

    # transpose and drop na
    flow_df = flow_df.dropna()
    flow_df = flow_df.transpose()

    # normalize data: first make values range from 0 to 1 and then center it
    data = flow_df.to_numpy()
    data = data / np.nanmax(data, axis=0)
    data = data - np.nanmean(data, axis=0)
    feature_col_names = list(flow_df.columns)  # are road section names

    # run pca and calc explained variances
    pca_model = PCA()
    pca_model.fit(data)
    variances = pca_model.explained_variance_ratio_
    cum_variances = np.cumsum(np.round(variances, decimals=3) * 100)

    plt.plot(cum_variances)
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    print('individual variance explained by PCs\n', np.round(variances, decimals=3) * 100)
    print('cumulative variance explained by PCs\n', cum_variances)

    # project data on PCs and plot them
    num_pcs = next(i + 1 for i, cum_var in enumerate(cum_variances) if cum_var >= desired_variance_explained)

    # year row labels
    flow_size = 24 * 4 * 3
    years = [2013, 2017, 2019]  # forget about 2015

    # plot projections of timesteps on the pcs (pc_projection vs. pc_projection)
    pcs = [pca_model.components_[i] for i in range(num_pcs)]
    projections_on_pcs = [np.dot(data, pc) for pc in pcs]

    fig, axes = plt.subplots(nrows=math.ceil(num_pcs / 4) - 1, ncols=2, figsize=(15, 15))
    axes = axes.flatten()
    for ax_i, i in enumerate(range(1, num_pcs, 2)):
        for yi, year in enumerate(years):
            axes[ax_i].set_title('Projections along PC%d and PC%d' % (i, i + 1))
            axes[ax_i].scatter(projections_on_pcs[i - 1][yi * flow_size: (yi + 1) * flow_size],
                               projections_on_pcs[i][yi * flow_size: (yi + 1) * flow_size],
                               label=str(year))
            axes[ax_i].set_xlabel('PC%d Projections' % i)
            axes[ax_i].set_ylabel('PC%d Projections' % (i + 1))
            axes[ax_i].set_xlim([-2, 2])
            axes[ax_i].set_ylim([-2, 2])
            axes[ax_i].legend()

    # plot pc_projections by itself
    fig = plt.figure()
    for i, pc_proj in enumerate(projections_on_pcs):
        plt.plot(pc_proj)
        plt.ylabel('Projection on PC' + str(i + 1))
        plt.xlabel('# of Features')
        plt.show()

    ###################################################################################
    # create kepler map to visualize pcs weights on their corresponding road sections
    ###################################################################################

    streetline_df = gpd.GeoDataFrame.from_file(streetline_folder + 'Streetline.shp')
    streetline_df = streetline_df[['OBJECTID', 'Name', 'Direction', 'geometry']]
    streetline_df = streetline_df.set_geometry('geometry')
    streetline_df = streetline_df.to_crs(epsg=4326)
    streetline_df = streetline_df.set_index(['Name', 'Direction'])

    all_pcs = []
    for pc in pcs:
        all_pcs.extend(pc)
    # normalize (scale) pcs to go from [0, 1]
    min_val = min(all_pcs)
    max_val = max(all_pcs)
    scaled_pcs = (np.array(all_pcs) - min_val) / (max_val - min_val)
    scaled_pcs = np.reshape(scaled_pcs, (num_pcs, len(scaled_pcs) // num_pcs))
    scaled_pcs = [list(scaled_pcs[i, :]) for i in range(num_pcs)]

    kepler_map = kp.KeplerGl(height=600)
    for i, pc in enumerate(scaled_pcs):
        multi_index = pd.MultiIndex.from_tuples(feature_col_names, names=['Name', 'Direction'])
        pc_df = gpd.GeoDataFrame(pc, crs='epsg:4326', index=multi_index, columns=['PC' + str(i + 1)])
        merged_pc_df = pc_df.join(streetline_df)
        merged_pc_df = merged_pc_df.reset_index()
        kepler_map.add_data(data=merged_pc_df, name='PC' + str(i + 1))

    kepler_map.config = heatmap_pca_config
    kepler_map.save_to_html(file_name=pca_folder + "pca_heatmap.html")

    ###################################################################################
    # heat map for flow data per year at the 4pm timestep
    ###################################################################################

    # reload the data bc 2015 was filtered
    flow_df = flow_df_raw.transpose()
    road_names = flow_df.columns
    num_years = 4
    num_days = 3
    day_size = 24 * 4
    year_size = day_size * num_days
    data = flow_df.to_numpy()
    num_roads = data.shape[1]

    # get flow data at 4pm
    years = ['2013', '2015', '2017', '2019']
    year_flows_4pm = {}
    year_remove_idx = {}
    for r in range(num_roads):
        road_flow = data[:, r]
        road_flow = np.reshape(road_flow, (num_years, year_size))
        for y, year in enumerate(years):
            if year not in year_flows_4pm:
                year_flows_4pm[year] = []
            if year not in year_remove_idx:
                year_remove_idx[year] = []

            road_flow_days = np.reshape(road_flow[y, :], (num_days, day_size))
            # index of time 4pm, 4*4
            flow_avg_4pm = np.nanmean(road_flow_days[:, 16])

            # to be use to remove elements that are empty
            if np.isnan(flow_avg_4pm):
                year_remove_idx[year].append(r)

            year_flows_4pm[year].append(flow_avg_4pm)

    # scale flow data to range 0, 1
    all_flow = []
    for _, vals in year_flows_4pm.items():
        all_flow.extend(vals)
    min_val = min(all_flow)
    max_val = max(all_flow)
    scaled_flow = (np.array(all_flow) - min_val) / (max_val - min_val)
    scaled_flow = np.reshape(scaled_flow, (4, num_roads))

    for y, year in enumerate(years):
        year_flows_4pm[year] = list(scaled_flow[y, :])

    for year, flows_4pm in year_flows_4pm.items():
        # load data
        year_road_names = [road for i, road in enumerate(road_names) if i not in year_remove_idx[year]]
        flows_4pm = [flow for i, flow, in enumerate(flows_4pm) if i not in year_remove_idx[year]]
        multi_index = pd.MultiIndex.from_tuples(list(year_road_names), names=['Name', 'Direction'])
        flows_4pm_df = gpd.GeoDataFrame(flows_4pm, crs='epsg:4326', index=multi_index, columns=['flow_4pm'])
        merged_df = flows_4pm_df.join(streetline_df)
        merged_df = merged_df.reset_index()

        # load config
        config = json.dumps(heatmap_4pmconfig)
        config = config.replace('label_name', '%s_flow_4pm' % year)
        config = json.loads(config)

        # create kepler map
        kepler_map = kp.KeplerGl(height=600, data={year + '_flow_4pm': merged_df}, config=config)
        kepler_map.save_to_html(file_name=pca_folder + ("%s_flow_4pm_heatmap.html" % year))

    ###################################################################################
    # Heatmaps for two-way 24hr counts per year
    ###################################################################################
    # read data flow data for all 3 days
    flow_df = pd.read_csv(detector_folder + '/flow_processed_section.csv')
    flow_df = flow_df.set_index(['Name', 'Direction'])

    # keep only data timesteps of day 1
    years = ['2013', '2015', '2017', '2019']
    map_year_to_data = {}
    kepler_maps = []
    for year in years:
        # get only time step data for given year
        timestep_name = 'Day 1 - %s' % year
        timestep_cols = [col for col in flow_df.columns if timestep_name in col]
        year_flow_df = flow_df[timestep_cols]

        # sum over column timesteps
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
        kepler_map.save_to_html(file_name=pca_folder + "%s_24HrCounts_heatmap.html" % year)


def get_keplermap_configs():
    heatmap_pca_config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [
        {'id': 'emx2z1b', 'type': 'geojson',
         'config': {'dataId': 'PC1', 'label': 'PC1', 'color': [241, 92, 23], 'columns': {'geojson': 'geometry'},
                    'isVisible': True, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                                     'colorRange': {'name': 'Global Warming', 'type': 'sequential',
                                                                    'category': 'Uber',
                                                                    'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                               '#E3611C', '#F1920E', '#FFC300']},
                                                     'strokeColorRange': {'name': 'ColorBrewer Reds-6',
                                                                          'type': 'singlehue',
                                                                          'category': 'ColorBrewer',
                                                                          'colors': ['#fee5d9', '#fcbba1', '#fc9272',
                                                                                     '#fb6a4a', '#de2d26', '#a50f15']},
                                                     'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50],
                                                     'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                                     'filled': False, 'enable3d': False, 'wireframe': False},
                    'textLabel': [
                        {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                         'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'PC1', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}, {'id': '2jx4mo', 'type': 'geojson',
                                                        'config': {'dataId': 'PC2', 'label': 'PC2',
                                                                   'color': [34, 63, 154],
                                                                   'columns': {'geojson': 'geometry'},
                                                                   'isVisible': False,
                                                                   'visConfig': {'opacity': 0.8, 'thickness': 0.5,
                                                                                 'strokeColor': None, 'colorRange': {
                                                                           'name': 'Global Warming',
                                                                           'type': 'sequential', 'category': 'Uber',
                                                                           'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                      '#E3611C', '#F1920E', '#FFC300']},
                                                                                 'strokeColorRange': {
                                                                                     'name': 'ColorBrewer Reds-6',
                                                                                     'type': 'singlehue',
                                                                                     'category': 'ColorBrewer',
                                                                                     'colors': ['#fee5d9', '#fcbba1',
                                                                                                '#fc9272', '#fb6a4a',
                                                                                                '#de2d26', '#a50f15']},
                                                                                 'radius': 10, 'sizeRange': [0, 10],
                                                                                 'radiusRange': [0, 50],
                                                                                 'heightRange': [0, 500],
                                                                                 'elevationScale': 5, 'stroked': True,
                                                                                 'filled': False, 'enable3d': False,
                                                                                 'wireframe': False}, 'textLabel': [
                                                                {'field': None, 'color': [255, 255, 255], 'size': 18,
                                                                 'offset': [0, 0], 'anchor': 'start',
                                                                 'alignment': 'center'}]},
                                                        'visualChannels': {'colorField': None, 'colorScale': 'quantile',
                                                                           'sizeField': None, 'sizeScale': 'linear',
                                                                           'strokeColorField': {'name': 'PC2',
                                                                                                'type': 'real'},
                                                                           'strokeColorScale': 'quantile',
                                                                           'heightField': None, 'heightScale': 'linear',
                                                                           'radiusField': None,
                                                                           'radiusScale': 'linear'}},
        {'id': '33d09gn', 'type': 'geojson',
         'config': {'dataId': 'PC3', 'label': 'PC3', 'color': [218, 112, 191], 'columns': {'geojson': 'geometry'},
                    'isVisible': False, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                                      'colorRange': {'name': 'Global Warming', 'type': 'sequential',
                                                                     'category': 'Uber',
                                                                     'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                '#E3611C', '#F1920E', '#FFC300']},
                                                      'strokeColorRange': {'name': 'ColorBrewer Reds-6',
                                                                           'type': 'singlehue',
                                                                           'category': 'ColorBrewer',
                                                                           'colors': ['#fee5d9', '#fcbba1', '#fc9272',
                                                                                      '#fb6a4a', '#de2d26', '#a50f15']},
                                                      'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50],
                                                      'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                                      'filled': False, 'enable3d': False, 'wireframe': False},
                    'textLabel': [
                        {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                         'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'PC3', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}, {'id': 'v16b0b', 'type': 'geojson',
                                                        'config': {'dataId': 'PC4', 'label': 'PC4',
                                                                   'color': [18, 92, 119],
                                                                   'columns': {'geojson': 'geometry'},
                                                                   'isVisible': False,
                                                                   'visConfig': {'opacity': 0.8, 'thickness': 0.5,
                                                                                 'strokeColor': None, 'colorRange': {
                                                                           'name': 'Global Warming',
                                                                           'type': 'sequential', 'category': 'Uber',
                                                                           'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                      '#E3611C', '#F1920E', '#FFC300']},
                                                                                 'strokeColorRange': {
                                                                                     'name': 'ColorBrewer Reds-6',
                                                                                     'type': 'singlehue',
                                                                                     'category': 'ColorBrewer',
                                                                                     'colors': ['#fee5d9', '#fcbba1',
                                                                                                '#fc9272', '#fb6a4a',
                                                                                                '#de2d26', '#a50f15']},
                                                                                 'radius': 10, 'sizeRange': [0, 10],
                                                                                 'radiusRange': [0, 50],
                                                                                 'heightRange': [0, 500],
                                                                                 'elevationScale': 5, 'stroked': True,
                                                                                 'filled': False, 'enable3d': False,
                                                                                 'wireframe': False}, 'textLabel': [
                                                                {'field': None, 'color': [255, 255, 255], 'size': 18,
                                                                 'offset': [0, 0], 'anchor': 'start',
                                                                 'alignment': 'center'}]},
                                                        'visualChannels': {'colorField': None, 'colorScale': 'quantile',
                                                                           'sizeField': None, 'sizeScale': 'linear',
                                                                           'strokeColorField': {'name': 'PC4',
                                                                                                'type': 'real'},
                                                                           'strokeColorScale': 'quantile',
                                                                           'heightField': None, 'heightScale': 'linear',
                                                                           'radiusField': None,
                                                                           'radiusScale': 'linear'}},
        {'id': 'vtl23l5', 'type': 'geojson',
         'config': {'dataId': 'PC5', 'label': 'PC5', 'color': [77, 193, 156], 'columns': {'geojson': 'geometry'},
                    'isVisible': False, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                                      'colorRange': {'name': 'Global Warming', 'type': 'sequential',
                                                                     'category': 'Uber',
                                                                     'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                '#E3611C', '#F1920E', '#FFC300']},
                                                      'strokeColorRange': {'name': 'ColorBrewer Reds-6',
                                                                           'type': 'singlehue',
                                                                           'category': 'ColorBrewer',
                                                                           'colors': ['#fee5d9', '#fcbba1', '#fc9272',
                                                                                      '#fb6a4a', '#de2d26', '#a50f15']},
                                                      'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50],
                                                      'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                                      'filled': False, 'enable3d': False, 'wireframe': False},
                    'textLabel': [
                        {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                         'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'PC5', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}, {'id': '3dtg1d', 'type': 'geojson',
                                                        'config': {'dataId': 'PC6', 'label': 'PC6',
                                                                   'color': [119, 110, 87],
                                                                   'columns': {'geojson': 'geometry'},
                                                                   'isVisible': False,
                                                                   'visConfig': {'opacity': 0.8, 'thickness': 0.5,
                                                                                 'strokeColor': None, 'colorRange': {
                                                                           'name': 'Global Warming',
                                                                           'type': 'sequential', 'category': 'Uber',
                                                                           'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                      '#E3611C', '#F1920E', '#FFC300']},
                                                                                 'strokeColorRange': {
                                                                                     'name': 'ColorBrewer Reds-6',
                                                                                     'type': 'singlehue',
                                                                                     'category': 'ColorBrewer',
                                                                                     'colors': ['#fee5d9', '#fcbba1',
                                                                                                '#fc9272', '#fb6a4a',
                                                                                                '#de2d26', '#a50f15']},
                                                                                 'radius': 10, 'sizeRange': [0, 10],
                                                                                 'radiusRange': [0, 50],
                                                                                 'heightRange': [0, 500],
                                                                                 'elevationScale': 5, 'stroked': True,
                                                                                 'filled': False, 'enable3d': False,
                                                                                 'wireframe': False}, 'textLabel': [
                                                                {'field': None, 'color': [255, 255, 255], 'size': 18,
                                                                 'offset': [0, 0], 'anchor': 'start',
                                                                 'alignment': 'center'}]},
                                                        'visualChannels': {'colorField': None, 'colorScale': 'quantile',
                                                                           'sizeField': None, 'sizeScale': 'linear',
                                                                           'strokeColorField': {'name': 'PC6',
                                                                                                'type': 'real'},
                                                                           'strokeColorScale': 'quantile',
                                                                           'heightField': None, 'heightScale': 'linear',
                                                                           'radiusField': None,
                                                                           'radiusScale': 'linear'}},
        {'id': 'h392ken', 'type': 'geojson',
         'config': {'dataId': 'PC7', 'label': 'PC7', 'color': [23, 184, 190], 'columns': {'geojson': 'geometry'},
                    'isVisible': False, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                                      'colorRange': {'name': 'Global Warming', 'type': 'sequential',
                                                                     'category': 'Uber',
                                                                     'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                '#E3611C', '#F1920E', '#FFC300']},
                                                      'strokeColorRange': {'name': 'ColorBrewer Reds-6',
                                                                           'type': 'singlehue',
                                                                           'category': 'ColorBrewer',
                                                                           'colors': ['#fee5d9', '#fcbba1', '#fc9272',
                                                                                      '#fb6a4a', '#de2d26', '#a50f15']},
                                                      'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50],
                                                      'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                                      'filled': False, 'enable3d': False, 'wireframe': False},
                    'textLabel': [
                        {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                         'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'PC7', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}, {'id': 'q84y6ei', 'type': 'geojson',
                                                        'config': {'dataId': 'PC8', 'label': 'PC8',
                                                                   'color': [246, 209, 138],
                                                                   'columns': {'geojson': 'geometry'},
                                                                   'isVisible': False,
                                                                   'visConfig': {'opacity': 0.8, 'thickness': 0.5,
                                                                                 'strokeColor': None, 'colorRange': {
                                                                           'name': 'Global Warming',
                                                                           'type': 'sequential', 'category': 'Uber',
                                                                           'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                      '#E3611C', '#F1920E', '#FFC300']},
                                                                                 'strokeColorRange': {
                                                                                     'name': 'ColorBrewer Reds-6',
                                                                                     'type': 'singlehue',
                                                                                     'category': 'ColorBrewer',
                                                                                     'colors': ['#fee5d9', '#fcbba1',
                                                                                                '#fc9272', '#fb6a4a',
                                                                                                '#de2d26', '#a50f15']},
                                                                                 'radius': 10, 'sizeRange': [0, 10],
                                                                                 'radiusRange': [0, 50],
                                                                                 'heightRange': [0, 500],
                                                                                 'elevationScale': 5, 'stroked': True,
                                                                                 'filled': False, 'enable3d': False,
                                                                                 'wireframe': False}, 'textLabel': [
                                                                {'field': None, 'color': [255, 255, 255], 'size': 18,
                                                                 'offset': [0, 0], 'anchor': 'start',
                                                                 'alignment': 'center'}]},
                                                        'visualChannels': {'colorField': None, 'colorScale': 'quantile',
                                                                           'sizeField': None, 'sizeScale': 'linear',
                                                                           'strokeColorField': {'name': 'PC8',
                                                                                                'type': 'real'},
                                                                           'strokeColorScale': 'quantile',
                                                                           'heightField': None, 'heightScale': 'linear',
                                                                           'radiusField': None,
                                                                           'radiusScale': 'linear'}},
        {'id': 'u10wh87', 'type': 'geojson',
         'config': {'dataId': 'PC9', 'label': 'PC9', 'color': [183, 136, 94], 'columns': {'geojson': 'geometry'},
                    'isVisible': False, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                                      'colorRange': {'name': 'Global Warming', 'type': 'sequential',
                                                                     'category': 'Uber',
                                                                     'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                '#E3611C', '#F1920E', '#FFC300']},
                                                      'strokeColorRange': {'name': 'ColorBrewer Reds-6',
                                                                           'type': 'singlehue',
                                                                           'category': 'ColorBrewer',
                                                                           'colors': ['#fee5d9', '#fcbba1', '#fc9272',
                                                                                      '#fb6a4a', '#de2d26', '#a50f15']},
                                                      'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50],
                                                      'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                                      'filled': False, 'enable3d': False, 'wireframe': False},
                    'textLabel': [
                        {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                         'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'PC9', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}, {'id': '29iggvr', 'type': 'geojson',
                                                        'config': {'dataId': 'PC10', 'label': 'PC10',
                                                                   'color': [255, 203, 153],
                                                                   'columns': {'geojson': 'geometry'},
                                                                   'isVisible': False,
                                                                   'visConfig': {'opacity': 0.8, 'thickness': 0.5,
                                                                                 'strokeColor': None, 'colorRange': {
                                                                           'name': 'Global Warming',
                                                                           'type': 'sequential', 'category': 'Uber',
                                                                           'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                      '#E3611C', '#F1920E', '#FFC300']},
                                                                                 'strokeColorRange': {
                                                                                     'name': 'ColorBrewer Reds-6',
                                                                                     'type': 'singlehue',
                                                                                     'category': 'ColorBrewer',
                                                                                     'colors': ['#fee5d9', '#fcbba1',
                                                                                                '#fc9272', '#fb6a4a',
                                                                                                '#de2d26', '#a50f15']},
                                                                                 'radius': 10, 'sizeRange': [0, 10],
                                                                                 'radiusRange': [0, 50],
                                                                                 'heightRange': [0, 500],
                                                                                 'elevationScale': 5, 'stroked': True,
                                                                                 'filled': False, 'enable3d': False,
                                                                                 'wireframe': False}, 'textLabel': [
                                                                {'field': None, 'color': [255, 255, 255], 'size': 18,
                                                                 'offset': [0, 0], 'anchor': 'start',
                                                                 'alignment': 'center'}]},
                                                        'visualChannels': {'colorField': None, 'colorScale': 'quantile',
                                                                           'sizeField': None, 'sizeScale': 'linear',
                                                                           'strokeColorField': {'name': 'PC10',
                                                                                                'type': 'real'},
                                                                           'strokeColorScale': 'quantile',
                                                                           'heightField': None, 'heightScale': 'linear',
                                                                           'radiusField': None,
                                                                           'radiusScale': 'linear'}},
        {'id': 'qo1tyyae', 'type': 'geojson',
         'config': {'dataId': 'PC11', 'label': 'PC11', 'color': [248, 149, 112], 'columns': {'geojson': 'geometry'},
                    'isVisible': False, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                                      'colorRange': {'name': 'Global Warming', 'type': 'sequential',
                                                                     'category': 'Uber',
                                                                     'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                '#E3611C', '#F1920E', '#FFC300']},
                                                      'strokeColorRange': {'name': 'ColorBrewer Reds-6',
                                                                           'type': 'singlehue',
                                                                           'category': 'ColorBrewer',
                                                                           'colors': ['#fee5d9', '#fcbba1', '#fc9272',
                                                                                      '#fb6a4a', '#de2d26', '#a50f15']},
                                                      'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50],
                                                      'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                                      'filled': False, 'enable3d': False, 'wireframe': False},
                    'textLabel': [
                        {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                         'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'PC11', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}, {'id': 'axh9f04', 'type': 'geojson',
                                                        'config': {'dataId': 'PC12', 'label': 'PC12',
                                                                   'color': [130, 154, 227],
                                                                   'columns': {'geojson': 'geometry'},
                                                                   'isVisible': False,
                                                                   'visConfig': {'opacity': 0.8, 'thickness': 0.5,
                                                                                 'strokeColor': None, 'colorRange': {
                                                                           'name': 'Global Warming',
                                                                           'type': 'sequential', 'category': 'Uber',
                                                                           'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                      '#E3611C', '#F1920E', '#FFC300']},
                                                                                 'strokeColorRange': {
                                                                                     'name': 'ColorBrewer Reds-6',
                                                                                     'type': 'singlehue',
                                                                                     'category': 'ColorBrewer',
                                                                                     'colors': ['#fee5d9', '#fcbba1',
                                                                                                '#fc9272', '#fb6a4a',
                                                                                                '#de2d26', '#a50f15']},
                                                                                 'radius': 10, 'sizeRange': [0, 10],
                                                                                 'radiusRange': [0, 50],
                                                                                 'heightRange': [0, 500],
                                                                                 'elevationScale': 5, 'stroked': True,
                                                                                 'filled': False, 'enable3d': False,
                                                                                 'wireframe': False}, 'textLabel': [
                                                                {'field': None, 'color': [255, 255, 255], 'size': 18,
                                                                 'offset': [0, 0], 'anchor': 'start',
                                                                 'alignment': 'center'}]},
                                                        'visualChannels': {'colorField': None, 'colorScale': 'quantile',
                                                                           'sizeField': None, 'sizeScale': 'linear',
                                                                           'strokeColorField': {'name': 'PC12',
                                                                                                'type': 'real'},
                                                                           'strokeColorScale': 'quantile',
                                                                           'heightField': None, 'heightScale': 'linear',
                                                                           'radiusField': None,
                                                                           'radiusScale': 'linear'}},
        {'id': 'n9qv3lc', 'type': 'geojson',
         'config': {'dataId': 'PC13', 'label': 'PC13', 'color': [231, 159, 213], 'columns': {'geojson': 'geometry'},
                    'isVisible': False, 'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                                      'colorRange': {'name': 'Global Warming', 'type': 'sequential',
                                                                     'category': 'Uber',
                                                                     'colors': ['#5A1846', '#900C3F', '#C70039',
                                                                                '#E3611C', '#F1920E', '#FFC300']},
                                                      'strokeColorRange': {'name': 'ColorBrewer Reds-6',
                                                                           'type': 'singlehue',
                                                                           'category': 'ColorBrewer',
                                                                           'colors': ['#fee5d9', '#fcbba1', '#fc9272',
                                                                                      '#fb6a4a', '#de2d26', '#a50f15']},
                                                      'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50],
                                                      'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                                      'filled': False, 'enable3d': False, 'wireframe': False},
                    'textLabel': [
                        {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                         'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'PC13', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}], 'interactionConfig': {'tooltip': {
        'fieldsToShow': {'PC1': ['Name', 'Direction', 'PC1', 'OBJECTID'],
                         'PC2': ['Name', 'Direction', 'PC2', 'OBJECTID'],
                         'PC3': ['Name', 'Direction', 'PC3', 'OBJECTID'],
                         'PC4': ['Name', 'Direction', 'PC4', 'OBJECTID'],
                         'PC5': ['Name', 'Direction', 'PC5', 'OBJECTID'],
                         'PC6': ['Name', 'Direction', 'PC6', 'OBJECTID'],
                         'PC7': ['Name', 'Direction', 'PC7', 'OBJECTID'],
                         'PC8': ['Name', 'Direction', 'PC8', 'OBJECTID'],
                         'PC9': ['Name', 'Direction', 'PC9', 'OBJECTID'],
                         'PC10': ['Name', 'Direction', 'PC10', 'OBJECTID'],
                         'PC11': ['Name', 'Direction', 'PC11', 'OBJECTID'],
                         'PC12': ['Name', 'Direction', 'PC12', 'OBJECTID'],
                         'PC13': ['Name', 'Direction', 'PC13', 'OBJECTID']}, 'enabled': True},
        'brush': {'size': 0.5, 'enabled': False}},
                                                                   'layerBlending': 'normal', 'splitMaps': [],
                                                                   'animationConfig': {'currentTime': None,
                                                                                       'speed': 1}},
                                                      'mapState': {'bearing': 0, 'dragRotate': False,
                                                                   'latitude': 37.51807046403553,
                                                                   'longitude': -121.93088157608598, 'pitch': 0,
                                                                   'zoom': 11.92151140700923, 'isSplit': False},
                                                      'mapStyle': {'styleType': 'dark', 'topLayerGroups': {},
                                                                   'visibleLayerGroups': {'label': True, 'road': True,
                                                                                          'border': False,
                                                                                          'building': True,
                                                                                          'water': True, 'land': True,
                                                                                          '3d building': False},
                                                                   'threeDBuildingColor': [9.665468314072013,
                                                                                           17.18305478057247,
                                                                                           31.1442867897876],
                                                                   'mapStyles': {}}}}
    heatmap_4pmconfig = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [
        {'id': 'faoiatj', 'type': 'geojson',
         'config': {'dataId': 'label_name', 'label': 'label_name', 'color': [221, 178, 124],
                    'columns': {'geojson': 'geometry'}, 'isVisible': True,
                    'visConfig': {'opacity': 0.8, 'thickness': 0.5, 'strokeColor': None,
                                  'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber',
                                                 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E',
                                                            '#FFC300']},
                                  'strokeColorRange': {'name': 'ColorBrewer Reds-6', 'type': 'singlehue',
                                                       'category': 'ColorBrewer',
                                                       'colors': ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26',
                                                                  '#a50f15']}, 'radius': 10, 'sizeRange': [0, 10],
                                  'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                  'filled': False, 'enable3d': False, 'wireframe': False}, 'textLabel': [
                 {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                  'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'flow_4pm', 'type': 'real'}, 'strokeColorScale': 'quantile',
                            'heightField': None, 'heightScale': 'linear', 'radiusField': None,
                            'radiusScale': 'linear'}}], 'interactionConfig': {
        'tooltip': {'fieldsToShow': {'label_name': ['Name', 'Direction', 'flow_4pm', 'OBJECTID']}, 'enabled': True},
        'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': [],
                                                                  'animationConfig': {'currentTime': None, 'speed': 1}},
                                                     'mapState': {'bearing': 0, 'dragRotate': False,
                                                                  'latitude': 37.51824840850006,
                                                                  'longitude': -121.93325058399995, 'pitch': 0,
                                                                  'zoom': 12, 'isSplit': False},
                                                     'mapStyle': {'styleType': 'dark', 'topLayerGroups': {},
                                                                  'visibleLayerGroups': {'label': True, 'road': True,
                                                                                         'border': False,
                                                                                         'building': True,
                                                                                         'water': True, 'land': True,
                                                                                         '3d building': False},
                                                                  'threeDBuildingColor': [9.665468314072013,
                                                                                          17.18305478057247,
                                                                                          31.1442867897876],
                                                                  'mapStyles': {}}}}
    heatmap_24hr_config = {'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [
        {'id': 'wfpgbp', 'type': 'geojson',
         'config': {'dataId': 'label_name', 'label': 'label_name', 'color': [183, 136, 94],
                    'columns': {'geojson': 'geometry'}, 'isVisible': True,
                    'visConfig': {'opacity': 0.8, 'thickness': 2, 'strokeColor': None,
                                  'colorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber',
                                                 'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E',
                                                            '#FFC300']},
                                  'strokeColorRange': {'name': 'Custom Palette', 'type': 'custom', 'category': 'Custom',
                                                       'colors': ['#1EF013', '#E49309', '#1DC9DD', '#F364F3',
                                                                  '#4C0369']}, 'radius': 10, 'sizeRange': [0, 10],
                                  'radiusRange': [0, 50], 'heightRange': [0, 500], 'elevationScale': 5, 'stroked': True,
                                  'filled': False, 'enable3d': False, 'wireframe': False}, 'textLabel': [
                 {'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start',
                  'alignment': 'center'}]},
         'visualChannels': {'colorField': None, 'colorScale': 'quantile', 'sizeField': None, 'sizeScale': 'linear',
                            'strokeColorField': {'name': 'Avg 2-Way 24hr Counts', 'type': 'real'},
                            'strokeColorScale': 'quantile', 'heightField': None, 'heightScale': 'linear',
                            'radiusField': None, 'radiusScale': 'linear'}}], 'interactionConfig': {
        'tooltip': {'fieldsToShow': {'label_name': ['Name', 'Avg 2-Way 24hr Counts']}, 'enabled': True},
        'brush': {'size': 0.5, 'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': [],
                                                                    'animationConfig': {'currentTime': None,
                                                                                        'speed': 1}},
                                                       'mapState': {'bearing': 0, 'dragRotate': False,
                                                                    'latitude': 37.51747575092308,
                                                                    'longitude': -121.93332419092869, 'pitch': 0,
                                                                    'zoom': 12.223268867164116, 'isSplit': False},
                                                       'mapStyle': {'styleType': 'dark', 'topLayerGroups': {},
                                                                    'visibleLayerGroups': {'label': True, 'road': True,
                                                                                           'border': False,
                                                                                           'building': True,
                                                                                           'water': True, 'land': True,
                                                                                           '3d building': False},
                                                                    'threeDBuildingColor': [9.665468314072013,
                                                                                            17.18305478057247,
                                                                                            31.1442867897876],
                                                                    'mapStyles': {}}}}

    return heatmap_pca_config, heatmap_4pmconfig, heatmap_24hr_config


def raise_exception():
    # easy method to stop code at some desired location
    raise(Exception('stop code here'))


def main():
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    run_pca(dropbox_dir)


if __name__ == '__main__':
    main()