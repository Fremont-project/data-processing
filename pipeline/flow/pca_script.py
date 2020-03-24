import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
import geopandas as gpd
import keplergl as kp

def run_pca(input_folder, desired_variance_explained, streetline_folder):
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
    desired_variance_explained = 96
    flow_df_raw = pd.read_csv(input_folder + '/flow_processed_section.csv')
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
    print('cumulative variance explained by PCs\n', cum_variances)

    # project data on PCs and plot them
    num_pcs = next(i + 1 for i, cum_var in enumerate(cum_variances) if cum_var >= desired_variance_explained)

    # year row labels
    flow_size = 24 * 4 * 3
    years = [2013, 2017, 2019]  # forget about 2015
    # year_row_labels = np.array([np.ones((flow_size)) * year for year in years]).flatten()

    # plot projections of timesteps on the pcs (pc_projection vs. pc_projection)
    pcs = [pca_model.components_[i] for i in range(num_pcs)]
    projections_on_pcs = [np.dot(data, pc) for pc in pcs]

    fig, axes = plt.subplots(nrows=math.ceil(num_pcs / 4), ncols=2, figsize=(15, 15))
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

    # # create kepler map to visualize pcs weights on their corresponding road sections
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

    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    kepler_map.save_to_html(
        file_name=data_process_folder + "Temporary exports to be copied to processed data/Network/Infrastructure/pca_heatmap.html")

    # heat map for flow data per year at the 4pm timestep
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
            flow_4pm = road_flow_days[0, 16]  # 4pm flow at day 1

            # to be use to remove elements that are empty
            if np.isnan(flow_4pm):
                year_remove_idx[year].append(r)

            year_flows_4pm[year].append(flow_4pm)

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
        year_road_names = [road for i, road in enumerate(road_names) if i not in year_remove_idx[year]]
        flows_4pm = [flow for i, flow, in enumerate(flows_4pm) if i not in year_remove_idx[year]]
        multi_index = pd.MultiIndex.from_tuples(list(year_road_names), names=['Name', 'Direction'])
        flows_4pm_df = gpd.GeoDataFrame(flows_4pm, crs='epsg:4326', index=multi_index, columns=['flow_4pm'])
        merged_df = flows_4pm_df.join(streetline_df)
        merged_df = merged_df.reset_index()
        kepler_map = kp.KeplerGl(height=600)
        kepler_map.add_data(data=merged_df, name=year + '_flow_4pm')
        dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
        data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
        kepler_map.save_to_html(
            file_name=data_process_folder +
                      "Temporary exports to be copied to processed data/Network/Infrastructure/%s_flow_4pm_heatmap.html" % year)


    return None, None, None
    # return pcs, projections_on_pcs, map


def raise_exception():
    # easy method to stop code at some desired location
    raise(Exception('stop code here'))

def main():
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    input_folder = data_process_folder + "Auxiliary files/Network/Infrastructure/Detectors/"
    streetline_folder = data_process_folder + "Raw/Demand/Flow_speed/Road section/"
    run_pca(input_folder, 96, streetline_folder)

    pass

if __name__ == '__main__':
    main()