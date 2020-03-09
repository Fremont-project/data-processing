import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def run_pca(input_dir, desired_variance_explained):
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
    city_flow_df = pd.read_csv(input_dir + '/flow_processed_section.csv')

    city_flow_df = city_flow_df.set_index('Name')
    city_flow_df = city_flow_df.drop(['OBJECTID', 'Direction', 'Day 1 2013',
                                      'Day 1 2015', 'Day 1 2017', 'Day 1 2019'], axis=1)

    print('number of road sections', city_flow_df.shape[0])
    print('number of road sections lost from nan drop', city_flow_df.shape[0] - city_flow_df.dropna().shape[0])

    # transpose and drop na
    city_flow_df = city_flow_df.dropna()
    city_flow_df = city_flow_df.transpose()

    # normalize data: first make values range from 0 to 1 and then center it
    data = city_flow_df.to_numpy()
    data = data / np.nanmax(data, axis=0)
    data = data - np.nanmean(data, axis=0)
    data_columns = list(city_flow_df.columns)

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
    years = [2013, 2015, 2017, 2019]
    # year_row_labels = np.array([np.ones((flow_size)) * year for year in years]).flatten()

    # plot projections of timesteps on the pcs
    pcs = [pca_model.components_[i] for i in range(num_pcs)]
    projections_on_pcs = [np.dot(data, pc) for pc in pcs]

    fig, axes = plt.subplots(nrows=num_pcs // 4, ncols=2, figsize=(10, 10))
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

    return pcs, projections_on_pcs

def main():
    # data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    # data_folder = data_process_folder + "Temporary exports to be copied to processed data/Network/Infrastructure/Detectors/"
    pass

if __name__ == '__main__':
    main()