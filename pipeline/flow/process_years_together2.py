import pandas as pd
import numpy as np


def create_flow_processed_section(line_to_detectors_dir, flow_speed_dir, output_dir):
    """
    Creates flow_processed_section.csv file where one row is a road section and 
    column data is flow for one specific day - year - 15 minute timestep 

    :param line_to_detectors_dir: directory for line_to_detectors.csv file  
    :param flow_speed_dir: parent directory for flow_processed_source.csv files where source=(pems, city)

    :return: writes flow_processed_section.csv as described above
    """
    print('\nCreating flow_processed_section.csv')
    section_df = pd.read_csv(line_to_detectors_dir + 'lines_to_detectors.csv')
    flow_processed_city_df = pd.read_csv(flow_speed_dir + 'Flow_processed/flow_processed_city.csv')
    flow_processed_pems_df = pd.read_csv(flow_speed_dir + 'PeMS/flow_processed_pems.csv')
    output = open(output_dir + 'flow_processed_section.csv', 'w')

    # Format pems to 15min timesteps instead of 5min
    # create new 15min timestep flow legend for pems
    flow_legend = []
    num_days, num_hours, num_timesteps = 3, 24, 4
    expected_year_flow_data_size = num_days * num_hours * num_timesteps
    for day in range(num_days):
        for hour in range(num_hours):
            for timestep in range(0, 60, 15):
                flow_legend.append('Day %s - %s:%s' % (str(day + 1), str(hour), str(timestep)))

    # get start index of flow
    flow_start_index = list(flow_processed_pems_df.columns).index('Day 1 - 0:0')
    # create rows for new pems
    pems_flow_rows = []
    for _, row in flow_processed_pems_df.iterrows():
        # parse pems flow data and get columns needed
        flow_data = row.to_numpy()[flow_start_index:]
        # format pems to 15min timesteps instead of 5min
        flow_data = flow_data.reshape((expected_year_flow_data_size, 3))
        flow_data = np.nansum(flow_data, axis=1)
        row_series_dict = {'Day 1': row['Day 1'], 'Id': str(row['Id']),
                           'Year': row['Year'], 'Name': row['Name']}
        row_series_dict.update(dict(zip(flow_legend, flow_data)))
        pems_flow_rows.append(pd.Series(row_series_dict)) # much faster to create Series from dict in one go

    # create new dataframe from it to parse it with same code as flow city data
    flow_processed_pems_df = pd.DataFrame(pems_flow_rows)

    # create legend for csv output file and write it to csv
    years = ['2013', '2015', '2017', '2019']
    flow_legend = []
    for year in years:
        for day in range(num_days):
            for hour in range(num_hours):
                for timestep in range(0, 60, 15):
                    flow_legend.append('Day %s - %s - %s:%s' % (str(day + 1), str(year), str(hour), str(timestep)))
    legend = ['OBJECTID', 'Name', 'Direction', ]
    legend += ['Day 1 ' + year for year in years]
    legend += flow_legend
    output.write(','.join(legend) + '\n')

    # Code is much cleaner if we processed city flow and pems flow separately, this is due to
    # when querying for flow data using a city detector id it returns one year worth of flow
    # but when querying for flow data using pems detector id it returns 4 years worth of flow at once
    # Parse city flow first iterating through the roads
    for _, road_section in section_df.iterrows():
        if 'pems' in road_section['Name'].lower():
            continue # skip pems

        all_years_data = []
        for year in years:
            detector_id = road_section[year]  # by user construction, detector id is in column 'year'

            # if detector was not assigned for this road section and year
            if not isinstance(detector_id, str) and np.isnan(detector_id):
                # add empty data for ease of writing to csv
                # expects one year of flow data and one start recording day
                one_year_detector_data = np.full((1, expected_year_flow_data_size + 1), np.NaN)
                all_years_data.append(one_year_detector_data)
                continue

            # multiple detectors can be assigned to one (road, year)
            # check if only one detector was assigned. If multiple detectors assigned then detector_id = 'id1 - id2'
            # where - is used as a delimeter for multiple detector ids
            detector_id = str(detector_id)
            if '-' not in detector_id:
                # one detector was assigned for this road section
                one_year_detector_data = parse_detector_year_data(int(float(detector_id)), flow_processed_city_df,
                                                              year, expected_year_flow_data_size)
                all_years_data.append(one_year_detector_data)
            else:
                # multiple detectors assigned for this road and year, take average
                detector_ids = [int(id.strip()) for id in detector_id.split('-')]

                # accumulate detector data in numpy matrix
                detector_data_matrix = []
                for id in detector_ids:
                    detector_id_data = parse_detector_year_data(id, flow_processed_city_df,
                                                                year, expected_year_flow_data_size)
                    detector_data_matrix.append(detector_id_data.flatten())
                detector_data_matrix = np.array(detector_data_matrix)

                # get all recording dates (in first column of matrix)
                recording_dates = ' - '.join(str(date) for date in detector_data_matrix[:, 0])
                # average flow data row wise (2nd col and beyond of matrix)
                mean_flow = np.mean(detector_data_matrix[:, 1:], axis=0)

                # combine start date and flow data and add to all years list
                one_year_detector_data = np.concatenate((recording_dates, mean_flow), axis=None)
                one_year_detector_data = one_year_detector_data.reshape((1, one_year_detector_data.shape[0]))
                all_years_data.append(one_year_detector_data)

        # write all_years_data for this road section to csv output
        # get start day of recordings and flow data
        # (year_data is a (1, num) array where first element is recording date and rest are flow for that date/year)
        recording_dates = [year_data[0, 0] for year_data in all_years_data]
        flow_data = [year_data[0, 1:] for year_data in all_years_data]
        flow_data = list(np.array(flow_data).flatten())

        # write them to csv
        csv_line = [str(road_section['OBJECTID']), road_section['Name'], road_section['Direction']]
        csv_line += recording_dates + flow_data
        csv_line = ','.join([str(x) for x in csv_line])
        output.write(csv_line + '\n')

    # now parse pems flow
    for _, road_section in section_df.iterrows():
        if 'pems' not in road_section['Name'].lower():
            continue # skip non pems

        detector_id = road_section['pems']

        all_years_data_lst = []
        # for pems, a query will return flow data for all years at once
        all_years_data = flow_processed_pems_df[int(detector_id) == flow_processed_pems_df['Id'].astype(int)]
        for year in years:
            one_year_detector_data = all_years_data[int(year) == all_years_data['Year']]
            if one_year_detector_data.empty:
                # place holder of nans for flow data for this year
                one_year_detector_data = np.full((expected_year_flow_data_size + 1), np.NaN)
                all_years_data_lst.append(one_year_detector_data)
            else:
                # get day1 and flow data for this year
                day1 = one_year_detector_data.iloc[0]['Day 1']
                flow_start_index = list(one_year_detector_data.columns).index('Day 1 - 0:0')
                one_year_detector_data = one_year_detector_data.to_numpy()[0, flow_start_index:]
                one_year_detector_data = np.concatenate((day1, one_year_detector_data), axis=None)
                all_years_data_lst.append(one_year_detector_data)

        # get all recording dates and all flow data in a list each
        recording_dates = [year_data[0] for year_data in all_years_data_lst]
        flow_data = [year_data[1:] for year_data in all_years_data_lst]
        flow_data = list(np.array(flow_data).flatten())

        # write them to csv
        csv_line = [str(road_section['OBJECTID']), road_section['Name'], road_section['Direction']]
        csv_line += recording_dates + flow_data
        csv_line = ','.join([str(x) for x in csv_line])
        output.write(csv_line + '\n')

    output.close()


def parse_detector_year_data(detector_id, flow_df, year, expected_year_flow_data_size):
    detector_year_data = flow_df[int(detector_id) == flow_df['Id'].astype(int)]

    # check if flow exists
    if detector_year_data.empty:
        # create empty data (NaNs) for ease of writing to csv
        # expects flow data and one start recording date for each year
        detector_year_data = np.full((1, expected_year_flow_data_size + 1), np.NaN)
    else:
        # day of recording and data flow start at 4th column (0 index)
        # get day 1
        day1 = detector_year_data['Day 1']
        # get detector year data (flow data)
        data_flow_start_idx = [i for i, col in enumerate(detector_year_data.columns) if col == 'Day 1 - 0:0']
        if not data_flow_start_idx:
            raise(Exception('Cant find data flow start index for detector id %s' % detector_id))
        data_flow_start_idx = data_flow_start_idx[0]
        detector_year_data = detector_year_data.to_numpy()[:, data_flow_start_idx:]

        # create day 1 as a column to concatenate, column wise, to flow data for the year
        day1 = np.array(day1).reshape((day1.shape[0], 1))
        detector_year_data = np.concatenate((day1, detector_year_data), axis=1)

        # append the data size if less than size of one year flow data
        flow_data_size = detector_year_data.shape[1] - 1 # subtract recording date
        if flow_data_size < expected_year_flow_data_size:
            padding = np.full((1, expected_year_flow_data_size - flow_data_size), np.NaN)
            detector_year_data = np.concatenate((detector_year_data, padding), axis=1)
            # Note to self: this actually doesn't happen, with current data this code is never reached
        elif flow_data_size > expected_year_flow_data_size:
            raise (ValueError('Warning: expected one year flow data size of %s '
                              'but found a size of %s instead for year %s and '
                              'detector id %s'
                              % (expected_year_flow_data_size, flow_data_size, year, str(detector_id))))

    return detector_year_data


def raise_exception():
    # easy method to stop code at some desired location
    raise(Exception('stop code here'))

def main():
    # line_to_detectors_dir = 'test_output/'
    # flow_processed_all_dir = ''
    # output_dir = 'test_output/'
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    output_folder = data_process_folder + "Temporary exports to be copied to processed data/Network/Infrastructure/Detectors/"
    flow_speed_dir = data_process_folder + "Auxiliary files/Demand/Flow_speed/"
    create_flow_processed_section(output_folder, flow_speed_dir, output_folder)
    pass

if __name__ == '__main__':
    main()