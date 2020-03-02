import pandas as pd
import numpy as np

def create_flow_processed_section(line_to_detectors_dir, flow_processed_all_dir, output_dir):
    """
    Creates flow_processed_section.csv file where one row is a road section and 
    column data is flow for one specific day - year - 15 minute timestep 
    
    :param line_to_detectors_dir: directory for line_to_detectors.csv file  
    :param flow_processed_all_dir: directory for flow_processed_all.csv file
    
    :return: writes flow_processed_section.csv as described above
    """
    section_df = pd.read_csv(line_to_detectors_dir + 'lines_to_detectors.csv')
    flow_df = pd.read_csv(flow_processed_all_dir + 'Flow_processed_all.csv')
    output = open(output_dir + 'flow_processed_section.csv', 'w')

    section_legend = section_df.columns.to_numpy()
    # get columns names: OBJECTID, Name, Direction
    years = ['2013', '2015', '2017', '2019', 'pems']
    legend = [section_legend[0], section_legend[2], section_legend[3]]
    legend += ['Day 1 ' + year for year in years]
    legend = ','.join(legend)

    # create more column names (Day # - year - timestep)
    num_days, num_hours, num_timesteps = 3, 24, 4
    for year in years:
        for day in range(num_days):
            for hour in range(num_hours):
                for timestep in range(0, 60, 15):
                    legend += ',Day %s - %s - %s:%s' % (str(day + 1), str(year), str(hour), str(timestep))

    expected_year_flow_data_size = num_days * num_hours * num_timesteps
    # print('flow data size one year', expected_year_flow_data_size)

    output.write(legend)

    for _, road_section in section_df.iterrows():
        all_years_data = []
        for year in years:
            detector_id = road_section[year] # by user construction, detector id is in column year

            # check if detector was assigned for this road section and year (type(NaN) = float)
            if not isinstance(detector_id, str) and np.isnan(detector_id):
                # detector not found for this road section and year
                # add empty data for ease of writing to csv
                # expects one year of flow data and one start recording day
                detector_year_data = np.full((expected_year_flow_data_size + 1), np.NaN)
                all_years_data.append(detector_year_data)
                continue

            # multiple detectors can be assigned to one road in one year
            # check if only one detector was assigned. If multiple detectors then detector_id = 'id1 - id2'
            if '-' not in detector_id: # if there's only one detector for this road section
                detector_year_data = parse_detector_year_data(int(detector_id), flow_df, year,
                                                              expected_year_flow_data_size)
                all_years_data.append(detector_year_data)
            else:
                # multiple detectors assigned for this road and year, take average
                detector_ids = [int(id.strip()) for id in detector_id.split('-')]
                # accumulate detector data
                detector_data_matrix = []
                for id in detector_ids:
                    detector_id_data = parse_detector_year_data(id, flow_df, year, expected_year_flow_data_size)
                    detector_data_matrix.append(detector_id_data)
                detector_data_matrix = np.array(detector_data_matrix)

                # get recording date
                recording_dates = ' - '.join(str(date) for date in detector_data_matrix[:, 0])
                # average flow data
                mean_flow = np.mean(detector_data_matrix[:, 1:], axis=0)

                detector_year_data = np.concatenate((recording_dates, mean_flow), axis=None)
                all_years_data.append(detector_year_data)


        # write data for this road section into csv output
        # get start day of recordings and flow data
        recording_dates = [year_data[0] for year_data in all_years_data]
        flow_data = [year_data[1:] for year_data in all_years_data]
        flow_data = list(np.array(flow_data).flatten())

        csv_line = [str(road_section['OBJECTID']), road_section['Name'], road_section['Direction']]
        csv_line += recording_dates + flow_data
        csv_line = ','.join([str(x) for x in csv_line])
        output.write('\n')
        output.write(csv_line)

    output.close()

def parse_detector_year_data(detector_id, flow_df, year, expected_year_flow_data_size):
    detector_year_data = flow_df[int(detector_id) == flow_df['Id']]

    # check if flow exists
    if detector_year_data.empty:
        # create empty data (NaNs) for ease of writing to csv
        # expects flow data and one start recording date for each year
        detector_year_data = np.full((expected_year_flow_data_size + 1), np.NaN)

    else:
        # day of recording and data flow start at 4th column (0 index)
        detector_year_data = detector_year_data.to_numpy()[0, 4:]

        # append the data size if less than size of one year flow data
        flow_data_size = detector_year_data.shape[0] - 1  # subtract recording date
        if flow_data_size < expected_year_flow_data_size:
            padding = np.full((expected_year_flow_data_size - flow_data_size), np.NaN)
            detector_year_data = np.concatenate((detector_year_data, padding), axis=None)
            # Note this actually doesn't happen, with current data this code is never reached
        elif flow_data_size > expected_year_flow_data_size:
            raise (ValueError('Warning: expected one year flow data size of %s '
                              'but found a size of %s instead for year %s and '
                              'detector id %s in Flow_processed_all.csv'
                              % (expected_year_flow_data_size, flow_data_size, year, str(detector_id))))

    return detector_year_data


def main():
    line_to_detectors_dir = 'test_output/'
    flow_processed_all_dir = ''
    output_dir = 'test_output/'
    create_flow_processed_section(line_to_detectors_dir, flow_processed_all_dir, output_dir)
    pass


if __name__ == '__main__':
    main()