import pandas as pd
import os
import requests
import textract
import numpy as np
from pathlib import Path
import math
from datetime import datetime

API_KEY = "AIzaSyB8rJhDsfwvIod9jVTfFm1Dtv2eO4QWqxQ"
GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/geocode/json"

#MONTH = {'January': '01', 'Febuary'}
debug = False
"""
New generic method to process ADT data into csv files
"""
def process_adt_data(year, Processed_dir, Input_dir):
    """
    This function processes the Excel and PDF ADT data files (city data) into CSV files. Note that one file corresponds to one main road and the traffic flow data recordings in it.
    
    This function has input:
        - year which takes values 2013, 2015, 2017 or 2019
        - Processed_dir: path to the output
        - Input_dir: path to the inputs

    The function has output:
        - CSV files located in the Processed_dir/Year_processed/ folder where Year=2017 or 2019

    For the function to work:
        - Files should be located in 
            1. Input_dir/Year\ EXT/ folder if Year=2013, 2017 or 2019 where:
                a. Year=2013, 2017 or 2019 if Ext=ADT Data 
                b. Year=2017 or 2019 if Ext=doc for 2017 and 2019
            2. Input_dir/Raw\ data/ folder if Year=2015
    """
    output_folder = Processed_dir + "/" + "%d processed/" % year
    if not os.path.isdir(Processed_dir):
        os.mkdir(Processed_dir)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    input_file_doc = []
    if year in [2013, 2017, 2019]:
        input_folder_excel = Input_dir + "/" + "%d ADT Data/" % year
        input_files_excel = os.listdir()
    if year in [2017, 2019]:
        input_folder_doc = Input_dir + "/" + "%d doc/" % year
        input_file_doc = os.listdir(input_folder_doc)

    if year==2015:
        input_folder_excel = Input_dir + "/Raw Data/"

    input_files_excel = os.listdir(input_folder_excel)

    # parsing of the Excel files
    for file_name in input_files_excel:
        if is_excel_file(file_name):
            tmp_df = parse_adt_as_dataframe(input_folder_excel + file_name, year)

            output_name = output_folder + os.path.splitext(file_name)[0] + ".csv"
            if debug:
                print(output_folder)
                print(output_name)
            tmp_df.to_csv(output_name)

    # parsing of the doc files
    for file_name in input_file_doc:
        if is_doc_file(file_name):
            tmp_df = parse_adt_as_file(input_folder_doc + file_name, year, output_folder)


def compare(s, t):
    return Counter(s) == Counter(t)


def parse_adt_as_file(file_path, year, out_folder):
    """ 
    Process Doc files

    ***2017 and 2019 PDF files*** are structured with a header and 3 tables of traffic flow data (one table per day of subsequent days). 
    The header gives the site location and other miscellaneous meta data. Each table is titled by the date and timestep (15 minutes) of 
    the recording. A table is organized by columns each representing the hour of day (0 - 23). Hence for a given column, the first row 
    gives the hour of the day, the second gives the total flow for the hour, and the third to last row (4 rows total) gives traffic flow per
     15 minute timestep for the hour.

    
    Years 2017 and 2019 files have the same structure hence we can reuse this code.
    Structure refers to data organized in 3 tables split by *
    Note no doc files for 2013.
    """
    if debug:
        print(file_path)
    file_name = file_path.split('/')[-1]
    out_file = open(out_folder + '/' + remove_ext(file_name) + '.csv', 'w')
    out_file.write("Date,Count\n")

    # data is structured in 3 tables split by *
    text = textract.process(file_path)
    text = str(text).replace('\\n', ' ')
    data = text.split('*')
    interpreted_tables = {}
    for table in data[1:]:
        # parsing data block for day d
        array = table.split('|')
        # get day
        info_day = array[0].split(',')
        # get data start idx j
        for j in range(len(array)):
            if "0000" in array[j]:
                # print("data start idx", j)
                break
        # gather data in table_tmp
        # note each column represents an hour and the column has flow data
        # for that hour.
        table_tmp = np.zeros((6, 24))
        for i in range(6):
            data_row = [array[j + i * 26 + k] for k in range(24)]
            if debug:
                print(data_row)
                print(len(data_row))
            table_tmp[i] = np.array(data_row)
        interpreted_tables[table] = table_tmp
            # print("data_row", data_row)

        analysis = array[j + 6 * 26]  # some analysis, not useful for now

        # write data in out file
        for i in table_tmp[0]:
            # h = hour of the day (1 to 24hr)
            # j*15 = minutes (0, 15, 30, 45)
            # data given in 15 min intervals
            hr_index = ((int)(i / 100)) # (raw data given as 100 -> 1 hr)
            hr = str(hr_index)
            for j in range(4):
                # day example: Tuesday -   November 7 -  2017=13207
                day = info_day[0] + " - " + info_day[1] + " - " + info_day[2]
                time = hr + ":" + str(j * 15)
                count = str(table_tmp[2 + j][hr_index])
                #month = info_day[1].split(" ")[0]
                date_time = datetime.strptime(info_day[1] + " "  + str(year) + " " + time, " %B %d %Y %H:%M")
                Date = date_time.strftime("%Y-%m-%d %H:%M:%S") 
                out_file.write(Date + "," + count + "\n")

        # check eligibility of the doc files.
        curr_table = table_tmp
        num_rows = len(curr_table)
        for i in range(len(curr_table[0])):
            expected_sum = curr_table[1]
            actual_sum = []
            for j in range(2, num_rows):
                if actual_sum == []:
                    actual_sum = curr_table[j]
                else:
                    actual_sum = [sum(x) for x in zip(actual_sum, curr_table[j])]
            for i in range(len(actual_sum)):
                #there are actually many tiny errors in the table set, therefore, I set a boundary of 2. 
                assert abs(actual_sum[i] - expected_sum[i]) <=2 
        
        expected_total_sum = sum(expected_sum)
        real_total_sum = int(info_day[2].split("=")[1])
        #set the boundary for the test
        if debug == True:
            print("expected_sum" + str(expected_total_sum))
            print("real_total_sum " + str(real_total_sum))
        assert abs(expected_total_sum - real_total_sum) <= 15
    out_file.close()
    
    # for table in data[1:]:
    #     curr_table = interpreted_tables[table]
    #     num_rows = len(curr_table)
    #     for i in range(len(curr_table[0])):
    #         expected_sum = curr_table[1]
    #         actual_sum = []
    #         for j in range(2, num_rows):
    #             if actual_sum == []:
    #                 actual_sum = curr_table[j]
    #             else:
    #                 actual_sum = [sum(x) for x in zip(actual_sum, curr_table[j])]
    #         for i in range(len(actual_sum)):
    #             
    #             assert abs(actual_sum[i] - expected_sum[i]) <=2 
    #     expected_entire_sum =  





def parse_adt_as_dataframe(file_path, year):
    """ To do """
    xls_file = pd.ExcelFile(file_path)
    dfs = {}
    for sheet_name in xls_file.sheet_names:
        dfs[sheet_name] = xls_file.parse(sheet_name)
    
    if year==2013:
        return parse_excel_2013(dfs)
        
    if year==2015:
        return parse_excel_2015(dfs)
        
    if year==2017:
        return parse_excel_2017(dfs)
        
    if year==2019:
        return parse_excel_2019(dfs)

def parse_excel_2013(dfs):
    """ 
    ***2013 Excel files*** are structured in data sheets. The first data sheet "Summary" contains the main road, cross streets, city information and the start date of the recording. It also summarizes the data contained in all other sheets into a bar plot of traffic flow vs time of day bins (i.e Tuesday AM, Wednesday PM) for different flow directions and into a line plot of traffic flow vs. hour of day for different days of the week. The sheets that follow are named "D1", "D2",..."DN" where N denotes the N'th day since the start date. These sheets are structured into two tables, AM counts and PM counts. Each table row gives the traffic flow per timestep of 15 minutes. The first column is the time of day in hh:mm format follow by direction columns of traffic flow (NB, SB, EB, WB). 
    """
    tmp_df = dfs['Input']
    tmp_df.to_csv("test_tmp.csv")

    # read to csv directly and select columns
    tmp_df_2 = pd.read_csv("test_tmp.csv", skiprows=4)
    os.remove("test_tmp.csv")
    tmp_df_2 = tmp_df_2[['Unnamed: 1', 'TIME', 'NB', 'SB', 'EB', 'WB']]
    tmp_df_2['Date'] = tmp_df_2['Unnamed: 1'].astype(str) + ' ' + tmp_df_2['TIME']
    tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB', 'EB', 'WB']]
    if math.isnan(tmp_df_2['NB'][4]):
        tmp_df_2 = tmp_df_2[['Date', 'EB', 'WB']]
    else:
        tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB']]
    return tmp_df_2

def parse_excel_2015(dfs):
    """
    ***2015 Excel files***. The files come from Kimley Horn. Every excel file has 7 relevant sheets including the hidden sheets. 
    They are ['Data', 'ns Day 1', 'ns Day 2', 'ns Day 3', 'ew Day 1', 'ew Day 2', 'ew Day 3']. Here, we are only going to process the 'Data' sheet.
    """
    tmp_df = dfs['Data']
    tmp_df.to_csv("test_tmp.csv")

    # read to csv directly and select columns
    tmp_df_2 = pd.read_csv("test_tmp.csv", skiprows=1)
    os.remove("test_tmp.csv")
    tmp_df_2 = tmp_df_2[['Unnamed: 1', 'Northbound', 'Southbound', 'Eastbound', 'Westbound']]
    tmp_df_2 = tmp_df_2.rename(columns={'Unnamed: 1': 'Date', 'Northbound': 'NB', 'Southbound': 'SB', 'Eastbound': 'EB', 'Westbound': "WB"})
    if math.isnan(tmp_df_2['NB'][4]):
        tmp_df_2 = tmp_df_2[['Date', 'EB', 'WB']]
    else:
        tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB']]

    return tmp_df_2


def parse_excel_2017(dfs):
    """ 
    ***2017 Excel files*** are structured in one data sheet giving a header and a table for traffic flow. The header gives the start date and time of the recording, 
    site code and sensor location, and the table gives traffic flow per a 15 minute timestep. The table's first two columns give the date and time and the following 
    columns give traffic flow per directions. 
    """
    tmp_df = dfs['Sheet1']
    tmp_df.to_csv("test_tmp.csv")

    # read to csv directly and select columns
    tmp_df_2 = pd.read_csv("test_tmp.csv", skiprows=6)
    os.remove("test_tmp.csv")
    tmp_df_2 = tmp_df_2.drop(['5'], axis=1)
    tmp_df_2 = tmp_df_2.rename(columns={'Date': 'date'})
    tmp_df_2['Date'] = tmp_df_2['date'].apply(lambda x: x.split(" ")[0]) + ' ' + tmp_df_2['Time']
    # tmp_df_2 = tmp_df_2.drop('data', 1)
    cols = list(tmp_df_2.columns.values)
    tmp_df_2 = tmp_df_2[[cols[-1], cols[2], cols[3]]]
    return tmp_df_2

def parse_excel_2019(dfs):
    """ 
    ***2019 Excel files*** have similar structure as those of 2013. The data is organized in two types of sheet, "Day N" and "GR N" sheets. The "Day N" sheets give traffic flow data in the same fashion as the "DN" sheets of 2013 excel files. The day of recording can be found in the header of the two tables. The "GR N" sheets plot the corresponding flow data of the "Day N" sheets. A line plot of flow vs. hour of day for different flow directions is given.
    """
    # parse the excel sheets and concatenate their data
    Input1 = parse_sheet_helper_2019('Day 1', dfs)
    Input2 = parse_sheet_helper_2019('Day 2', dfs)
    Input3 = parse_sheet_helper_2019('Day 3', dfs)
    tmp_df_2 = Input1.append(Input2, ignore_index=True)
    tmp_df_2 = tmp_df_2.append(Input3, ignore_index=True)

    # rename columns accordingly
    # (matching was done visually printing the data frames and matching to xls data)
    # print(Input1)
    tmp_df_2 = tmp_df_2.rename(columns={'Unnamed: 25': 'time',
                                    'Unnamed: 26': 'NB', 'Unnamed: 27': 'SB',
                                    'Unnamed: 28': 'EB', 'Unnamed: 29': 'WB'})

    tmp_df_2['Date'] = tmp_df_2['date'].astype(str) + ' ' + tmp_df_2['time'].astype(str) 

    tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB', 'EB', 'WB']]

    if math.isnan(tmp_df_2['NB'][4]):
        tmp_df_2 = tmp_df_2[['Date', 'EB', 'WB']]
    else:
        tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB']]
    return tmp_df_2


def parse_sheet_helper_2019(name, dfs):
    """ To do """
    Input = dfs[name]
    day = Input.iloc(1)[1][3]
    #     print(day)
    col = Input.columns
    Input = Input.drop(col[:25], axis=1)
    Input = Input.drop(col[30:], axis=1)
    Input = Input.drop(Input.index[[i for i in range(9)]])
    Input = Input.assign(date=day)
    return Input


def is_excel_file(file_name):
    """ To do """
    _, file_ext = os.path.splitext(file_name)
    return (file_ext == '.xls' or file_ext == '.xlsx') and is_valid_file(file_name)


def is_valid_file(file_name):
    """ To do """
    is_folder = os.path.isdir(os.getcwd() + '/' + file_name)
    return '$' not in file_name and '.DS_Store' not in file_name and not is_folder

# removes extension from file name
def remove_ext(file_path):
    """ To do """
    file_name, _ = os.path.splitext(file_path)
    return file_name


def remove_direction(string):
    """ To do """
    return string.replace('Nb', '').replace('Sb', '')\
        .replace('Eb', '').replace('Wb', '')


def get_of_direction(string):
    """ To do """
    of_directions = ['S Of', 'E Of', 'W Of', 'N Of']
    for of in of_directions:
        if of in string:
            return of


def find_splitter(string, splitters):
    """ To do """
    for splitter in splitters:
        if splitter in string:
            return splitter


def is_doc_file(filename):
    """ To do """
    _, file_ext = os.path.splitext(filename)
    return file_ext == '.doc' and is_valid_file(filename)

def create_directory(path, name):
    """ To do """
    dir = path + "/" + name
    if not os.path.isdir(dir):
        os.mkdir(dir)

##############################
############ HERE ############
##############################

def get_geo_data(year, Input_dir, Processed_dir):
    """
    This function iterates over the ADT files and obtains the adresses of the detectors to then use with Google API to obtain latitude and longitude coordinates.
    
    This function has input:
        - Year takes values 2013, 2015, 2017, 2019
        - Processed_dir: path to the output
        - Input_dir: path to the inputs

    This function has output:
        - CSV file "year_info_coor.csv" containing the coordinates of detectors and located in the Processed_dir/Year_processed/ folder where Year=2017 or 2019

    For the function to work:
        - Files should be located in 
            1. Input_dir/Year\ EXT/ folder if Year=2013, 2017 or 2019 where:
                a. Year=2013, 2017 or 2019 if Ext=ADT Data 
                b. Year=2017 or 2019 if Ext=doc for 2017 and 2019
            2. Input_dir/Raw\ data/ folder if Year=2015 

    + add the doc files
    """
    if debug:
        print('Obtaining geo data from %d ADT files' % year)
    # input folder and files
    
    input_file_doc = []
    if year in [2013, 2017, 2019]:
        input_folder_excel = Input_dir + "/" + "%d ADT Data/" % year
        input_files_excel = os.listdir()
    if year in [2017, 2019]:
        input_folder_doc = Input_dir + "/" + "%d doc/" % year
        input_file_doc = os.listdir(input_folder_doc)
        # print("the doc files")
        # print(input_file_doc)
    if year==2015:
        input_folder_excel = Input_dir + "/Raw Data/"

    input_files_excel = os.listdir(input_folder_excel)

    # iterate over the excel files to obtain main road addresses
    cache_main_roads = []
    for file_name in input_files_excel:
        if is_valid_file(file_name) and is_excel_file(file_name):
            if debug:
                print("processing:", file_name)
            main_road_info = None
            if year == 2013:
                main_road_info = get_main_road_info_2013(input_folder_excel, file_name)
            elif year == 2015:
                main_road_info = get_main_road_info_2015(file_name)
            elif year == 2017:
                main_road_info = get_main_road_info_2017(file_name)
            elif year == 2019:
                main_road_info = get_main_road_info_2019(file_name)
            else:
                raise (Exception('Unable to get main road info for file: %s' % file_name))
            if debug:
                print('main road info:', main_road_info)
            cache_main_roads.append(main_road_info)
    
    # iterate over the doc files to obtain main road addresses
    if year in [2017, 2019]:
        for file_name in input_file_doc:
            if is_valid_file(file_name) and is_doc_file(file_name):
                if debug:
                    print("processing:", file_name)
                main_road_info = None
                if year == 2017:
                    main_road_info = get_main_road_info_2017(file_name)
                elif year == 2019:
                    main_road_info = get_main_road_info_2019(file_name)
                else:
                    raise (Exception('Unable to get main road info for file: %s' % file_name))
                if debug:
                    print('main road info:', main_road_info)
                cache_main_roads.append(main_road_info)
            
    # get geo coordinates using google API and cache_main_roads
    # write the results in the csv file, 'year_info_coor.csv'
    fname = Processed_dir + "/" + '%d_info_coor.csv' % year
    coordinate_file = open(fname, 'w')
    coordinate_file.write("Name,City,Main road,Cross road,Start lat,Start lng,End lat,End lng\n")
    for main_road_info in cache_main_roads:
        file_name, city, main_road, cross_road, cross1, cross2 = main_road_info
        if cross1 and cross2:
            lat1, lng1 = get_coords_from_address(main_road + ' & ' + cross1 + ', ' + city)
            lat2, lng2 = get_coords_from_address(main_road + ' & ' + cross2 + ', ' + city)
            line = file_name + ',' + city + ',' + main_road + ',' + cross_road + ',' + \
                   str(lat1) + ',' + str(lng1) + ',' + str(lat2) + ',' + str(lng2) + '\n'
            coordinate_file.write(line)
        elif cross1:
            lat, lng = get_coords_from_address(main_road + ' & ' + cross1 + ', ' + city)
            line = file_name + ',' + city + ',' + main_road + ',' + cross_road + ',' + \
                   str(lat) + ',' + str(lng) + '\n'
            coordinate_file.write(line)
        else:
            raise (Exception('Unable to get coordinates for main road of file %s' % file_name))
    if debug:
        print(coordinate_file)
    coordinate_file.close()

"""
Gets main road info from inside the file 
"""

def get_main_road_info_2013(in_folder, file_name):
    """ To do """
    if debug:
        print(file_name)
    if is_excel_file(file_name):
        # read excel data into dataframe
        xls_file = pd.ExcelFile(in_folder + file_name)
        dfs = {}
        for sheet_name in xls_file.sheet_names:
            dfs[sheet_name] = xls_file.parse(sheet_name)

        # get main road info (to see respective columns do a print)
        Input = dfs['Input']
        # print(Input)
        city = Input.columns[7]
        main_road = Input[city][0]
        cross_road = Input[city][1]
        city = city.split(",")[0]  # Fremont, CA -> Fremont
        cross1, cross2 = get_cross_roads_2013(cross_road)
        main_road_info = (file_name, city, main_road, cross_road, cross1, cross2)
        return main_road_info
    else:
        raise (Exception("Can't get main road info from file %s" % file_name))

# Between Arapaho and Paseo Padre -> Arapaho, Paseo Padre
# 200' s/o Starlite -> Starlite
def get_cross_roads_2013(cross):
    """ To do """
    cross1, cross2 = None, None
    if 'Between' in cross:
        # example: Between Arapaho and Paseo Padre
        cross = cross.replace('Between', '').strip()
        cross1 = cross.split('and')[0].strip()
        cross2 = cross.rsplit('and')[1].strip()
    else:
        # example: 200' s/o Starlite -> Starlite
        cross1 = cross[9:]
    return cross1, cross2



def get_main_road_info_2015(file_name):
    """
    getting 2015 main road info from file name

    """

    if is_excel_file(file_name):
        if debug:
            print(file_name)
        main_road = file_name.split('betw.')[0].strip()
        cross = file_name.split('betw.')[1].strip()[:-4]
        cross1 = cross.split('and')[0].strip()
        cross2 = cross.split('and')[1].strip()
        city = 'Fremont'
        main_road_info = (file_name, city, main_road, cross, cross1, cross2)
        return main_road_info


def get_main_road_info_2017(file_name):

    """
    Get 2017 main road info from file name
    input: file_name, examples below 
    file_name = mission blvd BT driscoll rd AND I 680 NB
    file_name = mission blvd S OF washington blvd signal
    output = (file_name, city, main_road, cross_road, cross1, cross2)
    """

    name = remove_ext(file_name).title()
    city = 'Fremont'
    main_road_info = None
    if 'Bt' in name:
        # Ex1: mission blvd BT driscoll rd AND I 680 NB
        main_road = name.split('Bt')[0].strip()
        cross_road = name.split('Bt')[1].strip()
        cross1 = remove_direction(cross_road.split('And')[0]).strip()
        cross2 = remove_direction(cross_road.split('And')[1]).strip()
        main_road_info = (file_name, city, main_road, cross_road, cross1, cross2)
    elif get_of_direction(name):
        # Ex2: mission blvd S OF washington blvd signal
        of_direction = get_of_direction(name)  # S OF
        main_road = name.split(of_direction)[0].strip()
        cross_road = name.split(of_direction)[1] \
            .replace('Signal', '') \
            .replace('Stop Sign', '') \
            .strip()
        cross1 = remove_direction(cross_road)
        main_road_info = (file_name, city, main_road, cross_road, cross1, None)
    else:
        raise (Exception('Unable to parse main road info from 2017 file name: %s' % file_name))

    return main_road_info


def get_main_road_info_2019(file_name):

    """
    Get 2019 main road info from file name
    input: file_name, examples below 
    file_name = Driscoll Rd Bet. Mission Blvd & Paseo Padre Pkwy
    file_name = AUTO MALL PKWY BT FREMONT BLVD AND I-680 EB
    output = (file_name, city, main_road, cross_road, cross1, cross2)
    """

    name = remove_ext(file_name).title()
    city = 'Fremont'

    bt = find_splitter(name, ['Bt', 'Bet.'])
    main_road = name.split(bt)[0].strip()
    cross_road = name.split(bt)[1].strip()

    And = find_splitter(cross_road, ['And', '&'])
    cross1 = remove_direction(cross_road).split(And)[0].strip()
    cross2 = remove_direction(cross_road).split(And)[1].strip()

    main_road_info = (file_name, city, main_road, cross_road, cross1, cross2)
    return main_road_info


def get_coords_from_address(address):
    """ To do """
    payload = {
        'address': address,
        'key': API_KEY
    }
    if debug:
        print('address: ', address)
    request = requests.get(GOOGLE_MAPS_URL, params=payload).json()
    results = request['results']

    lat = None
    lng = None

    if len(results):
        answer = results[0]
        lat = answer.get('geometry').get('location').get('lat')
        lng = answer.get('geometry').get('location').get('lng')
    if debug:
        print('address w coord lat, lng', address, str(lat), str(lng))
    return lat, lng



# for local testing only
if __name__ == '__main__':
    #process_adt_data(2015)
    #process_doc_data(2019)
    #get_geo_data(2015)
    pass
