import pandas as pd
import os
import requests
import textract
import numpy as np
from pathlib import Path
import math
from datetime import datetime
import glob
import csv
import re

API_KEY = "AIzaSyB8rJhDsfwvIod9jVTfFm1Dtv2eO4QWqxQ"
GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/geocode/json"

ERRONEOUS_FILES = ['DURHAM RD BT I-680 AND MISSION BLVD EB', 'MISSION BLVD BT WASHINGTON BLVD AND PINES ST SB']

debug = False


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
        _, file_ext = os.path.splitext(file_name)
        if (file_ext == '.xls' or file_ext == '.xlsx'): #and is_valid_file(file_name)
            tmp_df = parse_adt_as_dataframe(input_folder_excel + file_name, year)
            output_name = output_folder + os.path.splitext(file_name)[0] + ".csv"
            if debug:
                print(output_folder)
                print(output_name)
            tmp_df.to_csv(output_name)

    # parsing of the doc files
    for file_name in input_file_doc:
        _, file_ext = os.path.splitext(file_name)
        if (file_ext == '.doc'):
        #if is_doc_file(file_name):
            tmp_df = parse_adt_as_file(input_folder_doc + file_name, year, output_folder)


def doc_file_tester(Input_dir, years_of_interests):
    """
    This function takes in a list of years whose raw flow files contain doc files, 
    and test the eligibility of the doc files for each year of request.  
    """
    if (2015 in years_of_interests) or (2013 in years_of_interests):
        print("the raw flow data of 2013/2015 doesn't contain doc files")
        return
    for year in years_of_interests:
        input_folder_doc = Input_dir + "/" + "%d doc/" % year
        input_file_docs = os.listdir(input_folder_doc)
        for file_name in input_file_docs:
            _, file_ext = os.path.splitext(file_name)
            if file_ext == '.doc':
                doc_file_testing(input_folder_doc+file_name, year)
    print("Doc files are all good for both 2017 and 2019!")


def doc_file_testing(file_path, year):
    """
    This testing script ensures the accurary of datapoints in doc files by 
    applying an arbitrary upper bound on the difference between expected sum value
    and actual sum values.

    """

    file_name = file_path.split('/')[-1]
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
    out_file = open(out_folder + '/' + os.path.splitext(file_name)[0] + '.csv', 'w')
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
    
def parse_adt_as_dataframe(file_path, year):
    """ 
    This function adds every sheet in every ADT file for a certain year to a dictionary 
    and calls the helper function for each year to parse the data sheets. 
    """
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
    #tmp_df_2[pd.isnull(tmp_df_2['NB']) or pd.isnull(tmp_df_2['SB'] or pd.isnull(tmp_df_2['EB'] or pd.isnull(tmp_df_2['WB']))]
    
    if math.isnan(tmp_df_2['NB'][4]):
        tmp_df_2 = tmp_df_2[['Date', 'EB', 'WB']]
        tmp_df_2.dropna(subset=['EB'], inplace=True)
        tmp_df_2.dropna(subset=['WB'], inplace=True)
    else:
        tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB']]
        tmp_df_2.dropna(subset=['NB'], inplace=True)
        tmp_df_2.dropna(subset=['SB'], inplace=True)
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
        tmp_df_2.dropna(subset=['EB'], inplace=True)
        tmp_df_2.dropna(subset=['WB'], inplace=True)
    else:
        tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB']]
        tmp_df_2.dropna(subset=['NB'], inplace=True)
        tmp_df_2.dropna(subset=['SB'], inplace=True)
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
        print()
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
        _, file_ext = os.path.splitext(file_name)
        is_folder = os.path.isdir(os.getcwd() + '/' + file_name)
        print(is_folder)
        if (file_ext == '.xls' or file_ext == '.xlsx') and ('$' not in file_name and '.DS_Store' not in file_name and not is_folder):
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
            _, file_ext = os.path.splitext(file_name)
            is_folder = os.path.isdir(os.getcwd() + '/' + file_name)
            print(is_folder)
            if (file_ext == '.doc') and ('$' not in file_name and '.DS_Store' not in file_name and not is_folder):
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





def get_main_road_info_2013(in_folder, file_name):
    """
    get the main road info from file name for year 2013
    input: file_name, examples below 
    file_name = Auto Mall Pkwy betw. Fremont & I680
    file_name = Driscoll Rd betw. Mission & PPP
    output = (file_name, city, main_road, cross_road, cross1, cross2)
    """
    if debug:
        print(file_name)
    _, file_ext = os.path.splitext(file_name)
    is_folder = os.path.isdir(os.getcwd() + '/' + file_name)
    if (file_ext == '.xls' or file_ext == '.xlsx') and ('$' not in file_name and '.DS_Store' not in file_name and not is_folder):
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


def get_cross_roads_2013(cross):
    """ 
    get the cross road information of every detector for 2013:
    
    Example: 
    - Between Arapaho and Paseo Padre -> Arapaho, Paseo Padre
    - 200' s/o Starlite -> Starlite
    """
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
    Extract the main road info from file name for year 2015

    input: file_name, examples below 
    file_name = Durham Rd betw. I-680 and Mission
    file_name = Grimmer Blvd (South) betw. Osgood and Fremont
    output = (file_name, city, main_road, cross_road, cross1, cross2)
    """
    _, file_ext = os.path.splitext(file_name)
    is_folder = os.path.isdir(os.getcwd() + '/' + file_name)
    if (file_ext == '.xls' or file_ext == '.xlsx') and ('$' not in file_name and '.DS_Store' not in file_name and not is_folder):
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
    file_name, file_ext = os.path.splitext(file_name)
    name = file_name.title()
    city = 'Fremont'
    main_road_info = None


    of_directions = ['S Of', 'E Of', 'W Of', 'N Of']
    for of in of_directions:
        if of in name:
            Direction_b = of
    if 'Bt' in name:
        # Ex1: mission blvd BT driscoll rd AND I 680 NB
        main_road = name.split('Bt')[0].strip()
        cross_road = name.split('Bt')[1].strip()

        #remove_direction
        cross1 = (cross_road.split('And')[0].replace('Nb', '').replace('Sb', '')\
        .replace('Eb', '').replace('Wb', '')).strip()
        cross2 = (cross_road.split('And')[1].replace('Nb', '').replace('Sb', '')\
        .replace('Eb', '').replace('Wb', '')).strip()
        main_road_info = (file_name, city, main_road, cross_road, cross1, cross2)
    elif Direction_b:
        # Ex2: mission blvd S OF washington blvd signal
        of_direction = Direction_b  # S OF
        main_road = name.split(of_direction)[0].strip()
        cross_road = name.split(of_direction)[1] \
            .replace('Signal', '') \
            .replace('Stop Sign', '') \
            .strip()
        cross1 = cross_road.replace('Nb', '').replace('Sb', '')\
        .replace('Eb', '').replace('Wb', '')
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
    file_name, _ = os.path.splitext(file_name)
    name = file_name.title()
    city = 'Fremont'

    for splitter in ['Bt', 'Bet.']:
        if splitter in name:
            bt = splitter

    #bt = find_splitter(name, ['Bt', 'Bet.'])
    main_road = name.split(bt)[0].strip()
    cross_road = name.split(bt)[1].strip()

    for splitter in ['And', '&']:
        if splitter in cross_road:
            And = splitter
    #And = find_splitter(cross_road, ['And', '&'])
    cross1 = (cross_road.replace('Nb', '').replace('Sb', '')\
        .replace('Eb', '').replace('Wb', '')).split(And)[0].strip()
    cross2 = (cross_road.replace('Nb', '').replace('Sb', '')\
        .replace('Eb', '').replace('Wb', '')).split(And)[1].strip()

    main_road_info = (file_name, city, main_road, cross_road, cross1, cross2)
    return main_road_info


def get_coords_from_address(address):
    """
    Get the geographic coordination of an address using Google API
    """
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


def google_doc_generater(Processed_dir):
    """
    This script generated the google doc file with the exact same (Id, file_name) pairs. 
    The function goes over year_info_coor.csv files and concatenate them into one large csv in consistent format
    flow_out.csv is the same as the google doc 

    """
    error_id = [12, 60, 61, 62, 63, 64]
    path = Processed_dir
    all_files = glob.glob(path + "/*.csv")
    li = []
    id_counter = 1

    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        if curr_year in ['2013', '2017', '2019']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df.sort_values(df.columns[0], ascending = True)
            df = df.reset_index(drop=True)
            ids = []
            for i in range(len(df['Name'])):
                file_name = df['Name'][i]
                ids.append(id_counter)
                if ('EB' in file_name) or ('WB' in file_name) or ('NB' in file_name) or ('SB' in file_name):
                    id_counter += 1
                    while id_counter in error_id:
                        id_counter += 1
                else:
                    id_counter += 2
                    while id_counter in error_id:
                        id_counter += 1
            df.insert(0,'Id',ids)
            header = {'Id':'Id', 'Name':curr_opening, 'Main road':'Main road', 'Cross road':'Cross road', 'Start lat':'Start lat', 'Start lng':'Start lng', 'End lat':'End lat', 'End lng':'End lng'}
            df['Id'] = ids
            df = df.rename(columns={'Name': curr_opening})
            li.append(df)
    # For PeMS:
    PeMS_file = path + "/" + 'Flow_processed_all.csv'
    df = pd.read_csv(PeMS_file, index_col=None, header=0)
    PeMS_section = df[df['Name'].apply(lambda x: x.split(" ")[0] == "PeMS")]
    PeMS_section = PeMS_section[PeMS_section['Year'].apply(lambda x: x == 2013)]
    PeMS_section["Name"] = PeMS_section["Name"].apply(lambda x: x.split(" ")[-1])
    ids = []
    for i in range(len(PeMS_section)):
        detector_id = PeMS_section.iloc[i]
        ids.append(id_counter)
        id_counter += 1
    PeMS_section = PeMS_section[["Name"]]
    PeMS_section.insert(0,'Id',ids)
    PeMS_section = PeMS_section.rename(columns={'Name': "PeMS"})
    li.append(PeMS_section)  

    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        if curr_year in ['2015']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df.sort_values(df.columns[0], ascending = True)
            df = df.reset_index(drop=True)
            r = np.arange(len(df)).repeat(2)
            df = pd.DataFrame(df.values[r], df.index[r], df.columns)
            df = df.reset_index(drop=True)
            ids = []
            for i in range(len(df['Name'])):
                file_name = df['Name'][i]
                ids.append(id_counter)
                id_counter += 1
            df.insert(0,'Id',ids)
            header = {'Id':'Id', 'Name':curr_opening, 'Main road':'Main road', 'Cross road':'Cross road', 'Start lat':'Start lat', 'Start lng':'Start lng', 'End lat':'End lat', 'End lng':'End lng'}
            df['Id'] = ids
            df = df.rename(columns={'Name': curr_opening})
            li.append(df)

    df1,df2, df3, df4, df5  = li[0], li[1], li[2], li[3], li[4]

    df1.to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False)
    df2.to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')
    df3.to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')
    df4.to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')
    df5.to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')



def flow_processed_generater(Processed_dir):
    """
    This script create an updated version of Flow_processed_tmp.csv
    """
    error_id = [12, 60, 61, 62, 63, 64]
    path = Processed_dir
    all_files = glob.glob(path + "/*.csv")
    li = []
    id_counter = 1

    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        if curr_year in ['2013', '2017', '2019']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df.sort_values(df.columns[0], ascending = True)
            df = df.reset_index(drop=True)
            ids = []
            for i in range(len(df['Name'])):
                file_name = df['Name'][i]
                ids.append(id_counter)
                if ('EB' in file_name) or ('WB' in file_name) or ('NB' in file_name) or ('SB' in file_name):
                    id_counter += 1
                    while id_counter in error_id:
                        id_counter += 1
                else:
                    id_counter += 2
                    while id_counter in error_id:
                        id_counter += 1
            df.insert(0,'Id',ids)
            header = {'Id':'Id', 'Name':curr_opening, 'Main road':'Main road', 'Cross road':'Cross road', 'Start lat':'Start lat', 'Start lng':'Start lng', 'End lat':'End lat', 'End lng':'End lng'}
            df['Id'] = ids
            df = df.rename(columns={'Name': curr_opening})
            df =  df[['Id', curr_opening]]
            li.append(df)

    # For PeMS:
    PeMS_file = path + "/" + 'Flow_processed_all.csv'
    df = pd.read_csv(PeMS_file, index_col=None, header=0)
    PeMS_section = df[df['Name'].apply(lambda x: x.split(" ")[0] == "PeMS")]
    PeMS_section = PeMS_section[PeMS_section['Year'].apply(lambda x: x == 2013)]
    PeMS_section["Name"] = PeMS_section["Name"].apply(lambda x: x.split(" ")[-1])
    ids = []
    for i in range(len(PeMS_section)):
        detector_id = PeMS_section.iloc[i]
        ids.append(id_counter)
        id_counter += 1
    PeMS_section = PeMS_section[["Name"]]
    PeMS_section.insert(0,'Id',ids)
    PeMS_section = PeMS_section.rename(columns={'Name': "PeMS"})
    li.append(PeMS_section)  
    

    #for 2015
    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        if curr_year in ['2015']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df.sort_values(df.columns[0], ascending = True)
            df = df.reset_index(drop=True)
            r = np.arange(len(df)).repeat(2)
            df = pd.DataFrame(df.values[r], df.index[r], df.columns)
            df = df.reset_index(drop=True)
            ids = []
            for i in range(len(df['Name'])):
                file_name = df['Name'][i]
                ids.append(id_counter)
                id_counter += 1
            df.insert(0,'Id',ids)
            header = {'Id':'Id', 'Name':curr_opening, 'Main road':'Main road', 'Cross road':'Cross road', 'Start lat':'Start lat', 'Start lng':'Start lng', 'End lat':'End lat', 'End lng':'End lng'}
            df['Id'] = ids
            df = df.rename(columns={'Name': curr_opening})
            df =  df[['Id', curr_opening]]
            li.append(df)

    df1,df2, df3, df4, df5  = li[0], li[1], li[2], li[3], li[4]

    df1.to_csv(Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False)
    df2.to_csv(Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')
    df3.to_csv(Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')
    df4.to_csv(Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')
    df5.to_csv(Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')



def Speed_data_parser(speed_data_dir, Processed_dir):
    """
    This scripts parse the speed data 2015
    """

    df_num = pd.DataFrame()
    df_percent = pd.DataFrame()

    for f in os.listdir(speed_data_dir):
        print(f)
        if('xls' in f):
            input_file_all = pd.read_excel(speed_data_dir + "/" + f, sheet_name = '#2').rename(columns={'City of Fremont':'speed'})
            direction_descr = input_file_all['Unnamed: 10'][15] # 17K in the excel file
            speed_limit = input_file_all['Unnamed: 34'][72]#74AI
            if 'East' in direction_descr:
                direction = ['EB', 'WB']
            else:
                direction = ['NB', 'SB']
            input_file = input_file_all.iloc[17:68].rename(columns={'City of Fremont':'speed','Unnamed: 31': direction[0], 'Unnamed: 32': direction[1]})
            # input_file = input_file_all.iloc[17:68,31:33]
            input_file = input_file[['speed', direction[0], direction[1]]].fillna(value = 0).reset_index()
            sum1 = input_file[direction[0]].sum()
            sum2 = input_file[direction[1]].sum()
            series1 = input_file[direction[0]][::-1]/sum1
            cumsum1 = series1.cumsum()
            input_file[direction[0]+'_Cum']=cumsum1[::-1]
            series2 = input_file[direction[1]][::-1]/sum2
            cumsum2 = series2.cumsum()
            input_file[direction[1]+'_Cum']=cumsum2[::-1]
            input_file = input_file[::-1].transpose().drop('index', axis=0)
            input_file.insert(0,'Name',f)
            
            input_file.insert(1,'id','')
            input_file.insert(2,'Direction','')
            input_file.insert(3,'Speed limit',speed_limit)
            input_file.iloc[1,2]=direction[0]
            input_file.iloc[3,2]=direction[0]
            input_file.iloc[2,2]=direction[1]
            input_file.iloc[4,2]=direction[1]
            
            df_num = df_num.append(input_file.iloc[1:3])
            df_percent = df_percent.append(input_file.iloc[3:5])

    df_num.to_csv(Processed_dir + "/" + '2015_Speed_Processed_Num.csv')
    df_percent.to_csv(Processed_dir + "/" + '2015_Speed_Processed_Percent.csv')


def parse_2013(line, w, Processed_dir):
    year = 2013
    splitted = line.split(",")
    if len(splitted) == 2:
        id_flow, title = splitted
    else: 
        id_flow, title = splitted[0], splitted[1]
    title = title.replace('\n', '')
    if title == '':
        return
    if "EB " not in title and "WB " not in title:
        data = pd.read_csv(Processed_dir + "/City/" + str(year) + " reformat/" + title.split('.x')[0] + ".csv")
        for c in data.columns:
            if data[c].count() == 0:
                data = data.drop(columns=c)
        col = data.columns
        day = data[col[1]][0]
        direction1 = data[col[3]].to_numpy()[:data[col[3]].count()]
        direction2 = data[col[4]].to_numpy()[:data[col[4]].count()]
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + col[3] + "," + day)
        for i in direction1:
            w.write("," + str(((int)(i))))
        w.write("\n" + str(year) + "," + title + "," + str(((int)(id_flow)) + 1) + "," + col[4] + "," + day)
        for i in direction2:
            w.write("," + str(((int)(i))))
    else:
        data = pd.read_csv(Processed_dir + "/City/" + str(year) + " reformat/" + title.split('.x')[0] + ".csv")
        for c in data.columns:
            if data[c].count() == 0:
                data = data.drop(columns=c)
        col = data.columns
        day = data[col[1]][0]
        direction = data[col[3]].to_numpy()[:data[col[3]].count()]
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + col[3] + "," + day)
        for i in direction:
            w.write("," + str(((int)(i))))


def parse_2015(line, w, Processed_dir):
    year = 2015
    id_flow, title = line.split(",")
    title = title.replace('\n', '')
    if title == '':
        return
    if "EB " not in title and "WB " not in title:
        data = pd.read_csv(Processed_dir + "/City/" + str(year) + " reformat/" + title.split('.x')[0] + ".csv")
        for c in data.columns:
            if data[c].count() == 0:
                data = data.drop(columns=c)
        col = data.columns
        day = data[col[1]][0]
        direction1 = data[col[3]].to_numpy()[:data[col[3]].count()]
        direction2 = data[col[4]].to_numpy()[:data[col[4]].count()]
        
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + col[3] + "," + day)
        for i in direction1:
            w.write("," + str(((int)(i))))
        w.write("\n" + str(year) + "," + title + "," + str(((int)(id_flow)) + 1) + "," + col[4] + "," + day)
        for i in direction2:
            w.write("," + str(((int)(i))))
    else:
        data = pd.read_csv(Processed_dir + "/City/" + str(year) + " reformat/" + title.split('.x')[0] + ".csv")
        for c in data.columns:
            if data[c].count() == 0:
                data = data.drop(columns=c)
        col = data.columns
        day = data[col[1]][0]
        direction = data[col[3]].to_numpy()[:data[col[3]].count()]
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + col[3] + "," + day)
        for i in direction:
            w.write("," + str(((int)(i))))


def parse_2017(line, w, Processed_dir):
    year = 2017
    splitted = line.split(",")
    if len(splitted) == 2:
        id_flow, title = splitted
    else: 
        id_flow, title = splitted[0], splitted[1]
    title = title.replace('\n', '')
    if title == '':
        return
    if ".pdf" in title:
        data = pd.read_csv(Processed_dir + "/City/" + str(year) + " reformat/Format from pdf/" + title.split('.p')[0] + ".csv")
        day = data['Day'][0]
        direction = data['Count'].to_numpy()
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + title.split('.p')[0][-2:] + "," + day)
        for i in direction:
            w.write("," + str(((int)(i))))
    elif ".x" in title:
        data = pd.read_csv(Processed_dir + "/City/" + str(year) + " reformat/Format from xlsx/" + title.split('.x')[0] + ".csv")
        day = data['Date'][0]
        col = data.columns
        direction1 = data[col[3]].to_numpy()[:data[col[3]].count()]
        direction2 = data[col[4]].to_numpy()[:data[col[4]].count()]
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + col[3] + "," + day)
        for i in direction1:
            w.write("," + str(((int)(i))))
        w.write("\n" + str(year) + "," + title + "," + str(((int)(id_flow)) + 1) + "," + col[4] + "," + day)
        for i in direction2:
            w.write("," + str(((int)(i))))
    else:
        print('2017 error')
        print("ERROR HERE")
        return -1


def parse_2019(line, w, re_formated_Processed_dir):
    year = 2019
    splitted = line.split(",")
    if len(splitted) == 2:
        id_flow, title = splitted
    else: 
        id_flow, title = splitted[0], splitted[1]
    title = title.replace('\n', '')
    filename = os.path.splitext(title)[0]    #get_file_name(title)
    
    # Don't parse files known to be erroneous
    if filename in ERRONEOUS_FILES:
        print("file not processed: ", filename)  
        return

    if title == '':
        return

    if ".pdf" in title:
        data = pd.read_csv(re_formated_Processed_dir + "/City/" + str(year) + " reformat/Format from pdf/" + title.split('.p')[0] + ".csv")
        day = data['Day'][0]
        direction = data['Count'].to_numpy()
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + title.split('.p')[0][-2:] + "," + day)
        for i in direction:
            w.write("," + str(((int)(i))))
    elif ".x" in title:
        data = pd.read_csv(re_formated_Processed_dir + "/City/" + str(year) + " reformat/Format from xlsx/" + title.split('.x')[0] + ".csv")
        col = data.columns
        day = data[col[6]][0]
        for c in data.columns:
            if (data[c].sum() == 0):
                data = data.drop(columns=c)
        col = data.columns
        direction1 = data[col[2]].to_numpy()[:data[col[2]].count()]
        direction2 = data[col[3]].to_numpy()[:data[col[3]].count()]
        w.write("\n" + str(year) + "," + title + "," + id_flow + "," + col[2] + "," + day)
        for i in direction1:
            w.write("," + str(((int)(i))))
        w.write("\n" + str(year) + "," + title + "," + str(((int)(id_flow)) + 1) + "," + col[3] + "," + day)
        for i in direction2:
            w.write("," + str(((int)(i))))
    else:
        print('2019 error')
        print(line)
        print("ERROR HERE")
        return -1

def parse_PeMS(line, w, PeMs_dir):
    splitted = line.split(",")
    if len(splitted) == 2:
        id_flow, id_pems = splitted
    else: 
        id_flow, id_pems = splitted[0], splitted[1]
    id_pems = id_pems.replace('\n', '')
    data_flow = ""
    w.write("\nPeMS Detector " + id_pems + "," + id_flow)
    
    for year in [2013, 2015, 2017, 2019]:
        if (id_flow != "") and (id_pems != ""): 
            excel_dir = PeMs_dir + "/PeMS_" + str(year) + "/" + id_pems + "_" + str(year) + ".xlsx"
            #print(excel_dir)
            xl_file = pd.ExcelFile(excel_dir)
            dfs = {sheet_name: xl_file.parse(sheet_name)
                   for sheet_name in xl_file.sheet_names}
            data = dfs['Report Data']
            data_report = dfs['PeMS Report Description']
            name = data_report['Unnamed: 2'][12]
            if year == 2013:
                w.write("," + name)
            if data['5 Minutes'].shape[0] == 0:
                w.write(",Did not exist in " + str(year) + ",X")
                for k in range(3):
                    for i in range(24):
                        for j in range(4):
                            data_flow += ","
                continue
            day = data['5 Minutes'][0]
            obs = data["% Observed"].mean()
            direction_tmp = data['Flow (Veh/5 Minutes)'].to_numpy()
            direction = np.zeros(((int)(direction_tmp.shape[0] / 3)))
            for i in range(direction.shape[0]):
                data_flow += "," + str((int)(np.sum([direction_tmp[3 * i + k] for k in range(3)])))
            w.write("," + str(obs) + "," + str(day))
        w.write(data_flow)


def process_data(Processed_dir, re_formated_Processed_dir, PeMs_dir):
    # contains list of processed file names from all years and
    # all file types and from city and PeMS data
    curr_file = open(re_formated_Processed_dir + "/" + "Flow_processed_tmp.csv", "r", encoding= 'unicode_escape')#encoding='utf-8-sig').strip()
    w = open(re_formated_Processed_dir + "/" +'Flow_processed_city.csv', 'w')
    w2 = open(re_formated_Processed_dir + "/" +'Flow_processed_PeMS.csv', 'w')
    legend = "Year,Name,Id,Direction,Day 1"

    # legend for city csv
    for k in range(3):
        for i in range(24):
            for j in range(4):
                legend = legend + ",Day " + str(k + 1) + " - " + str(i) + ":" + str(15 * j)
    w.write(legend)

    # legend for pems csv
    legend2 = "Name,Id,Name PeMS,Observed 2013,Day 2013,Observed 2015,Day 2015,Observed 2017,Day 2017, Observed 2019,Day 2019"
    for year in [2013, 2015, 2017, 2019]:
        for k in range(3):
            for i in range(24):
                for j in range(4):
                    legend2 = legend2 + "," + str(year) + "-Day " + str(k + 1) + " - " + str(i) + ":" + str(15 * j)   
    w2.write(legend2)

    # parse flow processed temp file and write to city or pems csv
    year = 0
    pems = False

    for line in curr_file:
        #check whether we are processing PeMs data or ADT data 
        if 'PeMS' in line:
            pems = True
        elif "ADT" in line:
            pems = False
        #once we have determined which section we belong to, we could start process accordingly
        if pems:
            parse_PeMS(line, w2, PeMs_dir)
        elif "ADT" in line:
            year = (int)(line.split(',./')[1].split(' ')[0])
        elif year == 2013:
            if parse_2013(line, w, re_formated_Processed_dir) == -1:
                break
        elif year == 2015:
            if parse_2015(line, w, re_formated_Processed_dir) == -1:
                break        
        elif year == 2017:
            if parse_2017(line, w, re_formated_Processed_dir) == -1:
                break
        elif year == 2019:
            if parse_2019(line, w, re_formated_Processed_dir) == -1:
                break
        else:
            print(line)
            print("ERROR")
            break

    curr_file.close()
    w.close()
    w2.close()
