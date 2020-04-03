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
import fnmatch
import fiona
import geopandas as gpd
import time

API_KEY = "AIzaSyB8rJhDsfwvIod9jVTfFm1Dtv2eO4QWqxQ"
GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/geocode/json"

debug = False
DETECTOR_ID_NAME = 'Detector_Id'
ROAD_ID_NAME = 'Road_Id'  # eid in aimsum network

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

        analysis = array[j + 6 * 26]  # some analysis, not useful for now
        

        #For each small table, check whether the summation of all the rows matches the top "sum" row 
        # We check the match by column. And I set an upper bound of 2 for each column of the table
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
        
        # For each table, I check whether the actual summation of all count matches the expected sum 
        expected_total_sum = sum(expected_sum)
        real_total_sum = int(info_day[2].split("=")[1])
        #set the boundary for the test: arbitrarily set it to be 15
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
        try:
            processed_frame = parse_excel_2013(dfs)
        except ValueError:
            print(file_path)
        return processed_frame
        
    if year==2015:
        try:
            processed_frame = parse_excel_2015(dfs)
        except ValueError:
            print(file_path)
        return processed_frame
        
    if year==2017:
        try:
            processed_frame = parse_excel_2017(dfs)
        except ValueError:
            print(file_path)
        return processed_frame
        
    if year==2019:
        try:
            processed_frame = parse_excel_2019(dfs)
        except ValueError:
            print(file_path)
        return processed_frame


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
    # standardize the table header
    tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB', 'EB', 'WB']]
    
    # remove all the columns that are all NaN values
    cols = ['NB', 'SB', 'EB', 'WB']
    to_be_removed = []
    for col in cols:
        if tmp_df_2[[col]].isnull().all().all():
            to_be_removed.append(col)
    main_list = ['Date'] + list(set(cols) - set(to_be_removed))
    tmp_df_2 = tmp_df_2[main_list]
    #only keep the non-trivial direction: eg, if this detector only detects "EB", then "WB" is also gonna be removed. 
    tmp_df_2.dropna(subset=list(set(cols) - set(to_be_removed)), inplace=True)


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
    #tmp_df_2 = tmp_df_2.loc[:, (tmp_df_2 != 0).any(axis=0)]
    
    #for 2015, all the flows have two directions --> we remove the other two directions by checking whether the entire columns for both directions are NaN.
    if tmp_df_2[['NB', 'SB']].isnull().all().all():
        tmp_df_2 = tmp_df_2[['Date', 'EB', 'WB']]
        
    elif tmp_df_2[['WB', 'EB']].isnull().all().all():
        tmp_df_2 = tmp_df_2[['Date', 'NB', 'SB']]
        
    else:
        raise ValueError('this 2015 file has four non-NaN directions')
        
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


    cols = list(tmp_df_2.columns.values)
    tmp_df_2 = tmp_df_2[[cols[-1], cols[2], cols[3]]]
    #for 2017 raw files, the columns for empty directions are not NaN but zeros. So, the script checks and removes all all-zero columns
    tmp_df_2 = tmp_df_2.loc[:, (tmp_df_2 != 0).any(axis=0)]
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
    #for 2019 raw files, the columns for empty directions are not NaN but zeros. So, the script checks and removes all all-zero columns
    tmp_df_2 = tmp_df_2.loc[:, (tmp_df_2 != 0).any(axis=0)]
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
        #input_files_excel = os.listdir(input_folder_excel)
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
        _, file_ext = os.path.splitext(file_name)
        is_folder = os.path.isdir(os.getcwd() + '/' + file_name)
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
    base_file_name, file_ext = os.path.splitext(file_name)
    if ('$' not in file_name and '.DS_Store' not in file_name):
        # read excel data into dataframe
        xls_path =  in_folder + base_file_name + '.xls'
        if not os.path.exists(xls_path):
            xls_path = in_folder + base_file_name + '.xlsx'

        xls_file = pd.ExcelFile(xls_path)
        dfs = {}
        for sheet_name in xls_file.sheet_names:
            dfs[sheet_name] = xls_file.parse(sheet_name)

        # get main road info (to see respective columns do a print)
        Input = dfs['Input']
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
    # check whether this is a legit raw file
    if ('$' not in file_name and '.DS_Store' not in file_name and not is_folder):
        if debug:
            print(file_name)
        # extract the main road and crossroad information from the file name by splitting
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
    base_file_name, file_ext = os.path.splitext(file_name)
    name = base_file_name.title()
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

        # remove_direction
        cross1 = (cross_road.split('And')[0].replace('Nb', '').replace('Sb', '') \
                  .replace('Eb', '').replace('Wb', '')).strip()
        cross2 = (cross_road.split('And')[1].replace('Nb', '').replace('Sb', '') \
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
        cross1 = cross_road.replace('Nb', '').replace('Sb', '') \
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
    base_file_name, _ = os.path.splitext(file_name)
    name = base_file_name.title()
    city = 'Fremont'

    for splitter in ['Bt', 'Bet.']:
        if splitter in name:
            bt = splitter

    # bt = find_splitter(name, ['Bt', 'Bet.'])
    main_road = name.split(bt)[0].strip()
    cross_road = name.split(bt)[1].strip()

    for splitter in ['And', '&']:
        if splitter in cross_road:
            And = splitter
    # And = find_splitter(cross_road, ['And', '&'])
    cross1 = (cross_road.replace('Nb', '').replace('Sb', '') \
              .replace('Eb', '').replace('Wb', '')).split(And)[0].strip()
    cross2 = (cross_road.replace('Nb', '').replace('Sb', '') \
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


def flow_processed_generator_city(processed_dir, output_dir, raw_2013_folder, detectors_to_roads_dir):
    """
    Creates file flow_processed_city.csv and year_info.csv for year=[2013, 2015, 2017, 2019] 
      in the given output_dir directory. Each row in flow_processed_city.csv is a road section
      and the columns contain meta data and flow data for that road. Each row in year_info.csv
      is a road section and the columns are meta data and geographical data (ie longitude and latitude).

    :param processed_dir: directory containing folders 'year processed' for each year
        and those folders contain flow data in .csv files (one csv file per road section)
    :param output_dir: output directory for created files
    :param raw_2013_folder: directory for 2013 raw flow data files
    :param detectors_to_roads_dir: directory for detectors_to_road_segments_year.csv files which contain
        spatial join results to obtain road id for a given detector id
    :return: creates files as described above 
    """
    print('Creating flow_processed_city.csv')
    # read detectors to road segments files for all years
    years = ['2013', '2015', '2017', '2019']
    dic_year_detector_to_road = {}
    for year in years:
        dic_year_detector_to_road[year] = pd.read_csv(detectors_to_roads_dir +
                                                        'detectors_to_road_segments_' + year + '.csv')

    # lines to be written to csv
    csv_lines = []

    # create legend
    # note: road_id is the eid from aimsum network, calling it road_id for user clarity
    legend = ['Name', 'Direction', 'Detector_Id', 'Road_Id', 'Year', 'Day 1']
    days, hours, timestep = 3, 24, 15
    for day in range(days):
        for hr in range(hours):
            for min in range(0, 60, timestep):
                legend.append('Day %s - %s:%s' % (day + 1, hr, min))
    csv_lines.append(legend)

    # read flow data from processed folders 'year processed'
    all_directions = ['NB', 'SB', 'EB', 'WB']
    year_file_name_to_detector_id = {}
    for year in years:
        processed_folder = processed_dir + ('%s processed' % year)
        files = os.listdir(processed_folder)
        files = [f for f in files if fnmatch.fnmatch(f, '*.csv')]
        files.sort()
        detector_count = 1
        for _, file_name in enumerate(files):
            # read data
            flow_data_df = pd.read_csv(processed_folder + '/' + file_name)
            flow_data_df = flow_data_df.dropna()

            # get directions
            directions = [col for col in flow_data_df.columns if col in all_directions]
            if not directions:
                # if direction not found, attempt to find it in file name
                base_name = os.path.splitext(file_name)[0]  # remove .csv extension
                directions = [word.strip() for word in base_name.split(' ') if word.strip() in all_directions]
                if not directions:
                    raise (ValueError('Cant find direction column in file: ' + file_name))

            # get day 1
            day_1 = flow_data_df['Date'][0].split(' ')[0]

            # create csv line for each direction
            for direction in directions:
                # create detector id by counting
                detector_id = '0' + str(detector_count) if detector_count < 10 else str(detector_count)
                detector_id = year + detector_id  # we desire to create a detector id in this format
                detector_count += 1

                # save detector id and its direction dictionary with (file, year) hash
                if (year, file_name) not in year_file_name_to_detector_id:
                    year_file_name_to_detector_id[(year, file_name)] = []
                year_file_name_to_detector_id[(year, file_name)].append((detector_id, direction))

                # get road id by using spatial join results
                road_id = ''
                year_detector_to_road_df = dic_year_detector_to_road[year]
                road_matches_df = year_detector_to_road_df[
                    int(detector_id) == year_detector_to_road_df[DETECTOR_ID_NAME]
                ]
                if not road_matches_df.empty:
                    road_ids = [str(dr[ROAD_ID_NAME]) for _, dr in road_matches_df.iterrows()]
                    road_id = ' - '.join(road_ids)

                # name, direction, detector id, road id, year, day 1
                csv_line = [file_name, direction, detector_id, road_id, year, day_1]

                # flow data from day 1 - 0:0 to day 3 - 23:45
                if direction in flow_data_df.columns:
                    csv_line.extend(flow_data_df[direction].values)
                else:
                    csv_line.extend(flow_data_df['Count'].values)

                csv_lines.append(csv_line)

    # create csv output file
    flow_processed_city = open(output_dir + 'flow_processed_city.csv', 'w')
    for line in csv_lines:
        line = ','.join(str(x) for x in line)
        flow_processed_city.write(line + '\n')
    flow_processed_city.close()

    """
    Code to create year_info.csv files
    """
    # now create year_info.csv for year=[2013, 2015, 2017, 2019]
    for year in years:
        print('Creating %s_info.csv' % year)
        csv_lines = []  # lines to be written to csv

        # get files for this year
        processed_folder = processed_dir + ('%s processed' % year)
        files = os.listdir(processed_folder)
        files = [f for f in files if fnmatch.fnmatch(f, '*.csv')]

        # get main road info for address lookup to get lat and long
        for file_name in files:
            if year == '2013':
                main_road_info = get_main_road_info_2013(raw_2013_folder, file_name)
            elif year == '2015':
                main_road_info = get_main_road_info_2015(file_name)
            elif year == '2017':
                main_road_info = get_main_road_info_2017(file_name)
            elif year == '2019':
                main_road_info = get_main_road_info_2019(file_name)
            else:
                raise (Exception('Unable to get main road info for file: %s' % file_name))

            file_name, city, main_road, cross_road, cross1, cross2 = main_road_info
            detector_ids_and_directions = year_file_name_to_detector_id.get((year, file_name))
            for detector_id, direction in detector_ids_and_directions:
                # get lat and long coordinates for detector using its address (main road info)
                csv_line = None
                if cross1 and cross2:
                    lat1, lng1 = get_coords_from_address(main_road + ' & ' + cross1 + ', ' + city)
                    lat2, lng2 = get_coords_from_address(main_road + ' & ' + cross2 + ', ' + city)
                    csv_line = [file_name, city, detector_id, direction, main_road, cross_road, str(lat1), str(lng1),
                                str(lat2), str(lng2)]
                elif cross1:
                    lat, lng = get_coords_from_address(main_road + ' & ' + cross1 + ', ' + city)
                    csv_line = [file_name, city, detector_id, direction, main_road, cross_road, str(lat), str(lng)]
                else:
                    raise (Exception('Unable to get coordinates for main road of file %s' % file_name))
                csv_lines.append(csv_line)

        # create year_info.csv for this year
        year_info_csv = open(output_dir + year + '_info.csv', 'w')
        legend = ['Name', 'City', DETECTOR_ID_NAME, 'Direction', 'Main road', 'Cross road',
                  'Start lat', 'Start lng', 'End lat','End lng']
        year_info_csv.write(','.join(legend) + '\n')
        for line in csv_lines:
            year_info_csv.write(','.join(line) + '\n')
        year_info_csv.close()


def flow_processed_generator_pems(processed_dir, output_dir, detectors_to_roads_dir):
    """
    Creates file flow_processed_pems.csv in output_dir directory by parsing flow data in processed_dir folder.

    :param processed_dir: directory containing folders 'PeMS_year' for year=2013, 2015, 2017, 2019
        and those folders contain flow data in .xlsx files (one file per road section)
    :param output_dir: directory for output file flow_processed_pems.csv
    :param detectors_to_roads_dir: directory for detectors_to_road_segments_pems.csv file which contain
        spatial join results to obtain road id for a given detector id
    :return: creates flow_processed_pems.csv as described above
    """
    print('Creating flow_processed_pems.csv')
    csv_lines = []  # lines to be written to csv
    # threshold, if sum of flow data for a given detector is less or equal to threshold dont include it in output file
    flow_sum_threshold = 0

    # load spatial join results from detectors_to_road_segments_pems.csv
    detector_to_road_df = pd.read_csv(detectors_to_roads_dir + 'detectors_to_road_segments_pems.csv')

    # create legend
    legend = ['Year', 'Name', DETECTOR_ID_NAME, ROAD_ID_NAME, '%Observed', 'Day 1', 'Flow Sum']
    days, hours, timestep = 3, 24, 5
    for day in range(days):
        for hr in range(hours):
            for min in range(0, 60, timestep):
                legend.append('Day %s - %s:%s' % (day + 1, hr, min))
    csv_lines.append(legend)

    # read flow data from processed folders 'year processed'
    years = ['2013', '2015', '2017', '2019']
    for year in years:
        # get xls data files
        pems_year_folder = processed_dir + 'PeMS_' + year
        files = os.listdir(pems_year_folder)
        files = [f for f in files if fnmatch.fnmatch(f, '*.xlsx') and '$' not in f]
        for file_name in files:
            # get name and detector id
            base_name = os.path.splitext(file_name)[0]  # remove .xlsx
            detector_id = base_name.split('_')[0]
            name = 'PeMS Detector ' + detector_id

            # get road id
            road_id = ''
            road_matches_df = detector_to_road_df[int(detector_id) == detector_to_road_df[DETECTOR_ID_NAME]]
            if not road_matches_df.empty:
                road_ids = [str(road[ROAD_ID_NAME]) for _, road in road_matches_df.iterrows()]
                road_id = ' - '.join(road_ids)

            # get observed, day1 and flow data
            data_flow_df = pd.read_excel(pems_year_folder + '/' + file_name)
            if not data_flow_df.empty:  # don't include detectors with empty data
                # get observed, day1, flow, flow sum
                observed = data_flow_df['% Observed'][0]
                day1 = str(data_flow_df['5 Minutes'][0]).split(' ')[0]
                flow = data_flow_df['Flow (Veh/5 Minutes)'].values
                flow_sum = sum(flow)
                if flow_sum > flow_sum_threshold:
                    # create and append csv line
                    csv_line = [year, name, detector_id, road_id, observed, day1, flow_sum]
                    csv_line.extend(flow)
                    csv_lines.append(csv_line)

    # create csv output file
    flow_processed_pems = open(output_dir + 'flow_processed_pems.csv', 'w')
    for line in csv_lines:
        line = ','.join(str(x) for x in line)
        flow_processed_pems.write(line + '\n')
    flow_processed_pems.close()


def create_aimsum_flow_processed_files(flow_dir, output_dir):
    """
    Creates 4 csv files from parsing flow_processed_pems.csv and flow_processed_city.csv
    The created files correspond to flow data for a given year and are named flow_processed__year.csv 
        where year=(2013, 2015, 2017, 2019).
    The files created are for the aimsum team, the created file contains the flow data
        averaged over the 3 days of recording.

    :param flow_dir: folder containing Flow_processed/flow_processed_city and PeMS/flow_processed_pems.csv
    :return: creates files as described above
    """
    # read city and pems flow data
    city_dir = flow_dir + 'Flow_processed/'
    pems_dir = flow_dir + 'PeMS/'
    flow_processed_city_df = pd.read_csv(city_dir + 'flow_processed_city.csv')
    flow_processed_pems_df = pd.read_csv(pems_dir + 'flow_processed_pems.csv')

    # find starting column idx of flow data in the files
    start_idx_city_flow = list(flow_processed_city_df.columns).index('Day 1 - 0:0')
    start_idx_pems_flow = list(flow_processed_pems_df.columns).index('Day 1 - 0:0')

    # parse flow data into dic per year
    year_to_flow_data_dic = {}
    one_day_flow_size = 24 * 60 / 15 * 3  # 24 hrs, 15min timesteps in one hr, 3 days

    # parse city data first
    for _, row in flow_processed_city_df.iterrows():
        # get name, detector id, road id, year, flow data
        name = row['Name']
        detector_id = row[DETECTOR_ID_NAME]
        direction = row['Direction']
        road_id = row[ROAD_ID_NAME]
        year = row['Year']
        flow_data = row.to_numpy()[start_idx_city_flow:]
        # take flow data average over the 3 days
        flow_data = flow_data.reshape((3, int(one_day_flow_size / 3)))
        flow_data = np.nanmean(flow_data, axis=0)

        if year not in year_to_flow_data_dic:
            year_to_flow_data_dic[year] = []

        csv_line = [name, direction, detector_id, road_id, year] + list(flow_data)
        year_to_flow_data_dic[year].append(csv_line)

    # parse pems data
    for _, row in flow_processed_pems_df.iterrows():
        # get name, id, year, flow data
        name = row['Name']
        detector_id = row[DETECTOR_ID_NAME]
        road_id = row[ROAD_ID_NAME]
        year = row['Year']
        flow_data = row.to_numpy()[start_idx_pems_flow:]

        # pems has 5 min timesteps, format it to 15 min timesteps
        flow_data = flow_data.reshape((int(flow_data.shape[0] / 3), 3))
        flow_data = np.nansum(flow_data, axis=1)

        # take flow data average over the 3 days
        flow_data = flow_data.reshape((3, int(one_day_flow_size / 3)))
        flow_data = np.nanmean(flow_data, axis=0)

        # dummy direction
        direction = ''

        csv_line = [name, direction, detector_id, road_id, year] + list(flow_data)

        if year not in year_to_flow_data_dic:
            year_to_flow_data_dic[year] = []

        year_to_flow_data_dic[year].append(csv_line)

    # write to output csv file (one csv per year)
    for year, flow_data in year_to_flow_data_dic.items():
        output_filename = ('flow_processed_%s.csv' % year)
        print('Creating ' + output_filename)
        output = open(output_dir + output_filename, 'w')
        # create legend
        legend = ['Name', 'Direction', DETECTOR_ID_NAME, ROAD_ID_NAME,'Year']
        for hr in range(24):
            for minute in range(0, 60, 15):
                legend.append('%s:%s' % (hr, minute))

        # write to file
        output.write(','.join(legend) + '\n')
        for line in flow_data:
            for i in range(len(line)):
                if pd.isnull(line[i]):
                    line[i] = ''
            output.write(','.join(str(x) for x in line) + '\n')
        output.close()

    # format the files as needed for aimsum software
    create_aimsum_output(flow_dir, output_dir)


def create_aimsum_output(flow_dir, output_dir):
    """
    Creates aimsum output files in desried format. Format has 3 columns: name, count, time
        where name is 'detector' + detector_id, count is car flow count, and time is the timestep for the flow count.

    :param flow_dir: folder containing flow_processed_year.csv flow data files where year
        takes values of 2013, 2015, 2017, 2019.
    :param output_dir: folder to place the created output files
    """
    years = ['2013', '2015', '2017', '2019']

    for year in years:
        flow_df = pd.read_csv(flow_dir + 'flow_processed_%s.csv' % year)

        # unroll all flows, place them into a list
        unrolled_flows = []
        start_idx = list(flow_df.columns).index('0:0')
        for i, row in flow_df.iterrows():
            name = row['Name'].lower()
            detector_id = str(int(row[DETECTOR_ID_NAME]))
            if 'pems' in name:
                detector_id = 'pem_detector' + detector_id
            else:
                detector_id = 'detector' + detector_id

            for j in range(start_idx, flow_df.shape[1]):
                timestep = flow_df.columns.values[j]
                if timestep[-2:] == ':0':
                    timestep += '0'
                unrolled_flows.append([detector_id, flow_df.iloc[i, j], timestep])

        # write file
        output_filename = 'aimsum_flow_processed_%s.csv' % year
        print('Creating ' + output_filename)
        output = open(output_dir + output_filename, 'w')
        output.write('name,count,time\n')
        for flow in unrolled_flows:
            output.write(','.join(str(x) for x in flow) + '\n')
        output.close()


def change_detector_ids_in_shape_files(detectors_dir, flow_processed_dir, output_dir):
    """
    Creates copy of shp files with new detector ids from flow_processed_city.csv for all 4 years.
    
    :param detectors_dir: folder containing all shp files for all years 
    :param flow_processed_dir: folder containing flow_processed_city.csv
    :param output_dir: output folder for the new updated shp files
    """
    flow_processed_city_df = pd.read_csv(flow_processed_dir + 'flow_processed_city.csv')

    years = ['2013', '2015', '2017', '2019']
    for year in years:
        filename = 'location_%s_detector.shp' % year
        shape_file = detectors_dir + filename
        output_file = output_dir + filename
        print('Creating copy of shape file with new detector ids: ' + filename)

        # correcting directions
        map_directions_2019 = {
            ('S Grimmer Blvd Bet. Osgood Rd & Fremont Blvd.csv', 'NB'): 'WB',
            ('S Grimmer Blvd Bet. Osgood Rd & Fremont Blvd.csv', 'SB'): 'EB',
            ('S Grimmer Blvd Bet. Paseo Padre Pkwy & OSgood Rd.csv', 'NB'): 'WB',
            ('S Grimmer Blvd Bet. Paseo Padre Pkwy & OSgood Rd.csv', 'SB'): 'EB',
            ('Warre Ave Bet. Warm Springs Blvd & Navajo Rd.csv', 'NB'): 'EB',
            ('Warre Ave Bet. Warm Springs Blvd & Navajo Rd.csv', 'SB'): 'WB',
            ('Warrent Ave Bet. Navajo Rd & Curtner Rd.csv', 'NB'): 'EB',
            ('Warrent Ave Bet. Navajo Rd & Curtner Rd.csv', 'SB'): 'WB',
            ('Washington Blvd Bet. Driscoll Rd & Paseo Padre Pkwy.csv', 'NB'): 'WB',
            ('Washington Blvd Bet. Driscoll Rd & Paseo Padre Pkwy.csv', 'SB'): 'EB',
            ('Washington Blvd Bet. Paseo Padre Pkwy & Mission Blvd.csv', 'NB'): 'WB',
            ('Washington Blvd Bet. Paseo Padre Pkwy & Mission Blvd.csv', 'SB'): 'EB',
        }

        flow_city_road_names = flow_processed_city_df['Name'].str.lower()

        # read and write shape file with fiona
        with fiona.collection(shape_file, 'r') as input:
            schema = input.schema.copy()
            with fiona.collection(output_file, 'w', 'ESRI Shapefile', schema, crs=input.crs) as output:
                for row in input:
                    # create copy of row to avoid segmentation error
                    row_copy = row.copy()
                    # get Name and Direction original shape row data
                    # and use it to get the new detector id in processed city data
                    road_name = row['properties']['Name']
                    direction = row['properties']['Direction']
                    road_name = os.path.splitext(road_name)[0] + '.csv'  # remove extension and add .csv
                    matches = flow_processed_city_df[
                        (flow_city_road_names == road_name.lower()) &
                        (flow_processed_city_df['Direction'] == direction)
                    ]

                    # if a match is found edit the detector id of row and write it to output
                    if not matches.empty:
                        match = matches.iloc[0]
                        row_copy['properties']['Id'] = str(match[DETECTOR_ID_NAME])
                        output.write(row_copy)
                    elif year == '2019':
                        direction = map_directions_2019[(road_name, direction)]
                        matches = flow_processed_city_df[
                            (flow_city_road_names == road_name.lower()) &
                            (flow_processed_city_df['Direction'] == direction)
                            ]
                        if not matches.empty:
                            match = matches.iloc[0]
                            row_copy['properties']['Id'] = str(match[DETECTOR_ID_NAME])
                            row_copy['properties']['Direction'] = direction
                            output.write(row_copy)
                        else:
                            print('no match:', year, road_name, direction)
                    else:
                        print('no match:', year, road_name, direction)
                        # no match, thus we assume directions are wrong
                        # that is, we map NB <-> EB and SB <-> WB
                        # if year == '2017':
                        #     direction = direction_remap_2017[direction]
                        # else:
                        #     direction = direction_remap[direction]
                        #
                        # matches = flow_processed_city_df[
                        #     (flow_city_road_names == road_name.lower()) &
                        #      (flow_processed_city_df['Direction'] == direction)
                        # ]
                        # if not matches.empty:
                        #     match = matches.iloc[0]
                        #     row_copy['properties']['Id'] = str(match[DETECTOR_ID_NAME])
                        #     output.write(row_copy)
                        # else:
                        #     print('no match for road name w direction:', road_name, direction)


                output.close()
                output.flush()
            input.close()
            input.flush()

    # read updated to csv to see new created shape file
    for year in years:
        filename = 'location_%s_detector.shp' % year
        output_file = output_dir + filename
        streetline_df = gpd.GeoDataFrame.from_file(output_file)
        # streetline_df = streetline_df.to_crs(epsg=4326)
        streetline_df.to_csv(output_dir + 'detectors_%s.csv' % year)


def google_doc_generater(Processed_dir):
    """
    This script generated the google doc file with the exact same (Id, file_name) pairs. 
    The function goes over year_info_coor.csv files and concatenate them into one large csv in consistent format
    flow_out.csv is the same as the google doc 

    """
    error_id = [12, 60, 61, 62, 63, 64]
    path = Processed_dir
    all_files = glob.glob(path + "/*.csv")
    li = [] #to build up list of dataframes to be concatenated later.
    id_counter = 1

    #iterate through all the processed year_info_coor.csv files 
    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        if curr_year in ['2013', '2017', '2019']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            #in order to match the google doc, sort the dataframe by the file_name
            df = df.sort_values(df.columns[0], ascending = True)
            df = df.reset_index(drop=True)
            ids = []
            for i in range(len(df['Name'])):
                file_name = df['Name'][i]
                ids.append(id_counter)
                #if there is direction in file_name (one direction file), we increment the id_counter only by one
                if ('EB' in file_name) or ('WB' in file_name) or ('NB' in file_name) or ('SB' in file_name):
                    id_counter += 1
                    while id_counter in error_id:
                        id_counter += 1
                else:
                    #if there is no direction information in file_name, we by default assume that there are two direction associated with this detector (on this road)
                    id_counter += 2
                    #we first increment by 2. But if id_counter is in error_id, we keep incrementing until it's not. 
                    while id_counter in error_id:
                        id_counter += 1
            df.insert(0, 'Id', ids)
            #update the header of this dataframe
            header = {'Id':'Id', 'Name':curr_opening, 'Main road':'Main road', 'Cross road':'Cross road', 'Start lat':'Start lat', 'Start lng':'Start lng', 'End lat':'End lat', 'End lng':'End lng'}
            df['Id'] = ids
            df = df.rename(columns={'Name': curr_opening})
            li.append(df)
    
    # For PeMS:
    # read through Flow_processed_all.csv and extract the "PeMS" section to get the ordering of PeMS detector ID that matches the google doc (need to find out other ways)
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

    #repeat the process for 2015 at the last to match the ordering in google doc 
    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        if curr_year in ['2015']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df.sort_values(df.columns[0], ascending = True)
            df = df.reset_index(drop=True)
            # as each file in 2015 contains two directions. So, we count every file twice when constructing the IDs and google docs. 
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

    #concatenated the list of dataframes and write to a csv file

    li[0].to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False)
    li[1].to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')
    li[2].to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')
    li[3].to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')
    li[4].to_csv(Processed_dir + "/flowing_out.csv", encoding='utf-8', index=False, mode='a')

def flow_processed_generater(Processed_dir):
    """
    *** OLD VERSION ***
    *** New Version see below ***
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


def flow_processed_generater1(path, re_formated_Processed_dir):
    """
    *** NEW Version that merges flow_processed_generater and process_flow.process_data_City ***

    This script create an updated version of Flow_processed_tmp.csv and generating (re_formated_Processed_dir + '/Flow_processed_city.csv')
    """
    error_id = [12, 61, 62, 63, 64]
    all_files = glob.glob(path + "/*.csv")
    li = []
    id_counter = 1
    to_be_concatenated = [] # initialize a list of pandas dataframes that are transposed processed files

    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        # iterate through all processed year_info_coor.csv files
        if filename.split("/")[-1] in ['2013_info_coor.csv', '2017_info_coor.csv', '2019_info_coor.csv']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df.sort_values(df.columns[0], ascending = True)
            #sort the dataframe by file_name to match the google doc
            df = df.reset_index(drop=True)
            curr_year_dir = re_formated_Processed_dir + "/" + str(curr_year) + " processed"
            ids = [] #list of IDs for tmp file
            for i in range(len(df['Name'])):
                ids.append(id_counter)

                #extract the file_name (detector name) from year_info_coor file and find the processed csv file for that detector
                if os.path.splitext(df['Name'][i])[-1] in ['.xlsx', '.xls', '.pdf', '.doc']:
                    file_name = ''.join(os.path.splitext(df['Name'][i])[0:-1]) + '.csv'
                else:
                    file_name = df['Name'][i] + '.csv'  #os.path.splitext(df['Name'][i])[0] + '.csv'
                #read the processed csv file for this detector
                input_file = pd.read_csv(curr_year_dir + "/" + file_name)
                file_day_one = input_file['Date'].apply(lambda x: x.split(" ")[0])[0]
                #extract the Date, year, etc information from the processed csv

                # if
                if (('WB' in file_name) or ('EB' in file_name) or ('NB' in file_name) or ('SB' in file_name)):

                    if 'Count' in input_file.columns: #check whether the raw data comes from a doc file
                        one_direction = input_file.transpose().iloc[[1], :]
                        file_direction = file_name.split(".")[0].split(" ")[-1] #if from doc, then the direction info would be at the last of the file_name
                    elif len(input_file.columns) == 3: # if not from doc file, then file_direction is at the first
                        one_direction = input_file.transpose().iloc[[2], :]
                        file_direction = file_name.split(".")[0].split(" ")[0]
                    else:
                        raise ValueError("check with " + str(file_name))
                    flow_information = one_direction
                    flow_information['Id'] = id_counter
                    id_counter += 1
                    while id_counter in error_id:
                        id_counter += 1
                    flow_information['Direction'] = file_direction #two_directions.index.tolist()
                else: #if no direction info in the file_name, then we by default assume two directions in the file
                    two_directions = input_file.transpose().iloc[[2, 3], :]
                    flow_information = two_directions
                    flow_information['Id'] = [id_counter, id_counter + 1]
                    id_counter += 2
                    while id_counter in error_id:
                        id_counter += 1
                    flow_information['Direction'] = two_directions.index.tolist()

                flow_information['year'] = curr_year
                flow_information['Name'] = file_name
                flow_information['Day 1'] = file_day_one
                to_be_concatenated.append(flow_information)
            df.insert(0,'Id',ids)
            #reset the header of the dataframe
            header = {'Id':'Id', 'Name':curr_opening, 'Main road':'Main road', 'Cross road':'Cross road', 'Start lat':'Start lat', 'Start lng':'Start lng', 'End lat':'End lat', 'End lng':'End lng'}
            df['Id'] = ids
            df = df.rename(columns={'Name': curr_opening})
            df =  df[['Id', curr_opening]]
            li.append(df)

    # For PeMS:
    #similar code from before. Getting the PeMS section from Flow_processed_all.csv and process. We might need better ways to do it.
#     PeMS_file = path + "/" + 'Flow_processed_all.csv'
#     df = pd.read_csv(PeMS_file, index_col=None, header=0)
#     PeMS_section = df[df['Name'].apply(lambda x: x.split(" ")[0] == "PeMS")]
#     PeMS_section = PeMS_section[PeMS_section['Year'].apply(lambda x: x == 2013)]
#     PeMS_section["Name"] = PeMS_section["Name"].apply(lambda x: x.split(" ")[-1])
#     ids = []
#     for i in range(len(PeMS_section)):
#         detector_id = PeMS_section.iloc[i]
#         ids.append(id_counter)
#         id_counter += 1
#     PeMS_section = PeMS_section[["Name"]]
#     PeMS_section.insert(0,'Id',ids)
#     PeMS_section = PeMS_section.rename(columns={'Name': "PeMS"})
#     li.append(PeMS_section)

    #for 2015 --> similar formatting as 2013, 2017 and 2019 but by default assume two direction for every file
    for filename in all_files:
        curr_year = (filename.split("/")[-1]).split("_")[0]
        curr_year_dir = path + "/" + str(curr_year) + " processed"
        # print("currently parsing: ...")
        # print(filename.split("/")[-1])

        curr_year = (filename.split("/")[-1]).split("_")[0]
        if filename.split("/")[-1] in ['2015_info_coor.csv']:
            curr_opening = "./" + curr_year + " ADT Data"
            df = pd.read_csv(filename, index_col=None, header=0)
            df = df.sort_values(df.columns[0], ascending = True)
            df = df.reset_index(drop=True)
            r = np.arange(len(df)).repeat(2)
            df = pd.DataFrame(df.values[r], df.index[r], df.columns)
            df = df.reset_index(drop=True)
            ids = []
            for i in range(len(df['Name'])):
                if i % 2 == 0:
                    ids.append(id_counter)
                    ids.append(id_counter+1)
                    if os.path.splitext(df['Name'][i])[-1] in ['.xlsx', '.xls', '.pdf', '.doc']:
                        file_name = ''.join(os.path.splitext(df['Name'][i])[0:-1]) + '.csv'
                    else:
                        file_name = df['Name'][i] + '.csv'  #os.path.splitext(df['Name'][i])[0] + '.csv'
                    input_file = pd.read_csv(curr_year_dir + "/" + file_name)
                    file_day_one = input_file['Date'][0].split(" ")[0]

                    two_directions = input_file.transpose().iloc[[2, 3], :]
                    flow_information = two_directions
                    flow_information['Id'] = [id_counter, id_counter + 1]
                    id_counter += 2
                    flow_information['Direction'] = two_directions.index.tolist()
                    flow_information['year'] = curr_year
                    flow_information['Name'] = file_name
                    flow_information['Day 1'] = file_day_one
                    to_be_concatenated.append(flow_information)
                    file_name = df['Name'][i]
                #id_counter += 1
            df.insert(0,'Id',ids)
            header = {'Id':'Id', 'Name':curr_opening, 'Main road':'Main road', 'Cross road':'Cross road', 'Start lat':'Start lat', 'Start lng':'Start lng', 'End lat':'End lat', 'End lng':'End lng'}
            df['Id'] = ids
            df = df.rename(columns={'Name': curr_opening})
            df =  df[['Id', curr_opening]]
            li.append(df)

    #df1,df2, df3, df4, df5  = li[0], li[1], li[2], li[3], li[4]

    print(len(li))
    li[0].to_csv(re_formated_Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False)
    li[1].to_csv(re_formated_Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')
    li[2].to_csv(re_formated_Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')
    li[3].to_csv(re_formated_Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')
#     li[4].to_csv(re_formated_Processed_dir + "/processed_flow_tmp.csv", encoding='utf-8', index=False, mode='a')

    df_total = pd.concat(to_be_concatenated, ignore_index=True)
    #generate the legend for the dataframe: from Day1 0:00 to Day3 11:45pm.
    legend = []
    for k in range(3):
        for i in range(24):
            for j in range(4):
                legend = legend + ["Day " + str(k + 1) + " - " + str(i) + ":" + str(15 * j)]

    City_legend = legend + ['Id', 'Direction', 'year', 'Name', 'Day 1']

    #many of the processed files have many NaN rows at the tail. Then, we drop all the NaN columns (after transpose) first before reset column names and indexes.
    df_total = df_total.dropna(axis=1, how='all')
    df_total.reset_index(drop=True, inplace=True)
    df_total.columns = City_legend


    cols = df_total.columns.tolist()
    #reorder the columns to match the expected file
    cols = cols[-5:] + cols[0:-5]
    df_total = df_total[cols]
    df_total.to_csv(re_formated_Processed_dir + "/Flow_processed_city.csv")



def speed_data_parser(speed_data_dir, output_dir):
    """
    Creates 2015_Speed_Processed_Num.csv and 2015_Speed_Processed_Percent.csv using the speed data found in
    the speed_data_dir folder. The raw data is from Kimley_Horn_flow of year 2015

    :param speed_data_dir: directory containing raw speed data from Kimley_Horn_flow
    :param output_dir: output folder for created files
    """
    # data frames for the two output files
    df_num = pd.DataFrame()
    df_percent = pd.DataFrame()

    for f in os.listdir(speed_data_dir):
        if 'xls' in f:
            # read the excel into dataframe
            input_file_all = pd.read_excel(speed_data_dir + "/" + f, sheet_name = '#2').rename(columns={'City of Fremont':'speed'})
            length = len(input_file_all['Unnamed: 34'])
            direction_descr = input_file_all['Unnamed: 10'][15]  # 17K in the excel file
            speed_limit = input_file_all['Unnamed: 34'][72]  # 74AI
            start_time = input_file_all['Unnamed: 34'][length - 6]
            end_time = input_file_all['Unnamed: 34'][length - 5]

            # how i viewed the contents of the xls file
            # print('all cols', input_file_all.columns)
            # for col in input_file_all.columns:
            #     print('col name', col)
            #     print(input_file_all[col].values)

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
            input_file.insert(0,'Name', f)

            # add info
            input_file.insert(1, 'Detector_Id', '')
            input_file.insert(2, 'Road_Id', '')
            input_file.insert(3, 'StartTime', start_time)
            input_file.insert(4, 'EndTime', end_time)

            input_file.insert(5, 'Direction', '')
            input_file.insert(6, 'Speed limit', speed_limit)

            # add direction
            input_file.iloc[1, 5] = direction[0]
            input_file.iloc[3, 5] = direction[0]
            input_file.iloc[2, 5] = direction[1]
            input_file.iloc[4, 5] = direction[1]

            df_num = df_num.append(input_file.iloc[1:3])
            df_percent = df_percent.append(input_file.iloc[3:5])

    # write them to csv
    df_num.to_csv(output_dir + '/2015_Speed_Processed_Num.csv', index=False)
    df_percent.to_csv(output_dir + '/2015_Speed_Processed_Percent.csv', index=False)


def speed_processed_update(dir_flow_processed):
    """
    Updates 2015_Speed_Processed_Num.csv and 2015_Speed_Processed_Percent.csv with detector ids and road ids.
    The road ids are obtained from the spatial join result. Since similar work was done and compiled
    in flow_processed_city.csv, use the results in that file to get the detector and road ids.

    :param dir_flow_processed: directory containing the speed files and flow_processed_city.csv file
    """
    df_num = pd.read_csv(dir_flow_processed + '2015_Speed_Processed_Num.csv')
    df_percent = pd.read_csv(dir_flow_processed + '2015_Speed_Processed_Percent.csv')
    flow_df = pd.read_csv(dir_flow_processed + 'flow_processed_city.csv')

    # map speed road name to flow road name to obtain ids
    speed_to_flow_road = {
        "44 - Durham Rd - I-680 to Mission Blvd.xls": "Durham Rd betw. I-680 and Mission.csv",
        "61 - South Grimmer Blvd - Osgood Rd to Fremont Blvd.xls": "Grimmer Blvd (South) betw. Osgood and Fremont.csv",
        "60 - Grimmer Blvd - Paseo Padre Pkwy to Osgood Rd.xls": "Grimmer Blvd (South) betw. Paseo Padre and Osgood.csv",
        "84 - Mission Blvd - Durham Rd to Curtner Rd.xls": "Mission Blvd betw. Durham and Curtner.csv",
        "81 - Mission Blvd - Mission Rd to St Joseph Terrace.xls": "Mission Blvd betw. Mission Rd and St. Josephs.csv",
        "83 - Mission Blvd - Pine St to Durham Rd.xls": "Mission Blvd betw. Pine and Durham.csv",
        "82 - Mission Blvd - St Joseph Terr to Pine St.xls": "Mission Blvd betw. St. Josephs and Pine.csv",
        "106 - Paseo Padre Pkwy - Driscoll Rd to Washington Blvd - Resurvey.xls": "Paseo Padre Pkwy betw. Driscoll and Quema.csv",
        "108 - Paseo Padre Pkwy - Durham Rd to Onondaga Wy.xls": "Paseo Padre Pkwy betw. Durham and Onondaga.csv",
        "110 - Paseo Padre Pkwy - Mission Blvd to Curtner Rd.xls": "Paseo Padre Pkwy betw. Mission and Curtner.csv",
        "109 - Paseo Padre Pkwy - Onondaga Wy to Mission Blvd.xls": "Paseo Padre Pkwy betw. Onondaga and Mission.csv",
        "107 - Paseo Padre Pkwy - Washington Blvd to Durham Rd - Resurvey.xls": "Paseo Padre Pkwy betw. Quema and Durham.csv",
        "135 - Warren Avenue - Curtner Road to Warm Springs Boulevard.xls": "Warren Ave betw. Curtner and Warm Springs.csv",
        "139 - Washington Boulevard - Driscoll Road to Paseo Padre Pkwy.xls": "Washington Blvd betw. Driscoll and Paseo Padre.csv",
        "140 - Washington Boulevard - Paseo Padre Pkwy to Mission Blvd.xls": "Washington Blvd betw. Paseo Padre and Mission.csv"
    }

    direction_fix = {
        ("135 - Warren Avenue - Curtner Road to Warm Springs Boulevard.xls", "NB"): "EB",
        ("135 - Warren Avenue - Curtner Road to Warm Springs Boulevard.xls", "SB"): "WB",
        ("109 - Paseo Padre Pkwy - Onondaga Wy to Mission Blvd.xls", "EB"): "NB",
        ("109 - Paseo Padre Pkwy - Onondaga Wy to Mission Blvd.xls", "WB"): "SB",
        ("108 - Paseo Padre Pkwy - Durham Rd to Onondaga Wy.xls", "EB"): "NB",
        ("108 - Paseo Padre Pkwy - Durham Rd to Onondaga Wy.xls", "WB"): "SB"
    }

    # iterate over rows in num_df and add detector and road id from flow df
    for i, row in df_num.iterrows():
        direction = row['Direction']
        road_name = row['Name']

        # correcting directions
        if (road_name, direction) in direction_fix:
            direction = direction_fix[(road_name, direction)]

        flow_road = speed_to_flow_road[road_name]
        matches_df = flow_df[(flow_df['Name'] == flow_road) & (flow_df['Direction'] == direction)]
        if not matches_df.empty:
            match = matches_df.iloc[0]
            df_num.at[i, DETECTOR_ID_NAME] = match[DETECTOR_ID_NAME]
            df_num.at[i, ROAD_ID_NAME] = match[ROAD_ID_NAME]
        else:
            raise(Exception('No detector and road id match for %s %s' % (flow_road, direction)))

    # do the same for rows in percent_df
    for i, row in df_percent.iterrows():
        direction = row['Direction']
        road_name = row['Name']

        # correct directions
        if (road_name, direction) in direction_fix:
            direction = direction_fix[(road_name, direction)]

        flow_road = speed_to_flow_road[road_name]
        matches_df = flow_df[(flow_df['Name'] == flow_road) & (flow_df['Direction'] == direction)]
        if not matches_df.empty:
            match = matches_df.iloc[0]
            df_percent.at[i, DETECTOR_ID_NAME] = match[DETECTOR_ID_NAME]
            df_percent.at[i, ROAD_ID_NAME] = match[ROAD_ID_NAME]
        else:
            raise (Exception('No detector and road id match for %s %s' % (flow_road, direction)))

    # write the update dfs to csv
    df_num.to_csv(dir_flow_processed + '/2015_Speed_Processed_Num.csv', index=False)
    df_percent.to_csv(dir_flow_processed + '/2015_Speed_Processed_Percent.csv', index=False)


def test_create_aimsum_flow_processed_files():
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    flow_dir = data_process_folder + "Auxiliary files/Demand/Flow_speed/"
    create_aimsum_flow_processed_files(flow_dir, flow_dir)

def test_create_aimsum_output():
    dropbox_dir = '/Users/edson/Dropbox'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    flow_dir = data_process_folder + "Auxiliary files/Demand/Flow_speed/"
    create_aimsum_output(flow_dir, flow_dir)

def run_detectors_id_change():
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    detectors_folder = data_process_folder + "Raw/Demand/Flow_speed/detectors/"
    output_folder = data_process_folder + "Raw/Demand/Flow_speed/detectors id change/"
    flow_processed_dir = data_process_folder + "Auxiliary files/Demand/Flow_speed/Flow_processed/"
    change_detector_ids_in_shape_files(detectors_folder, flow_processed_dir, output_folder)

def test_flow_processed_generator_city():
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    data_process_folder = dropbox_dir + "/Private Structured data collection/Data processing/"
    raw_2013_city_folder = data_process_folder + "Raw/Demand/Flow_speed/City/2013 ADT Data/"
    detectors_to_roads_folder = data_process_folder + "Auxiliary files/Network/Infrastructure/Detectors/"
    flow_speed_folder = data_process_folder + "Auxiliary files/Demand/Flow_speed/"
    processed_city_folder = flow_speed_folder + "Flow_processed/"
    processed_pems_folder = flow_speed_folder + "PeMS/"
    output_folder = processed_city_folder
    flow_processed_generator_city(processed_city_folder, output_folder, raw_2013_city_folder,
                                         detectors_to_roads_folder)

def test_speed_data_parser():
    dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    ADT_dir = dropbox_dir + '/Private Structured data collection/Data processing/Raw/Demand/Flow_speed'
    Processed_dir = dropbox_dir + '/Private Structured data collection/Data processing/Auxiliary files/Demand/Flow_speed/Flow_processed'
    City_dir = ADT_dir + "/City"
    Kimley_Horn_flow_dir = ADT_dir + "/Kimley Horn Data"
    speed_data_parser(Kimley_Horn_flow_dir, Processed_dir)

def test_speed_data_update():
    # dropbox_dir = '/Users/edson/Fremont Dropbox/Theophile Cabannes'
    dropbox_dir = '/Users/edson/Dropbox'
    Processed_dir = dropbox_dir + '/Private Structured data collection/Data processing/Auxiliary files/Demand/Flow_speed/Flow_processed/'
    speed_processed_update(Processed_dir)

# for local testing only
def raise_exception():
    raise (Exception('stop code here'))

if __name__ == '__main__':
    #process_adt_data(2015)
    #process_doc_data(2019)
    #get_geo_data(2015)
    # test_create_aimsum_flow_processed_files()
    # run_detectors_id_change()
    # test_flow_processed_generator_city()
    # test_speed_data_parser()

    test_create_aimsum_output()

    # test_speed_data_update()
    pass
