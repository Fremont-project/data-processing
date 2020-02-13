import pandas as pd
import os
import requests
import textract
import numpy as np
from pathlib import Path

API_KEY = "AIzaSyB8rJhDsfwvIod9jVTfFm1Dtv2eO4QWqxQ"
GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/geocode/json"


from fremontdropbox_flow import get_dropbox_location
dropbox_dir = get_dropbox_location()
ADT_dir = dropbox_dir + '/Private Structured data collection/Data processing/Raw/Demand/Flow_speed/ADT'
local_download =  str(os.path.join(Path.home(), "Downloads"))
Processed_dir = dropbox_dir + '/Private Structured data collection/Data processing/Temporary exports to be copied to processed data/Flow_processed'
#directory to year reformat
City_dir = Processed_dir + "/" + "City"


"""
New generic method to process ADT data into csv files
"""
def process_adt_data(year):
    print("Processing %d ADT data" % year)
    output_folder = Processed_dir + "/" + "%d processed/" % year
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # parse excel files by iterating over them
    input_folder = ADT_dir + "/" + "%d ADT Data/" % year
    input_files = os.listdir(input_folder)
    for file_name in input_files:
        if is_excel_file(file_name):
            print("parsing:", file_name)
            xls_file = pd.ExcelFile(input_folder + file_name)
            dfs = {}
            for sheet_name in xls_file.sheet_names:
                dfs[sheet_name] = xls_file.parse(sheet_name)
            input_bis, main_road_info = None, None
            print(year == 2013)
            if year == 2013:
                input_bis = parse_excel_2013(dfs)
            elif year == 2015:
                input_bis = parse_excel_2015(dfs)
            elif year == 2017:
                input_bis = parse_excel_2017(dfs)
            elif year == 2019:
                input_bis = parse_excel_2019(dfs)
            else:
                raise(Exception("Unknown excel file format for year %d, "
                      "create a parser or edit code to accept it "
                      "if it has the same format of another year" % year))

            # write parsed results to csv file
            output_name = output_folder + remove_ext(file_name) + ".csv"
            print(output_name)
            input_bis.to_csv(output_name)


def parse_excel_2013(dfs):
    # for ease of parsing
    Input = dfs['Input']
    Input.to_csv("test_tmp.csv")

    # read to csv directly and select columns
    Input_bis = pd.read_csv("test_tmp.csv", skiprows=4)
    Input_bis = Input_bis[['Unnamed: 1', 'TIME', 'NB', 'SB', 'EB', 'WB']]
    Input_bis = Input_bis.rename(columns={'Unnamed: 1': 'Date'})

    return Input_bis







def parse_excel_2015(dfs):
    # for ease of parsing
    #parsing day1

    Input1 = process_input(dfs, 1)
    # Input1 = dfs['ew Day 1']
    # date1 = Input1.columns[::3][1].strftime("%Y-%m-%d")
    # Input1 = Input1.rename(columns={'Volumes for:': 'Time', 'Unnamed: 0': 'Date',
    #                                 Input1.columns[3]: 'morning_EB', Input1.columns[4]: 'afternoon_EB', Input1.columns[7]: 'morning_WB', Input1.columns[8]: 'afternoon_WB'})
    # morning1 = Input1[['Date', 'Time', 'morning_EB', 'morning_WB']]
    # morning1 = morning1.drop(morning1.index[[0, 1, 2]])
    # afternoon1 = Input1[['Date', 'Time', 'afternoon_EB', 'afternoon_WB']]
    # afternoon1 = afternoon1.drop(afternoon1.index[[0, 1, 2]])
    # morning1 = morning1.rename(columns={'morning_EB':'EB', 'morning_WB':'WB'})
    # afternoon1 = afternoon1.rename(columns={'afternoon_EB':'EB', 'afternoon_WB':'WB'})
    # morning1 = morning1.append(afternoon1, ignore_index=True)
    # Input1 = morning1
    # Input1['Date'] = date1

    #parsing day2
    Input2 = process_input(dfs, 2)
    #parsing day3
    Input3 = process_input(dfs, 3)#dfs['ew Day 3']
    
    Input1 = Input1.append(Input2, ignore_index=True)
    Input1 = Input1.append(Input3, ignore_index=True)
    return Input1

import math

def process_input(dfs, sheet_num): 
    """
    for ADT 2015 data
    example of the sheet_name would be 'ew Day1 1' or 'nw Day1'  
    if direction = 0: it's EB or WB
    if direction = 1: it's NB or SB
    """
    sheet_name = 'ew Day '+ str(sheet_num)
    Input = dfs[sheet_name]
    Input = Input[:-11]
    if math.isnan(Input.iloc[-1, 3]):
        direction = 1
        sheet_name = 'ns Day '+ str(sheet_num)
        Input = dfs[sheet_name]
    else:
        direction = 0
    date = Input.columns[::3][1].strftime("%Y-%m-%d") 

    if direction == 0:
        dir1 = 'EB'
        dir2 = 'WB'
    elif direction == 1:
        dir1 = 'NB'
        dir2 = 'SB'
    Input = Input.rename(columns={'Volumes for:': 'Time', 'Unnamed: 0': 'Date',
                                    Input.columns[3]: 'morning_'+ dir1, Input.columns[4]: 'afternoon_'+dir1, Input.columns[7]: 'morning_'+dir2, Input.columns[8]: 'afternoon_'+dir2})

    morning = Input[['Date', 'Time', 'morning_'+ dir1, 'morning_'+dir2]]
    morning = morning.drop(morning.index[[0, 1, 2]])
    afternoon = Input[['Date', 'Time', 'afternoon_'+dir1, 'afternoon_'+dir2]]
    afternoon = afternoon.drop(afternoon.index[[0, 1, 2]])
    morning = morning.rename(columns={'morning_'+ dir1: dir1, 'morning_'+ dir2:dir2})
    afternoon = afternoon.rename(columns={'afternoon_'+dir1:dir1, 'afternoon_'+dir2:dir2})
    morning['Time'] = morning['Time'].apply(lambda d : d.time())
    afternoon['Time'] = afternoon['Time'].apply(lambda d : d.time())
    morning = morning.append(afternoon, ignore_index=True)
    Input = morning
    Input['Date'] = date
    return Input


def parse_excel_2017(dfs):
    # for ease of parsing
    Input = dfs['Sheet1']
    Input.to_csv("test_tmp.csv")
    Input_bis = pd.read_csv("test_tmp.csv", skiprows=6)

    # drop useless col
    Input_bis = Input_bis.drop(['5'], axis=1)
    return Input_bis


def parse_excel_2019(dfs):
    # parse the excel sheets and concatenate their data
    Input1 = parse_sheet_helper_2019('Day 1', dfs)
    Input2 = parse_sheet_helper_2019('Day 2', dfs)
    Input3 = parse_sheet_helper_2019('Day 3', dfs)
    Input1 = Input1.append(Input2, ignore_index=True)
    Input1 = Input1.append(Input3, ignore_index=True)

    # rename columns accordingly
    # (matching was done visually printing the data frames and matching to xls data)
    # print(Input1)
    Input1 = Input1.rename(columns={'date': 'Date', 'Unnamed: 25': 'Time',
                                    'Unnamed: 26': 'NB', 'Unnamed: 27': 'SB',
                                    'Unnamed: 28': 'EB', 'Unnamed: 29': 'WB'})
    return Input1


def parse_sheet_helper_2019(name, dfs):
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
    _, file_ext = os.path.splitext(file_name)
    return (file_ext == '.xls' or file_ext == '.xlsx') and is_valid_file(file_name)


def is_valid_file(file_name):
    is_folder = os.path.isdir(os.getcwd() + '/' + file_name)
    return '$' not in file_name and '.DS_Store' not in file_name and not is_folder

# removes extension from file name
def remove_ext(file_path):
    file_name, _ = os.path.splitext(file_path)
    return file_name


def remove_direction(string):
    return string.replace('Nb', '').replace('Sb', '')\
        .replace('Eb', '').replace('Wb', '')


def get_of_direction(string):
    of_directions = ['S Of', 'E Of', 'W Of', 'N Of']
    for of in of_directions:
        if of in string:
            return of


def find_splitter(string, splitters):
    for splitter in splitters:
        if splitter in string:
            return splitter


"""
Process Doc files
Years 2017 and 2019 DOC files have the same structure hence we can reuse this code.
Structure refers to data organized in 3 tables split by *
Note no doc files for 2013.
"""
def process_doc_data(year):
    # create output folder to put results in
    out_folder = Processed_dir + '/' + '%d processed' % year
    #create_directory(os.getcwd(), out_folder)

    in_folder = Processed_dir + '/' + '%d doc' % year
    if not os.path.exists(in_folder):
        os.makedirs(in_folder)
    for file_name in os.listdir(in_folder):
        if is_doc_file(file_name):
            print("processing file:", file_name)
            # create csv file
            out_file = open(out_folder + '/' + remove_ext(file_name) + '.csv', 'w')
            out_file.write("Day,Time,Count\n")

            # data is structured in 3 tables split by *
            text = textract.process(in_folder + "/" + file_name)
            text = str(text).replace('\\n', ' ')
            data = text.split('*')
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
                    table_tmp[i] = np.array(data_row)
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
                        out_file.write(day + "," + time + "," + count + "\n")

            out_file.close()


def is_doc_file(filename):
    _, file_ext = os.path.splitext(filename)
    return file_ext == '.doc' and is_valid_file(filename)

def create_directory(path, name):
    dir = path + "/" + name
    if not os.path.isdir(dir):
        os.mkdir(dir)



def get_geo_data(year):
    print('Obtaining geo data from %d ADT files' % year)
    # input folder and files
    in_folder = "%d ADT Data/" % year
    input_files = os.listdir(in_folder)

    # iterate over the files to obtain main road addresses
    cache_main_roads = []
    for file_name in input_files:
        if is_valid_file(file_name):
            print("processing:", file_name)
            main_road_info = None
            if year == 2013:
                main_road_info = get_main_road_info_2013(in_folder, file_name)
            elif year == 2015:
                main_road_info = get_main_road_info_2015(file_name)
            elif year == 2017:
                main_road_info = get_main_road_info_2017(file_name)
            elif year == 2019:
                main_road_info = get_main_road_info_2019(file_name)
            else:
                raise (Exception('Unable to get main road info for file: %s' % file_name))

            print('main road info:', main_road_info)
            cache_main_roads.append(main_road_info)

    # get geo coordinates using google API and cache_main_roads
    # write the results in the csv file, 'year_info_coor.csv'
    coordinate_file = open('%d_info_coor.csv' % year, 'w')
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
    print(coordinate_file)
    coordinate_file.close()

"""
Gets main road info from inside the file 
"""

def get_main_road_info_2013(in_folder, file_name):
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
        print(file_name)
        main_road = file_name.split('betw.')[0].strip()
        cross = file_name.split('betw.')[1].strip()[:-4]
        cross1 = cross.split('and')[0].strip()
        cross2 = cross.split('and')[1].strip()
        city = 'Fremont'
        main_road_info = (file_name, city, main_road, cross, cross1, cross2)
        return main_road_info


"""
Get 2017 main road info from file name
input: file_name, examples below 
file_name = mission blvd BT driscoll rd AND I 680 NB
file_name = mission blvd S OF washington blvd signal
output = (file_name, city, main_road, cross_road, cross1, cross2)
"""

def get_main_road_info_2017(file_name):
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

"""
Get 2019 main road info from file name
input: file_name, examples below 
file_name = Driscoll Rd Bet. Mission Blvd & Paseo Padre Pkwy
file_name = AUTO MALL PKWY BT FREMONT BLVD AND I-680 EB
output = (file_name, city, main_road, cross_road, cross1, cross2)
"""

def get_main_road_info_2019(file_name):
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
    payload = {
        'address': address,
        'key': API_KEY
    }
    print('address: ', address)
    request = requests.get(GOOGLE_MAPS_URL, params=payload).json()
    results = request['results']

    lat = None
    lng = None

    if len(results):
        answer = results[0]
        lat = answer.get('geometry').get('location').get('lat')
        lng = answer.get('geometry').get('location').get('lng')
    print('address w coord lat, lng', address, str(lat), str(lng))
    return lat, lng



# for local testing only
if __name__ == '__main__':
    #process_adt_data(2015)
    #process_doc_data(2019)
    #get_geo_data(2015)
    pass
