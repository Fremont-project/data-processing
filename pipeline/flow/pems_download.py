import os
import webbrowser
import time
import requests
from pathlib import Path


"""
find relavant directories here
"""

debug = False
local_download =  str(os.path.join(Path.home(), "Downloads"))
if debug:
    print('Path to download folder: ' + str(local_download))

"""http://pems.dot.ca.gov/?report_form=1&dnode=VDS&content=loops&tab=det_timeseries&export=text&station_id=403250&s_time_id=1425340800&s_time_id_f=03%2F03%2F2015+00%3A00&e_time_id=1425599940&e_time_id_f=03%2F05%2F2015+23%3A59&tod=all&tod_from=0&tod_to=0&dow_0=on&dow_1=on&dow_2=on&dow_3=on&dow_4=on&dow_5=on&dow_6=on&holidays=on&q=flow&q2=&gn=5min&agg=on&lane1=on&lane2=on
Get url add-on for a given year

For every year, we go to this page: http://pems.dot.ca.gov/?report_form=1&dnode=VDS&content=loops&tab=det_timeseries&station_id=403250&s_time_id=1362441600&s_time_id_f=03%2F05%2F2013+00%3A00&e_time_id=1362700740&e_time_id_f=03%2F07%2F2013+23%3A59&tod=all&tod_from=0&tod_to=0&dow_0=on&dow_1=on&dow_2=on&dow_3=on&dow_4=on&dow_5=on&dow_6=on&holidays=on&q=flow&q2=&gn=5min&agg=on&lane1=on&lane2=onlane3=on
And pick the first Tuesday - Thursday of March for the dataset. 
"""
def time_for_year(year):
    """ To do """
    if year == 2013:
        url_addon = "&s_time_id=1362441600"
        url_addon = url_addon + "&s_time_id_f=" + "03" + "%2F" + "05" + "%2F" + "2013" + "+00%3A00"
        url_addon = url_addon + "&e_time_id=1362700740"
        url_addon = url_addon + "&e_time_id_f=" + "03" + "%2F" + "07" + "%2F" + "2013" + "+23%3A59"
    if year == 2015:
        url_addon = "&s_time_id=1425340800"
        url_addon = url_addon + "&s_time_id_f=" + "03" + "%2F" + "03" + "%2F" + "2015" + "+00%3A00"
        url_addon = url_addon + "&e_time_id=1425599940"
        url_addon = url_addon + "&e_time_id_f=" + "03" + "%2F" + "05" + "%2F" + "2015" + "+23%3A59"
    if year == 2017:
        url_addon = "&s_time_id=1488844800"
        url_addon = url_addon + "&s_time_id_f=" + "03" + "%2F" + "07" + "%2F" + "2017" + "+00%3A00"
        url_addon = url_addon + "&e_time_id=1489103940"
        url_addon = url_addon + "&e_time_id_f=" + "03" + "%2F" + "09" + "%2F" + "2017" + "+23%3A59"
    if year == 2019:
        url_addon = "&s_time_id=1551744000"
        url_addon = url_addon + "&s_time_id_f=" + "03" + "%2F" + "05" + "%2F" + "2019" + "+00%3A00"
        url_addon = url_addon + "&e_time_id=1552003140"
        url_addon = url_addon + "&e_time_id_f=" + "03" + "%2F" + "07" + "%2F" + "2019" + "+23%3A59"
    return url_addon


# &s_time_id=1362441600
# &s_time_id_f=03%2F05%2F2013+00%3A00
# &e_time_id=1362700740
# &e_time_id_f=03%2F07%2F2013+23%3A59

# &s_time_id=1425340800
# &s_time_id_f=03%2F03%2F2015+00%3A00
# &e_time_id=1425599940
# &e_time_id_f=03%2F05%2F2015+23%3A59
    
# &s_time_id=1488844800
# &s_time_id_f=03%2F07%2F2017+00%3A00
# &e_time_id=1489103940
# &e_time_id_f=03%2F09%2F2017+23%3A59


# s_time_id=1551744000
# &s_time_id_f=03%2F05%2F2019+00%3A00
# &e_time_id=1552003140
# &e_time_id_f=03%2F07%2F2019+23%3A59

"""
For a given year get the transportation data using station ids for that year
"""
def download(year, detector_ids, PeMS_dir):
    """
    This function downloads traffic data from the PeMS website (pems.dot.ca.gov).
    This function has for input:
        - PeMS detectors ID: detector_ids (an array of detectors)
        - Year for the desired data: year (one year as a integer, should be 2013, 2015, 2017 or 2019)

    This function has for output:
        - All corresponding PeMS detectors data file for the given year (and the given days encoded in the url).
        - Stored in the download folder as PeMS_dir/PeMS_year/PeMS-ID_YEAR.xlsx (where PeMS-ID is the detector ID given by PeMS).
        One xlsx file has two sheets:
            - PeMS Report Description
            - Report Data
                - Contains the traffic flow data
                - Each row gives the number of vehicles observed in one time step (5 minutes) per lane number over the columns.
                - The first column gives the date and time stamp, and the columns that follow are lanes (i.e. Lane 1 Flow, Lane 2 Flow). We care about the column 'Flow (Veh/5 Minutes)' which is the total flow for every lane every 5 minutes.
                - The column "% Observed" correspond to how much of the flow is due to real vehicles sensed or due to estimation from other days due to a technical issue that make the sensor not sensing every cars.
        - Flow data are download for three days
            - From the first Tuesday of March at 00:00am until the first thursday of March at 23:59pm
    For the function to work, you need to:
        - Log in to PeMS in the same browser that runs this Jupyter notebook
    """
    for i in detector_ids:
        url = 'http://pems.dot.ca.gov/?report_form=1'
        url = url + "&dnode=VDS"
        url = url + "&content=loops"

        url = url + "&tab=det_timeseries"
        url = url + "&export=xls"

        url = url + "&station_id=" + str(i)

        url = url + time_for_year(year)

        url = url + "&tod=all"
        url = url + "&tod_from=0"
        url = url + "&tod_to=0"
        url = url + "&dow_0=on"
        url = url + "&dow_1=on"
        url = url + "&dow_2=on"
        url = url + "&dow_3=on"
        url = url + "&dow_4=on"
        url = url + "&dow_5=on"
        url = url + "&dow_6=on"
        url = url + "&holidays=on"
        url = url + "&q=flow"
        url = url + "&q2="
        url = url + "&gn=5min"
        url = url + "&agg=on"
        url = url + "&lane1=on"
        url = url + "&lane2=on"
        url = url + "lane3=on"
        
        if debug:
            print(url)
        webbrowser.open(url, new=2)
        time.sleep(10)
        expected_dir = PeMS_dir + '/PeMS_' + str(year)
        if not os.path.exists(expected_dir):
            os.makedirs(expected_dir)
        os.rename(local_download + '/pems_output.xlsx', expected_dir + '/' + str(i) + "_" + str(year) + ".xlsx")


# local testing
if __name__ == '__main__':
    download(2017, 403250, '~/Desktop')
    pass
