import pandas as pd
import numpy as np

ERRONEOUS = ['DurhamRd I680 MissionBlv EB', 'Mission blvd Pine Washington SB']

def run(line_to_detectors, flow_processed_city):
    section = pd.read_csv(line_to_detectors)
    flow = pd.read_csv(flow_processed_city)
    section_no_na = section.dropna()

    w = open('flow_processed_section.csv', 'w')

    secnp = section_no_na.to_numpy()

    legend_tmp = section_no_na.columns.to_numpy()
    # print(legend_tmp)
    legend = legend_tmp[0] + "," + legend_tmp[2] + "," + legend_tmp[3] + ",day 1 2013,day 1 2017,day 1 2019"

    for y in range(3):
        if y == 0:
            year = "2013"
        if y == 1:
            year = '2017'
        if y == 2:
            year = '2019'
        for k in range(3):
            for i in range(24):
                for j in range(4):
                    legend = legend + ",Day " + str(k + 1) + " - " + year + " - " + str(i) + ":" + str(15 * j)
    # print(legend)
    w.write(legend)

    skipped_processing = []
    for l in secnp:
        det2013 = 0
        det2017 = 0
        det2019 = 0
        print(l)

        name_and_dir = get_name_and_direction(l)
        if name_and_dir in ERRONEOUS:
            skipped_processing.append(name_and_dir)
            continue

        if "-" not in l[4]:
            det2013 = (int)(l[4])
            data2013 = flow[flow['Id'] == det2013].to_numpy()[0][4:]
        if "-" not in l[5]:
            det2017 = (int)(l[5])
            data2017 = flow[flow['Id'] == det2017].to_numpy()[0][4:]
        if "-" not in l[6]:
            det2019 = (int)(l[6])
            data2019 = flow[flow['Id'] == det2019].to_numpy()[0][4:]
        if det2013 * det2017 * det2019 != 0:
            print((det2013, det2017, det2019))
            if (data2013.shape[0] != 289) or (data2017.shape[0] != 289) or (data2019.shape[0] != 289):
                print("**** Did not process the line because not three days where recorded")
            else:
                w.write("\n")
                day2013 = data2013[0]
                day2017 = data2017[0]
                day2019 = data2019[0]
                strflow = ""
                for f in data2013[1:]:
                    strflow += str(f) + ","
                for f in data2017[1:]:
                    strflow += str(f) + ","
                for f in data2019[1:]:
                    strflow += str(f) + ","
                strflow = strflow[:-1]
                strflow = day2013 + "," + day2017 + "," + day2019 + "," + strflow
                strflow = str(l[0]) + "," + l[2] + "," + l[3] + "," + strflow
                w.write(strflow)
        else:
            print('Did not process the line because of several detectors')

    w.close()

    print("")
    for name in skipped_processing:
        print("skipped processing", name)

def get_name_and_direction(l):
    name = l[2]
    direction = l[3]
    if direction not in name:
        name += ' ' + direction
    return name

if __name__ == '__main__':
    line_to_detectors = 'lines_to_detectors.csv'
    flow_processed_city = 'Flow_processed_city.csv'
    run(line_to_detectors, flow_processed_city)
    pass
