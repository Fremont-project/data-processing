# Flow-Speed

Code for processing the flow-speed data for 2013, 2015, 2017, 2019 in Fremont project.

Flow-Speed Preprocessing Pipeline 



## Outputs of the pipeline:
- One file matching detectors location to Aimsun road section
- One file with the processed flow data for 2013, 2015, 2017, 2019
- One file with the processed speed data for 2015
- One file with the flow data corresponding to road sections for 2013, 2015, 2017 and 2019
- PCA on flow data

### PCA Analysis
- Data matrix has rows as timesteps, columns as road sections and values are vehicle counts.
- PC3 and PC4 explained the variance in traffic count over years the most. 
- The scatter plot below plots traffic counts projected onto PC4 and PC3 and the coloring are over years. This shows how PC4 and PC3 explain the variance in traffic data over different years.

<p align="center">
  <img src="https://github.com/Fremont-project/data-processing/blob/master/pipeline/flow/pca_years.png" width="75%" height="75%">
</p>

### Example of Output Files
- Here are 3 type of output files produced by the pipeline. 
- Flow_processed_city.csv contains the city traffic flow data over all years (=2013, 2015, 2017, 2019). Each row is a road section and columns are traffic count data per 15 minute timesteps. 
- Flow_processed_pems.csv contains PeMS traffic flow data over all years as well. The general format is the same of Flow_processed_pems.csv.
- Year_info.csv contains geolocation data of the city detectors per year. That is, there is one year_info.csv file per year for 2013, 2015, 2017 and 2019. 

<p align="center">
  <img src="https://github.com/Fremont-project/data-processing/blob/master/pipeline/flow/output_files.png" width="75%" height="75%">
</p>

## Inputs of the pipeline

### Raw Data
- PeMS data downloaded from http://pems.dot.ca.gov
- City average annual daily traffic (AADT) data for 2013, 2017 and 2019 from city of Fremont
- Kimley-Horn flow and speed data for 2015 from city of Fremont

### Manually made dataset
- Aimsun Network
- Detection location 
- Road section layer
- Doc files or city ADT data corresponding to the PDF files
- Detectors ID to corresponding flow file name

## Temporarily-produced files along the pipeline
- CSV flow data
    - City and Kimley Horn
    - PeMS data
- geographic information of road detectors for 2013, 2015, 2017, 2019
- Processed speed data for Kimley Horn




