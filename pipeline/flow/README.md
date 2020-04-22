# Flow-Speed

Code for processing the flow-speed data for 2013, 2015, 2017, 2019 in Fremont project.

Flow-Speed Preprocessing Pipeline 



## Outputs of the pipeline:
- One file matching detectors location to Aimsun road section
- One file with the processed flow data for 2013, 2015, 2017, 2019
- One file with the processed speed data for 2015
- One file with the flow data corresponding to road sections for 2013, 2015, 2017 and 2019
- PCA on flow data

<p align="center">
  <img src="https://github.com/Fremont-project/data-processing/tree/master/pipeline/flow/pca_years.png" width="75%" height="75%">

  <img src="https://github.com/Fremont-project/data-processing/tree/master/pipeline/flow/output_files.png" width="75%" height="75%">
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




