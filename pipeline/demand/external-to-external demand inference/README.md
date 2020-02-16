# Manual Preprocess for external-to-external demand inference 
**Output**:
- ext_ext_OD_AM_PM.csv
- labeled_centroids.pdf

**Input**:
- SR_262_Streetlight.pdf

The Streelight SR262 link analysis provides the demand between external endpoints in the entire morning and afternoon.There are in total 4 figures in the Street Light pdf. In each figure, there's only one origin node and multiple destination nodes or only one destination node and multiple origin node. We can infer the demand between external centroids base on that.

1. Label the external centroids on the pdf

The external centroid in the raw data doesn't match with our external centroid. So the first step is to label the external centroids on the pdf for future analysis. As is shown in the figure below, there're 6 external centroids we can use, with external ID 4, 13, 20, 21, 22, 23. 
<p align="center">
  <img src="https://github.com/Fremont-project/data-processing/blob/master/pipeline/demand/external-to-external%20demand%20inference/labeled_centriods.jpg" width="50%" height="50%">
</p>

2. Infer the ext_ext_demand

Take the figure above as an example. The origin node is 1 and 2-7 are destination nodes. Since node 1 is the only origin node, we assume that the external demand between node 1 (External Centroid 13) and External Centroid 4 is 810 (in AM) and 720 (in PM). Then we can infer other ext_to_ext demand in a similar way.

3. Record the ext_to_ext demand in AM and PM in ext_ext_OD_AM_PM.csv.

Manually record the demand inferred from step 2 and save as a csv file.

