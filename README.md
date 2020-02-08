# Data processing

Scripts and code that is used for processing and working with the data for Fremont project.

Dropbox is currently being restructured (new folder structure is being introduced in separate folder called `Private Structured data collection`) and all the stuff here will be manually moved/re-created by Michal and Jiayi. It should not interfere with any ongoing work that is being done outside of this folder.

You can check new Dropbox structure proposed [here](https://docs.google.com/document/d/13c4xRLdxLRR_g7pWuXtTVXjEpxshui2kZBI-i5DG8lo/edit).

### Working with data from Dropbox

- Most of the code operating on the data will be kept in Jupyter notebooks.
- Paths to the files or datasets should be relative so anybody can run `jupyter notebook` from any folder and the code should be able to find all files that are being used.
- `Raw data` folder should not be directly accessed or used as an output destination.
- Data on which the code operates will be stored in `Processed data`.
- Temporary files that are generated during any data processing can be stored in `/tmp` folder - it will be ignored by Git. Make sure to not store any useful data there.
