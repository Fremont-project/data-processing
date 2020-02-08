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

### Working with this repository

- Check [issues](https://github.com/Fremont-project/data-processing/issues) if you are currently without any assigned work. We'll try to keep list of coding tasks that needs to be done here. They should be discussed in meetings prior.
- Discuss the issue or create new one if something is not working as expected (bugs, problems etc.).
- Fork the repository
- Do some changes
- Create new pull request stating what are you trying to add/change or what impact your change will have (we'll try to provide some guidance with prepared issue and pull request templates).
- Ping your colleagues to review your code or Jupyter notebook (not strictly needed, but helps tremendously to do the code review)
- Ping [@michaltakac](https://github.com/michaltakac) to review your PR. It should be merged in under 24 hours if possible.
