## skillfactory_rds / module_final

*Topic*: Anomaly detection for industrial process.

*Target*: Explore various methods of online anomaly detection for industrial production processes. Compare methods with each other for different datasets for the selected metric.

*Content*:
1. **00-rdata-to-csv** - converting RData from TEP Harvard dataset to CSV, splitting by run and fault-type. To work correctly, need to run it as a script in a virtual python environment (not in jupyter or pycharm).
2. **01-ghl-cooking** - converting GHL dataset to CSV with processing for more realistic (drop virtual tags and downsample to 1 min interval).
3. 

*Datasets*:
1. **GHL - Gasoil Heating Loop**
	- Kaspersky Lab: [Source](https://kas.pr/ics-research/dataset_ghl_1) and [sci-paper](https://arxiv.org/abs/1612.06676)
2. **TEP - Tennessee Eastman Process**
	- Harvard Dataverse: [Source](https://doi.org/10.7910/DVN/6C3JR1) and [some information](https://depts.washington.edu/control/LARRY/TE/download.html) and [more information](https://github.com/camaramm/tennessee-eastman-profBraatz)
	- Kaspersky Lab: [Source](https://kas.pr/ics-research/dataset_tep_59) and [sci-paper](https://arxiv.org/abs/1709.02232)

*Conclusions*:
