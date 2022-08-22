## skillfactory_rds / module_final

*Topic*: Anomaly detection for industrial process.

*Target*: Industrial process is complex interaction of objects of different nature to obtain a certain product. 
Modern industrial processes are performed under digital control (local controllers, complex PLC, SCADA etc.).
Industrial process is influenced by various factors:
- changing the parameters (temperature, concentration, flow rate etc.) of the sources;
- various equipment malfunctions (leaks, wear, jammed etc.);
- failures of sensors and actuators;
- failures of digital control (faulty I/O modules, incorrect algorithms etc.);
- manual intervention of personnel;
- malicious attacks.
Digital control can compensate for some of these factors, but in this case, the industrial process will not work optimally.

In this project, we want explore various methods of anomaly detection for industrial production processes.
Preference will be for methods that can be implement for online detection in real systems.

*About*: Industrial system is presented by a time series of data from sensors and actuators.
All of features have a physical meaning: temperature, level, flow rate, feed ratio etc.
Some important physical features may not be included due to the inability to measure them.
Relationships between the features is unknown, nonlinear and has different dynamics.
The data is sampled by typical frequency of data accumulation in modern industrial systems. Usually it's minutes, not seconds.
Training data usually corresponds to a normal process. Test data can be either normal or include malfunctions.

*Metrics*: Metrics are selected based on the possible implementation. The detectors works online likes mini-batch mode, not stream mode.
With some frequency, the detector receives a batch of new data from the idustrial system (SCADA for example). 
The detector evaluates the presence of anomalies on the batch as a whole. The exact indication of the time within one batch does matter.
The selected metrics are counted on the batches (15 minutes by default). After that, the ariphmetic mean is calculated over the entire time series.
The main metric is F-score, because it has the largest area of definition and is the harmonic mean of precision and recall. By default, F1-score is used.
Precision and recall scores are used as additional.

*Content*:
1. **0x-xxx** - preparing datasets for using in classes:
  - 00-TEP-rdata-pretreatment-parquet - convert RData TEP Harvard dataset to Parquet, splitting by run and fault-type. To work correctly, need to run it as a script in a virtual python environment (not in jupyter or pycharm).
  - 01-GHL-pretreatment-parquet - convert GHL from Kaspersky dataset to Parquet with processing for more realistic (drop virtual tags and downsample to 1 min interval).
  - 02-TEP-kaspersky-pretreatment-parquet - convert TEP from Kaspersky to Parquet with processing for more realistic (combine some tags and downsample to 1 min interval).
  - 03-SWaT-iTrust-parquet - convert xlsx SWat from iTrust to Parquet with processing for more realistic (downsample to 1 min interval).
2. **1x-xxx** - dataset's visualization
  - 10-
3. **20-simple-watchman-ghl**, **21-simple-watchman-tep** - check SimpleWatchman on various datasets.
4. 
5.
6.
7.
8.
9. **9x-xxx** - comparison of methods among themselves.
  - 

*Datasets*:
1. **GHL - Gasoil Heating Loop** - filling and emptying of tanks, heating of liquid. Saw-type graphs. 2 types of malfucntions.
	- Kaspersky Lab: [Source](https://kas.pr/ics-research/dataset_ghl_1) and [sci-paper](https://arxiv.org/abs/1612.06676)
2. **TEP - Tennessee Eastman Process** - continuous smooth chemical process. Graphs of the oscillation with noise type. 20+ types of malfunctions. Kaspersky datasets also have transient-state runs.
	- Harvard Dataverse: [Source](https://doi.org/10.7910/DVN/6C3JR1) and [some information](https://depts.washington.edu/control/LARRY/TE/download.html) and [more information](https://github.com/camaramm/tennessee-eastman-profBraatz)
	- Kaspersky Lab: [Source](https://kas.pr/ics-research/dataset_tep_59) and [sci-paper](https://arxiv.org/abs/1709.02232)
3. **SWaT - Secure Water Treatment** - real data from water treatment plant.
    - iTrust Centre of Research in Cyber Security: [Overview](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/) and [characterics](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

*Conclusions*:
