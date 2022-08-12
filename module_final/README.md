## skillfactory_rds / module_final

*Topic*: Anomaly detection for industrial process.

*Target*: Industrial process is complex interaction of objects of different nature to obtain a certain product. 
Modern industrial processes are performed under digital control (local controllers, complex PLC, SCADA etc.).
Industrial process is influenced by various factors:
- changing the parameters (temperature, concentration, flow etc.) of the sources;
- various equipment malfunctions (leaks, wear, jammed etc.);
- failures of sensors and actuators;
- failures of digital control (faulty I/O modules, incorrect algorithms etc.);
- manual intervention of personnel;
- malicious attacks.
Digital control can compensate for some of these factors, but in this case, the industrila process will not work optimally.
In this project, we want explore various methods of anomaly detection for industrial production processes.
Preference will be for methods that can be deploy for online detection.

*About*: Industrial system is presented by a time series of data from sensors and actuators.
The data is sampled by typical frequency of data accumulation in modern industrial systems. Usually it's minutes, not seconds.
Training data usually corresponds to a normal process. Test data can be either normal or include malfunctions.

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
3. **SWaT - Secure Water Treatment**
    - iTrust Centre of Research in Cyber Security: [Overview](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/) and [characterics](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

*Conclusions*:
