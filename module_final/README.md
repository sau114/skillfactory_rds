## skillfactory_rds / module_final

*Topic*: 
Anomaly detection for industrial process.

*Target*: 
Industrial process is complex interaction of objects of different nature to obtain a certain product. 
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

*About*: 
Industrial system is presented by multivariate constant frequency time series of values from sensors and actuators.
The values can be either continuous (temperature, pressure, level, flow rate etc.) or discrete/multistate (valve state, discrete level, pump speed stage etc.).
All of features have a physical meaning, but some important physical features may not be included due to the inability to measure them.
Relationships between the features is unknown, nonlinear and has different dynamics.
The data is sampled by typical frequency of data accumulation in modern industrial systems. Usually it's minutes, not seconds.
Training data corresponds to a normal process. Training data can be in one or more files.
Test data can be either normal or include malfunctions. Test data can be in one or more files.

*Metrics*: 
Metrics are selected based on the possible implementation. In this project, detectors are called watchmen.
The watchman works online likes mini-batch mode, not stream mode.
With some frequency (for example every 15 minutes), the watchman receives new data batch from the industrial system (SCADA for example).
The values at one point in time are called a sample. The watchman evaluates the presence of anomalies on the batch for every sample.
The exact indication of the time within one batch doesn't matter in real life.
Thus, divide entire time series into batches. On each batch, determine true anomaly and predict values.
Then calculate the numbers of TP, FP, TN on whole time series. Then calculate the metrics by usual rules.
The main metric is F-score, because it has the largest area of definition and it is the harmonic mean of precision and recall.
By default, F1-score is used, but possible to use beta-parameter. Precision and recall scores are used as additional metrics.
F-score can be NaN if neither true anomalies nor predict on this time series.

*Datasets*:
1. **GHL - Gasoil Heating Loop** - filling and emptying of tanks, heating of liquid. Saw-type graphs. 2 types of malfucntions.
	- Kaspersky Lab: [Source](https://kas.pr/ics-research/dataset_ghl_1) and [sci-paper](https://arxiv.org/abs/1612.06676)
2. **TEP - Tennessee Eastman Process** - continuous smooth chemical process. Graphs of the oscillation with noise type. 20+ types of malfunctions. Kaspersky datasets also have transient-state runs.
	- Harvard Dataverse: [Source](https://doi.org/10.7910/DVN/6C3JR1) and [some information](https://depts.washington.edu/control/LARRY/TE/download.html) and [more information](https://github.com/camaramm/tennessee-eastman-profBraatz)
	- Kaspersky Lab: [Source](https://kas.pr/ics-research/dataset_tep_59) and [sci-paper](https://arxiv.org/abs/1709.02232)
3. **SWaT - Secure Water Treatment** - real data from water treatment plant.
    - iTrust Centre of Research in Cyber Security: [Overview](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat/) and [characterics](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

*Content*:
1. **0x-xxx** - preparing datasets for using in classes:
  - 00-GHL-Kaspersky-pretreatment - pretreating GHL dataset from Kaspersky Lab (drop virtual tags, downsample to 1 min, rename).
  - 01-TEP-Harvard-pretreatment - pretreating TEP dataset from Harvard Dataverse (split by run, split by fault-type, rename).
  - 02-TEP-Kaspersky-pretreatment - pretreating TEP dataset from Kaspersky Lab (combine some tags, downsample to 1 min, rename).
  - 03-SWaT_A1-iTrust-pretreatment - pretreating SWat dataset from iTrust (downsample to 1 min, rename).
2. **1x-xxx** - dataset's visualization
  - 10-plot-datasets - visualization of dataset's time series.
3. **2x-xxx** - examine watchmen on datasets separetely
  - 20-dummy-watchman - simple watchman, predict only normal states.
  - 21-direct-limit-watchman - watchman with direct limits defining. Predict anomalies if out of limits with some tolerance.
  - 22-pca-limit-watchman - watchman with limits defining in principal components space. Predict anomalies if out of limits with some tolerance.
  - 23-isoforest-watchman - watchman, using isolation forest algorithm for predicting.
  - 24-linear-predict-watchman - watchman predict values by previous sample with linear algorithms. Predict anomalies using reconstruction errors.
  - 25-deep-predict-watchman - watchman predict values by previous sample with recurrent neural network. Predict anomalies using reconstruction errors.
3. **3x-xxx** - exploring forests.
4. **4x-xxx** - exploring deep predict.
9. **9x-xxx** - testing and comparison of all watchmen.
  - 90-watchmen-fit-n-save - fitting and saving all watchmen on all datasets.
  - 91-watchmen-validation - validating watchmen.
  - 92-watchmen-tournament - final battle between watchmen.

*Conclusions*:
