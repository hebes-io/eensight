![logo](https://github.com/hebes-io/eensight/blob/master/logo.png)
<br/><br/>

## The `eensight` tool for measurement and verification of energy efficiency improvements

The `eensight` Python package implements the measurement and verification (M&V) methodology that has been developed by the H2020 project [SENSEI - Smart Energy Services to Improve the Energy Efficiency of the European Building Stock](https://senseih2020.eu/). 

The online book *Rethinking Measurement and Verification of Energy Savings* (accessible [here](https://hebes-io.github.io/rethinking/index.html)) explains in detail both the methodology and its implementation.

## Installation

`eensight` can be installed by pip:

```bash
pip install eensight
```

## Usage

### 1. Through the command line

All the functionality in `eensight` is organized around data pipelines. Each pipeline consumes data and other artifacts (such as models) produced by a previous pipeline, and produces new data and artifacts for its successor pipelines.

There are four (4) pipelines in `eensight`. The names of the pipelines and the associations between pipelines and namespaces are summarized below:

|            	| train    	| test   	| apply   |
|------------	|----------	|----------	|---------|
| preprocess 	| &#10004; 	| &#10004; 	| &#10004;|
| predict    	| &#10004; 	| &#10004;	| &#10004;|
| evaluate    	|          	| &#10004;  | &#10004;|
| adjust    	|          	|           | &#10004;|

The primary way of using `eensight` is through the command line. The first argument is always the name of the pipeline to run, such as:

```bash
eensight run predict --namespace train
```
The command

```bash
eensight run --help
```
prints the documentation for all the options that can be passed to the command line.

### 2. As a library

The pipelines of `eensight` are separate from the methods that implement them, so that the latter can be used directly:

```python
import pandas as pd

from eensight.methods.prediction.baseline import UsagePredictor
from eensight.methods.prediction.activity import estimate_activity

non_occ_features = ["temperature", "dew point temperature"]

activity = estimate_activity(
    X, 
    y, 
    non_occ_features=non_occ_features, 
    exog="temperature",
    assume_hurdle=False,

)

X_act = pd.concat([X, activity.to_frame("activity")], axis=1)
model = UsagePredictor(skip_calendar=True).fit(X_act, y)
```

<br>
<img align="left" width="500" src="https://github.com/hebes-io/eensight/blob/master/EC_support.png">
