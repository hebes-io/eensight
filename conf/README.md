<img src="https://raw.githubusercontent.com/hebes-io/eensight/master/logo.png" alt="grouped" width="150" align="left"/>
</br></br>

# The configuration approach

The configuration functionality of `eensight` builds on top of the configuration functionality provided by [`Kedro`](https://github.com/quantumblacklabs/kedro). 

One of the changes that `eensight` has introduced to the `Kedro` approach is a custom `ConfigLoader` that utilizes [`OmegaConf`](https://github.com/omry/omegaconf) as the backend. This makes it easy to use [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) when writting the configuration files. As an example, the `globals.yaml` file contains values that can be reused in other files:  

*globals.yaml:*
```erlang
data_root : data

types:
  csv      : pandas.CSVDataSet
  multiple : PartitionedDataSet
  pickle   : pickle.PickleDataSet

folders:
  raw          : 01_raw
  intermediate : 02_intermediate
  primary      : 03_primary
  feature      : 04_feature
  model_input  : 05_model_input
  model        : 06_models
  model_output : 07_model_output
  report       : 08_reporting
```

... and then *some_catalog.yaml:*
```erlang
site_name : demo

sources:
  train.root_input:
    type: ${globals:types.multiple}
    path: ${globals:data_root}/${site_name}/${globals:folders.raw}/train
    dataset:
      type: ${globals:types.csv}
      load_args:
        sep: ','
        index_col: 0
    filename_suffix: '.csv'
```

There are four (4) types of configuration files in `eensight`:

- Data catalogs
- Model configurations
- Parameter values
- Run command arguments

## Data catalogs

Data catalogs are a way to describe the different datasets that are consumed and/or be produced by the `eensight` pipelines. A data catalog includes: 

- The name of the building/site

- Its location (this information is used for automatically generating holiday information),  

- A mapping between specific feature names that the `eensight` functionality expects (consumption, temperature, holiday, timestamp) and the actual names used in the catalog's datasets

- The different data sources consumed and produced by `eensight`. The information that is necessary to adequately describe a data source can be found at Kedro's documentation: https://kedro.readthedocs.io/en/stable/05_data/01_data_catalog.html    

`eensight` supports three namespaces: `train`, `test` and `post`. 

`train` concerns the pre-intervention data and the pipelines that are used for building, optimizing, cross-validating and adding uncertainty information to models that predict baseline energy consumption. 

`test` concerns pre-intervention data that the baseline model did not see during its fitting and optimization, and pipelines that evaluate the baseline model on this data. 

`post` concerns post-intervention data, and pipelines that calculate cumulative energy savings and their uncertainty intervals.

Adding the appropriate namespace to a dataset's name helps automate the process of selecting and running pipelines.

```erlang
train.preprocessed_data:
    type: ${globals:types.csv}
    filepath: ${globals:data_root}/${site_name}/${globals:folders.intermediate}/train/preprocessed.csv
    load_args:
      sep: ','
      index_col : 0
      parse_dates : 
        - 0
    save_args:
      index: true
    versioned : ${versioned}

  test.preprocessed_data:
    type: ${globals:types.csv}
    filepath: ${globals:data_root}/${site_name}/${globals:folders.intermediate}/test/preprocessed.csv
    load_args:
      sep: ','
      index_col : 0
      parse_dates :
        - 0
    save_args:
      index: true
    versioned : ${versioned}
 ```

## Model configurations

`eensight` is built around ensembles of localized linear regreesion models. The structure of these base models (main effects and pairwise interactions) can be defined using YAML files: 

```yaml
regressors:
  month:           # the name of the regressor
    feature: month # the name of the feature to use and encode so that to create the regressor 
    type: categorical
    max_n_categories: null
    encode_as: onehot 
    interaction_only: false
  
  tow:
    feature: hourofweek
    type: categorical
    max_n_categories: null 
    stratify_by: null 
    excluded_categories: null 
    unknown_value: null
    min_samples_leaf: null  
    max_features: null 
    encode_as: onehot 
    interaction_only: false
  
  temperature:
    feature: temperature
    type: linear
    include_bias: false

interactions:
  tow, temperature:
    tow:
      max_n_categories: 5
      stratify_by: temperature 
      min_samples_leaf: 20
    temperature:
      type: spline
      n_knots: 4
      degree: 2
      strategy: quantile
      extrapolation: constant
```

## Parameter values

`eensight` pipelines get their parameter settings for YAML files in the conf/base/parameters directory.

```
conf
│   README.md 
│
└───base
│   │   globals.yaml
│   │   logging.yaml
│   │
│   └── parameters
│       └── default
│           │   preprocess.yaml
│           │   ...
```

Parameters are accessed and treated exactly as prescribed by the Kedro documentation: https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/02_configuration.html#parameters 

## Run command arguments

The primary way of using `eensight` is through the command line:

<code>python -m eensight </code>

The command line functionality uses the [Hydra framework](https://hydra.cc/), which can dynamically create a hierarchical configuration by composition and override it through config files and the command line.

```python
from .settings import DEFAULT_CATALOG, DEFAULT_MODEL, DEFAULT_PARAMETERS, PROJECT_PATH

@hydra.main(config_path="hydra", config_name="run_config")
def run(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)

    runner = cfg.get("runner") or "SequentialRunner"
    runner_class = load_obj(runner, "kedro.runner")

    parameters = cfg.get("parameter_config") or DEFAULT_PARAMETERS
    catalog = cfg.get("catalog_config") or DEFAULT_CATALOG
    model = cfg.get("model_config") or DEFAULT_MODEL
    ...
```


The `run_config.yaml` file includes all the availabe command line options:

```yaml
# The name of the catalog configuration file to load
catalog : null 
# The name of the model configuration file to load
model : null
# The name of the parameters configuration file to load
parameters : null 
#Name of the modular pipeline to run. If not set, the project pipeline is run 
# by default
pipeline : null 
# Specify a runner that you want to run the pipeline with.
# Available runners: `SequentialRunner`, `ParallelRunner` and `ThreadRunner`
# If not set, `SequentialRunner` will be used.
runner : null 
# Load and save node inputs and outputs asynchronously with threads. 
# If not specified, load and save datasets synchronously
async : False
# Kedro configuration environment name. Defaults to `local`.
env: null 
# A list of dataset names which should be used as a starting point
from_inputs : null 
# A list of dataset names which should be used as an end point
to_outputs : null 
# A list of node names which should be used as a starting point
from_nodes : null 
# A list of node names which should be used as an end point
to_nodes : null 
# A list with node names. The `run` function will run only the nodes with 
# specified names
nodes : null 
# List of tags. Construct a pipeline from nodes having any of these tags
tags : null
# A mapping between dataset names and versions to load. Has no effect on 
# data sets without enabled versioning. 
load_versions : null
# Specify extra parameters that you want to pass to the context initializer. 
# The value of these parameters will override the values in the `parameters`
# configuration file
params : null 
```

`eensight` uses its onw `CustomContext` (that extends `KedroContext`) so that the selected catalog, model and parameter configuration files are passed to the `OmegaConfigLoader`.
 