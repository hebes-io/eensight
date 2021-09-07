# The configuration approach

`eensight` is a [Kedro](https://github.com/quantumblacklabs/kedro)-based application. Accordingly, the configuration functionality of `eensight` builds on top of the configuration functionality provided by Kedro.

One of the changes that `eensight` has introduced to the Kedro's standard approach is a custom `ConfigLoader` (`eensight.config.OmegaConfigLoader`) that utilizes [`OmegaConf`](https://github.com/omry/omegaconf) as backend. This makes it easy to use [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) when writting the configuration files. As an example, the `globals.yaml` file contains values that can be reused in other files:  

*globals.yaml:*
```yaml
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

... and then in *some_catalog.yaml:*
```yaml
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

It is not necessary to use variable interpolation in `eensight`'s configuration files, but it simplifies the process of writting them. One could have also defined *some_catalog.yaml* as: 

```yaml
site_name : demo

sources:
  train.root_input:
    type: PartitionedDataSet
    path: data/demo/01_raw/train
    dataset:
      type: pandas.CSVDataSet
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

Data catalogs are a way to describe the different datasets that are consumed and/or produced by the `eensight` pipelines. A data catalog includes: 

- The name of the building/site

example:
```yaml
site_name : some_site
```

- Its location (this information is used for automatically generating holiday information) 

example:
```yaml
location:
  country   : Italy  
  province  : null 
  state     : null 
```

- A mapping between specific feature names that the `eensight` functionality expects (consumption, temperature, holiday, timestamp) and the actual names used in the catalog's datasets

example:
```yaml
rebind_names:
  consumption : eload
  temperature : temp 
  holiday     : null 
  timestamp   : dates
```

- The different data sources consumed and produced by `eensight`. The information that is necessary to adequately describe a data source can be found at Kedro's documentation: https://kedro.readthedocs.io/en/stable/05_data/01_data_catalog.html    

### Namespaces

`eensight` supports three namespaces: `train`, `test` and `post`. 

`train` refers to the pre-intervention data and the pipelines that are used for building, optimizing, cross-validating and adding uncertainty information to models that predict baseline energy consumption. 

`test` refers to the pre-intervention data that the baseline model did not see during its fitting and optimization, and the pipelines that evaluate the baseline model on this data. 

`post` refers to the post-intervention data, and the pipelines that calculate cumulative energy savings and their uncertainty intervals.

Adding the appropriate namespace at the beginning of a dataset's name helps automate the process of selecting and running pipelines.

## Model configurations

`eensight` is built around ensembles of localized linear regression models. The structure of these base models can be defined using YAML files. These files have three sections: (a) added features, (b) regressors and (c) interactions.

### Added features

The information in this section is passed to one of the feature generators in `eensight.features.generate`:

- `TrendFeatures`
- `DatetimeFeatures`
- `CyclicalFeatures`

```yaml
add_features:
  time:            # the name of the generator
    feature: null  # the name of the feature (if null, it is the input's index)
    type: datetime
    remainder: passthrough
    subset: month, hourofweek
```

All feature generators create time-related features, and their output is a pandas `DataFrame`. Also, they all have two common parameters:

```yaml
remainder : str, :type : {'drop', 'passthrough'}, default='passthrough'
    By specifying `remainder='passthrough'`, all the remaining columns of the
    input dataset will be automatically passed through (concatenated with the
    output of the transformer).
replace : bool, default=False
    Specifies whether replacing an existing column with the same name is allowed
    (when `remainder=passthrough`).
```


### Regressors

The information for each regressor includes its name, the name of the feature to use and encode so that to create this regressor, the type of the encoder (linear, spline or categorical), and the parameters to pass to the corresponding encoder class from `eensight.features.encode`:

- `IdentityEncoder`
- `SplineEncoder`
- `CategoricalEncoder`

 

```yaml
regressors:
  month:                   # the name of the regressor
    feature: month         # the name of the feature 
    type: categorical
    max_n_categories: null
    encode_as: onehot 
    interaction_only: false
    
  tow:                     # the name of the regressor
    feature: hourofweek    # the name of the feature 
    type: categorical
    max_n_categories: null 
    encode_as: onehot 
    interaction_only: false

```
 

### Interactions

`eensight` supports pairwise interactions between:

- categorical and categorical encoders
- categorical and linear encoders
- categorical and spline encoders
- linear and linear encoders
- spline and spline encoders

Interactions can introduce new regressors, reuse regressors already defined in the regressors section, as well as alter the parameters of regressors that are already defined in the regressors section:

```yaml
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

`eensight` pipelines get their parameter settings from YAML files in the *conf/base/parameters* directory.

```
conf
│   README.md 
│
└───base
│   │   globals.yaml
│   │   logging.yaml
│   │
│   └── parameters
│       ├── default
│           │   preprocess.yaml
│           │   ...
```

Parameters are accessed and treated exactly as prescribed by the Kedro documentation: https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/02_configuration.html#parameters

## Run command arguments

The primary way of using `eensight` is through the command line:

<code>python -m eensight </code>

The command line functionality uses the [Hydra framework](https://hydra.cc/):

*from eensight.cli.py:*

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


The `eensight/hydra/run_config.yaml` file includes all the availabe command line options:

| Argument 	| Description 	|
|---	|---	|
| catalog 	| The name of the catalog configuration file to load 	|
| model 	| The name of the model configuration file to load 	|
| parameters 	| The name of the parameters configuration file to load 	|
| pipeline 	| The name of the modular pipeline to run 	|
| runner 	| The runner that you want to run the pipeline with 	|
| async 	| If load and save node inputs and outputs should be done asynchronously with threads 	|
| env 	| Kedro configuration environment name 	|
| from_inputs 	| A list of dataset names which should be used as a starting point 	|
| to_outputs 	| A list of dataset names which should be used as an end point 	|
| from_nodes 	| A list of node (pipeline step) names which should be used as a starting point 	|
| to_nodes 	| A list of node names which should be used as an end point 	|
| nodes 	| A list with node names. The `run` function will run only these nodes 	|
| tags 	| List of tags. Construct a pipeline from nodes having any of these tags 	|
| load_versions 	| A mapping between dataset names and versions to load 	|
| params 	| These values will override the values in the `parameters` configuration file 	|


`eensight` uses its onw `CustomContext` (that extends `KedroContext`) so that the selected catalog, model and parameter configuration files are passed to the `OmegaConfigLoader`, and become available in the application's [data catalog object](https://kedro.readthedocs.io/en/stable/kedro.io.DataCatalog.html).