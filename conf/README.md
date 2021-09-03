<img src="https://raw.githubusercontent.com/hebes-io/eensight/master/logo.png" alt="grouped" width="150" align="left"/>
</br></br>

# The configuration approach

The configuration functionality of `eensight` builds on top of the configuration functionality provided by [`Kedro`](https://github.com/quantumblacklabs/kedro). 

One of the changes that `eensight` has introduced to the `Kedro` approach is a custom `ConfigLoader` that utilizes [`OmegaConf`](https://github.com/omry/omegaconf) as the backend. This makes it easy to use [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) when writting the configuration files. As an example, the `globals.yaml` file contains values that can be reused in other files:  

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

 
