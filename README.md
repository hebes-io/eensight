![logo](https://github.com/hebes-io/eensight/blob/master/logo.png)
<br/><br/>

## The `eensight` tool for measurement and verification of energy efficiency improvements

The `eensight` Python package accompanies the deliverable <b>D7.1 Methods for the dynamic measurement and verification of energy savings</b> of the H2020 project [SENSEI Smart Energy Services to Improve the Energy Efficiency of the European Building Stock](https://senseih2020.eu/). The deliverable can be found at [Zenodo](https://zenodo.org/record/4695123#.YHiUD-gzY2w).

The goal of the `eensight` tool is to contribute to the advancement of the automated measurement and verification (M&V) methods for energy efficiency. This goal raises the question of why another M&V methodology anf tool are needed in the first place. 

* The first reason is that there are only a very limited number of M&V frameworks that are ready to be tested and adopted by practitioners. Although the literature on M&V applications is extensive, the gap between: (a) presenting a methodology and its results and (b) offering the tools for practitioners to experiment with this methodology and integrate the parts that they find valuable is most often significant.

* The second reason is that no methodology can be considered a priori the best one. The interaction between any methodology and a specific building – as described by the dataset of its energy/power consumption – determines whether the methodology is suitable for this building or, from the opposite point of view, whether the building is adequately predictable given the selected methodology for M&V. Understanding the aforementioned interaction requires further work on M&V method comparisons, as well as access to a diverse set of M&V models and workflows. 

* Finally, energy efficiency improvements can only be estimated through counterfactual analysis, which leads to increased uncertainty and potential for disputes. Although there has been a lot of focus on comparing M&V methodologies according to their predictive accuracy, improving accuracy does not mitigate by itself the uncertainty; uncertainty is mainly driven by all the building’s aspects that may change over time and render the M&V model irrelevant. Accordingly, the proposed M&V methodology focuses on identifying and deconstructing the consumption patterns of a building as much as it focuses on the M&V model’s predictive accuracy. Such information is essential when the M&V model is monitored during its operation so as to verify the degree to which it remains relevant, as well as to determine what has actually changed and how a model adjustment should be guided.

`eensight` is a [Kedro](https://github.com/quantumblacklabs/kedro)-based application, and all its functionality is provided in the form of Kedro pipelines. The [notebooks/method_explanation](https://github.com/hebes-io/eensight/tree/master/notebooks/method_explanation) folder includes information on how `eensight` can be configured, what methods it utilizes and why it does things this way.  
<br>
<img align="left" width="500" src="https://github.com/hebes-io/eensight/blob/master/EC_support.png">
