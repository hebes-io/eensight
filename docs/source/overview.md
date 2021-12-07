|
.. image:: https://github.com/hebes-io/eensight/blob/master/logo.png
   :width: 600

|

The `eensight` tool for measurement and verification of energy efficiency improvements
======================================================================================

Goal
----

The goal of the `eensight` tool is to contribute to the advancement of the automated measurement and verification (M&V) methods for energy efficiency. This goal raises the question of why another M&V methodology and tool are needed in the first place.

* The first reason is that there are only a very limited number of M&V frameworks that are ready to be tested and adopted by practitioners. Although the literature on M&V applications is extensive, the gap between: (a) presenting a methodology and its results and (b) offering the tools for practitioners to experiment with this methodology and integrate the parts that they find valuable is most often significant.

* The second reason is that no methodology can be considered a priori the best one. The interaction between any methodology and a specific building – as described by the dataset of its energy/power consumption – determines whether the methodology is suitable for this building or, from the opposite point of view, whether the building is adequately predictable given the selected methodology for M&V. Understanding the aforementioned interaction requires further work on M&V method comparisons, as well as access to a diverse set of M&V models and workflows.

* Finally, energy efficiency improvements can only be estimated through counterfactual analysis, which leads to increased uncertainty and potential for disputes. Although there has been a lot of focus on comparing M&V methodologies according to their predictive accuracy, improving accuracy does not mitigate by itself the uncertainty; uncertainty is mainly driven by all the building’s aspects that may change over time and render the M&V model irrelevant. Accordingly, the proposed M&V methodology focuses on identifying and deconstructing the consumption patterns of a building as much as it focuses on the M&V model’s predictive accuracy. Such information is essential when the M&V model is monitored during its operation so as to verify the degree to which it remains relevant, as well as to determine what has actually changed and how a model adjustment should be guided.

How to use `eensight`
---------------------

`eensight` is a [Kedro](https://github.com/quantumblacklabs/kedro)-based application, and all its functionality is provided in the form of Kedro pipelines. Please see our `API documentation <https://hebes-eensight.readthedocs.io/en/latest/eensight.html>`__ for a complete list of available functions and see our informative 
`tutorials <https://hebes-eensight.readthedocs.io/en/latest/latest/tutorials.html>`__ on how `eensight` can be configured, what methods it utilizes and why it does things this way.  

In addition, this brief instructional video can get you started:
[![Watch the video](https://img.youtube.com/vi/cYBRFeBa_xo/sddefault.jpg)](https://youtu.be/cYBRFeBa_xo)

Python Version
--------------

`eensight` supports `Python 3.7+ <https://python3statement.org/>`__


License
-------

Copyright 2021 Hebes Intelligence. Released under the terms of the Apache License, Version 2.0.

|
.. image:: https://github.com/hebes-io/eensight/raw/main/EC_support.png
   :width: 600

|