## Modules

*Sparkling* framework comes into two parts: 

* **heaven** scala module, which contains custom implementations of clustering algorithms and measures. It is written 
on Scala 2.11.12 and does not require any dependencies except Apache Spark. Note, that the module has been already 
compiled into [heaven.jar](../bin/heaven.jar), so you only need to include jar into spark application;

* **sparkling** python module, which is responsible for data preprocessing, 
[optimisation pipeline](GLOSSARY.md#optimisation-pipeline) and interaction with user. 
This module is dependent on some other [python packages](MODULES.md#python-dependencies). 
To manage dependencies, we recommend using [conda](https://docs.conda.io/en/latest/).

### Python dependencies

List of dependencies can be found in [requirements](../requirements) directory. Most of *Sparkling*'s dependencies 
are accessed only if they are really needed, that's why you can install dependencies that suites your case:

* [minimal.txt](../requirements/minimal.txt) - obligatory dependencies, enough to process tabular data;
* [meta.txt](../requirements/meta.txt) - dependencies to launch [CVI predictor](GLOSSARY.md#cvi-predictor);
* [deep.txt](../requirements/deep.txt) - basic dependencies for processing text and image modalities;
* [pytorch.txt](../requirements/pytorch.txt) - using [pytorch](https://pytorch.org/) as deep learning framework;
* [tensorflow.txt](../requirements/tensorflow.txt) - using [tensorflow](https://www.tensorflow.org/) as deep learning framework.

We recommend installing [pytorch.txt](../requirements/pytorch.txt) to access all *Sparkling* functionality.
