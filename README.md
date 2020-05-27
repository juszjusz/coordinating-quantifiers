# Coordinating Quantifiers - Agent Based Simulation of Language Development
## Description
The project is an agent based simulation, implemented as a part of 
 .....
 of the language development and it consists of 
main programs - [simulation.py](https://github.com/juszjusz/coordinating-quantifiers/blob/master/simulation.py) 
and [data_postprocess.py](https://github.com/juszjusz/coordinating-quantifiers/blob/master/data_postprocess.py).
The former is used as  a simulation runner where user may specify parameters, i.e. the length of the game, the size
of the population. The latter inputs the output  of the former and outputs 
visualizations giving the perspective of the language development on step and agent as well
as the cumulative level, i.e. language effectiveness measured as a discriminative success.
In the following sections the tuning and the execution of the simulation will be described. 
## Build
In order to run the program you will need python ver. 3.6 with dependencies specified in 
[requirements.txt](https://github.com/juszjusz/coordinating-quantifiers/blob/master/requirements.txt) file, to setup these dependencies you may run 
a following command in terminal (assuming that terminal is opened in project root directory):
```commandline
>>> pip install -r requirements.txt
```
It is recommended, although not necessary, to isolate the program environment together with its requirements by creating 
a virtual environment, 
[this](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) 
might be use as guide to create a python virtual environment.

## Simulation
To launch simulation where 10 agents interacts with each other over the 100 rounds run the 
following command in terminal:
```commandline
>>> python simulation.py -p 10 -s 100 -r 2
``` 
Simulation will output a serialized representation of language development per agent. Development of agent'
language will be represented as a sequence of matrices where numeric values in matrix cells
correspond to a strengthness of  association between word and category. 
To map, internal to program language representation, on human readable plots you may run 
[data_postprocess.py](https://github.com/juszjusz/coordinating-quantifiers/blob/master/data_postprocess.py)
```commandline
>>> python data_plot.py ...
``` 
## Data Plot
Data plot offers set of parameters that allows you to plot simulation results.
To plot matrices run:
```commandline
>>> python data_plot.py --plot_matrices
```
The above command will output plots to simulation_results/matrices naming matrixN_S.png 
where N, S are numbers standing for agent and simulation step respectively, i.e. 
![matrix](/data_plot_examples/matrix9_31.png)  
## Animated Plots
Below you may observe an animation of the resulting plots:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=gMqZR3pqMjg
" target="_blank"><img src="http://img.youtube.com/vi/gMqZR3pqMjg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>