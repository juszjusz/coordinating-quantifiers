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
In order to run the program you will need python ver. 3.6 with dependencies specified in [requirements.txt]() file, to setup these dependencies you may run 
a following command in terminal (assuming that terminal is opened in project root directory):
```commandline
>>> pip install -r requirements.txt
```
It is recommended, although not necessary, to isolate the program environment together with its requirements by creating 
a virtual environment, you may find this resource [venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
helpful.
## Basic Setup and Execution
To launch simulation where 10 agents interacts with each other over the 100 rounds run the 
following command in terminal:
```commandline
>>> python simulation.py -p 10 -s 100 -r 2
``` 
Simulation will output a representation of language development per agent represented as 
a sequence of matrices, i.e. for agent A matrices m_A_1, m_A_2, ..., m_A_100 where m_A_i 
represents i=1, 2, ..., 100 corresponds to simulation steps and A represents agent. 
To map this - internal to program representation - on human readable plots run the following
command:
```commandline
python data_plot.py ...
``` 
## Animated plots
Below you may observe an animation of the resulting plots:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=gMqZR3pqMjg
" target="_blank"><img src="http://img.youtube.com/vi/gMqZR3pqMjg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>