# BE-embedding
Codes for the experiments carried out in the paper "Efficient Network Embedding by Approximate Equitable Partitions"

### Environment
All the results are generated in a virtual environment with the following specifications:
- python = 3.10.9
- scikit-learn = 1.2.2
- numpy = 1.23.5
- networkx = 2.8.4
- matplotlib = 3.7.1

### Iterative eps BE algorithm
The algorithm presented in the paper is provided as an executable .jar file called epsBE.jar. It requires 5 parameters in the following order:
- N the number of nodes in the networks
- net the name of the network {"BrazilAir", "EUAir", "USAir", "actor", "film","Barbell"}
- eps_0 the initial epsilon
- D the maximum epsilon considered Delta
- d the step delta

The syntax for the execution is the following:
```
java -jar epsBE.jar N net eps_0 D d
```
##### Example
The following command computes the embedding for the network BrazilAir composed of 131 nodes starting from eps equal to 0 up to 3 with a step equal to 1
```
java -jar epsBE.jar 131 BrazilAir 0 3 1
```
The results are saved in the folder embedNEW.

### Experiments
To reproduce the experiments in the paper, we provide the python script main.py. It takes 2 parameters:
- net the name of the network 
- task the name of the task {"cla","regr","viz","synt"}
  
The syntax for the execution is the following:
```
python main.py net task
```
Not all the possible combinations are available. We present as follow the alternatives:
```
1- python main.py net cla        
2- python main.py net regr       
3- python main.py Barbell viz
4- python main.py synt synt

```
Commands 1 and 2 compute the classification and regression task for any network in the set {"BrazilAir", "EUAir", "USAir", "actor", "film"}.
##### Example
The following command computes the regression task for the network BrazilAir.
```
python main.py BrazilAir regr
```
The command 3 computes the visualization task for the Barbell networks. It is not available for another network.

The command 4 computes the classification for generated synthetic networks.

All the results are stored in the results folder.
