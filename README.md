# BE-embedding
Codes for the experiments carried out in the paper "Efficient Network Embedding by Approximate Equitable Partitions" (Accepted at ICDM 2024) https://arxiv.org/abs/2409.10160

### Environment
All the results are generated in a virtual environment with the following specifications:
- python = 3.10.9
- scikit-learn = 1.2.2
- numpy = 1.23.5
- networkx = 2.8.4
- matplotlib = 3.7.1

### Iterative eps BE algorithm
The algorithm presented in the paper is provided as an executable .jar file called epsBE.jar. It requires the following 6 parameters and produce an eps-BE partition:
- netPat path to the input network
- N the number of nodes in the networks
- eps_0 the initial epsilon
- D the maximum epsilon considered Delta
- d the step delta
- partitionPath path to the output partition file

The syntax for the execution is the following:
```
java -jar epsBE.jar netPath N eps_0 D d partitionPath
```
##### Example
The following command computes the embedding for the network BrazilAir composed of 131 nodes starting from eps equal to 0 up to 3 with a step equal to 1
```
java -jar epsBE.jar ./datasets/BrazilAir.edgelist 131  0 3 1 ./embed/BrazilAirBE
```
##### Networks

The method currently supports undirected binary networks represented using .edgelist files. These files list the edges in the network. For instance, an edge between nodes 3 and 4 can be specified as either 3 4 or 4 3, but not both. The .edgelist files do not contain duplicate entries, and node IDs start from 0.

##### Embedding
To compute the embedding matrix you can use the embedNet.py script with the following parameters:
- netPat path to the input network
- N the number of nodes in the networks
- partitionPath path to the partition file
- embedPath path to the output embed file

The syntax for the execution is the following:
```
python embedNet.py netPath N partitionPath embedPath
```
##### Example
The following command computes the embedding for the network BrazilAir composed of 131 nodes with the partition previously computed
```
python embedNet.py ./datasets/BrazilAir.edgelist 131 ./embed/BrazilAirBE ./BrazilAirEMB
```

### Experiments
To reproduce the experiments in the paper, we provide the python script main.py. It takes 2 parameters:
- net the name of the network {"BrazilAir", "EUAir", "USAir", "actor", "film"}
- task the name of the task {"cla","regr","viz"}
  
The syntax for the execution is the following:
```
python main.py net task
```
Not all the possible combinations are available. We present as follow the alternatives:
```
1- python main.py net cla        
2- python main.py net regr       
3- python main.py Barbell viz

```
Commands 1 and 2 compute the classification and regression task for any network in the set {"BrazilAir", "EUAir", "USAir", "actor", "film"}.
##### Example
The following command computes the regression task for the network BrazilAir.
```
python main.py BrazilAir regr
```
The command 3 computes the visualization task for the Barbell networks. It is not available for another network.

All the results are stored in the results folder.

##### Contacts
For any problem, you can send an email to giuseppesquillace92@gmail.com
