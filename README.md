# BE-embedding
Code for the experiments carried out in the paper "Efficient Network Embedding by Approximate Equitable Partitions" (Accepted at ICDM 2024) https://arxiv.org/abs/2409.10160

UPDATE: Added extension for weighted and directed networks.

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
- directed bool indicating if the network is directed
- weighted bool indicating if the network is weighted

The syntax for the execution is the following:
```
java -jar epsBE.jar netPath N eps_0 D d partitionPath directed weighted
```
For directed network it is necessary to compute a partition for the original network and another for the transpose.
```
java -jar epsBE.jar netPathT N eps_0 D d partitionPath directed weighted
```
netPath and netPath are respectively the original network and the traspose specified as edgelist files.
##### Example
The following command computes the embedding for the network BrazilAir composed of 131 nodes starting from eps equal to 0 up to 3 with a step equal to 1
```
java -jar epsBE.jar ./datasets/BrazilAir.edgelist 131 0 3 1 ./embed/BrazilAirBE false false
```
##### Networks

The method currently supports undirected/directed unweighted/weighted networks represented using .edgelist files. These files list the edges in the network. For instance, an edge from node 3 to node 4 with weight 5 can be specified as follows 
```
3 4 5 
```
For undirected networks 3 4 5 is equivalent to 4 3 5 and one of them is listed in the file.
The .edgelist files do not contain duplicate entries, and node IDs start from 0.

##### Embedding
To compute the embedding matrix you can use the embedNet.py script with the following parameters:
- netPat path to the input network
- N the number of nodes in the networks
- directed bool indicating if the network is directed
- weighted bool indicating if the network is weighted
- partitionPath path to the partition file computed on the original network
- partitionPathT path to the partition file computed on the transpose (for undirected network "-")
- embedPath path to the output embed file

The syntax for the execution is the following:
```
python embedNet.py netPath N directed weighted partitionPath partitionPathT embedPath
```
##### Example
The following command computes the embedding for the network BrazilAir composed of 131 nodes with the partition previously computed
```
python embedNet.py ./datasets/BrazilAir.edgelist 131 False False ./embed/BrazilAirBE - ./BrazilAirEMB
```

### Experiments
To reproduce the experiments in the paper, we provide the python script main.py. It takes 2 parameters:
- net the name of the network:
 	- REAL NETWORKS
		 - unweighted undirected networks ["BrazilAir", "EUAir", "USAir", "actor", "film"]
		 - unweighted directed networks ["AnybeatD", "FilmTrustD", "FAAD", "EcoliD", "uniEmailD"]
		 - weighted undirected networks ["lesmisW", "newZW", "BibleW", "HSW", "SITCW"]
		 - weighted directed networks ["FBDW", "USairportDW", "AdvogatoDW", "HallDW", "cshiringDW"]
	- SYNTHETIC NETWORKS
		 - unweighted directed networks ["syntD0","syntD5","syntD10","syntD15"]
		 - weighted undirected networks ["syntW0","syntW5","syntW10","syntW15"]
         - weighted directed networks ["syntDW0","syntDW5","syntDW10","syntDW15"]
    - BARBELL NETWORKS
		 - unweighted undirected networks ["Barbell"]
		 - unweighted directed networks ["BarbellD"]
		 - weighted undirected networks ["BarbellW"]
		 - weighted directed networks ["BarbellDW"]
- task the name of the task {"cla","regr","viz"}
  
The syntax for the execution is the following:
```
python main.py net task
```
The possible combinations are the one corresponding to the experiments presented in the paper. 
We present as follow the alternatives:
```
1- python main.py net cla        
2- python main.py net regr       
3- python main.py Barbell viz

```
Commands 1 and 2 compute the classification and regression task for netowrk net.
##### Example
The following command computes the regression task for the network BrazilAir.
```
python main.py BrazilAir regr
```
The command 3 computes the visualization task for the Barbell networks. It is not available for the other networks.

All the results are stored in the results folder.

##### Contacts
For any problem, you can send an email to giuseppesquillace92@gmail.com
