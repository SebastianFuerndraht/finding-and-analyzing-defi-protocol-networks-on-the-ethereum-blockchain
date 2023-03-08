# Finding and analyzing DeFi Protocol networks on the Ethereum Blockchain

## Network plots and topology measures

You can find the:

* network plots 

* network topology measures 

In the folder
```sh
/results
```

## To repeat the steps by yourself follow the steps describe below 

These scripts can be used to analyze the networks of 21 DeFi Protocols on the Ethereum Blockchain.
The networks are created out of transactions between the protocols smart contract accounts. 
The smart contracts are represented as the nodes and the transaction as the edges of the network.

The networks are created using python and the networkx library and are saved as graphml files.
Networkx is also used to perform calculations of different network measures such as node degrees or centrality measures.

The networks can be plotted using R and the igraph Library.
Additionally networks are plotted together for comparison and
heatmaps for the top nodes ranked by centrality measures can be plotted.

The python libraries needed to run the scripts are listed in the requirements.txt file.

## Data needed to run the scripts

CSV file which includes the Ethereum transaction data, where the networks will be created from. `seed_dataset_extended.csv` 

JSON file which includes the categories and protocol names that you want to create the networks for. `protocols-per-category.json`.

The files must be located at:
```sh
./data/input/
```
## Create Networks & calculate network measures (python script)

```sh
./scripts/main.py/
```

The script can be called and the function that should be executed
must be added after the flag `-f` 
To call all functions use `-f "all"`

```sh
python ./scripts/main.py -f "all"
```

It is also possible to call just one function like in the example below:

```sh
python ./scripts/main.py -f "create"
```

* `create`: creates all the protocol networks


* `calc` - Calculates all measures
  * `-p - "protocol"` to the calc measures for
  
* `etherscan` - gets the etherscan labels (if available) for top 10 nodes for each centrality value 
  * `-p - "protocol"` to get the labels for

* `degdist` - Plot Degree Distribution diagrams
  * `-p - "protocol"` to plot
  
  
## Plot the networks (R scripts)
Before any R script can run:
* the graphml files with the network data need to be present for the protocols you want to plot.
* their centrality values have to be calculated
* they need to have etherscan labels

If anything is missing, you can run the according functions of the main.py script:

* `create`
* `calc`
* `etherscan`

Open the script to use directly in R studio

You have to set your working directory inside each script with `setwd()`.

```sh
setwd("C:/Users/Name/PycharmProjects/network-analysis-of-defi")
```

Make sure the correct output paths exist to avoid errors! e.g.
```sh
"./results/plots/networks/pdf/"
"./results/plots/heatmaps/png/"
"./results/plots/networks-per-category/pdf/"
```
The plots can be pdf, png or svg. The format can be changed directly in the R scripts using the variable `format` (heatmaps are only png or svg)

The size of the plots can be adjusted using the variable `size`

* `plot-networks.R` plots each the network for each protocol

* `plot-categories.R` plots the networks specified for each category

If you want plots for a particular subset of protocols you can do that by
changing the values in the file `protocols-per-category.json`.

* `heatmaps.R` plots s heatmap for the centrality measures for each protocol
