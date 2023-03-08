from argparse import ArgumentParser
from urllib.request import Request, urlopen
from socket import timeout
from bs4 import BeautifulSoup
from time import sleep
from random import randint
import yaml
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import time
import numpy as np
import scipy

with open('../config.yaml', 'r') as f:
    doc = yaml.safe_load(f)

path_processed_networks = doc["files"]["path_processed_networks"]
path_seed_dataset = doc["files"]["path_seed_dataset"]
path_seed_dataset_extended = doc["files"]["path_seed_dataset_extended"]
path_results = doc["files"]["path_results"]
path_protocols_category = doc["files"]["path_protocols_category"]
path_plots_degree_dist = doc["plots"]["path_degree_dist"]

def loadGraphFromFile(protocol_name):
    return nx.read_graphml(path_processed_networks + protocol_name + '.graphml')

def seed():
    df_seed = pd.read_csv(path_seed_dataset)
    df_extended = pd.read_csv(path_seed_dataset_extended).drop(columns=['Unnamed: 0'])

    result = dict()

    seed_extended = df_extended[['to_address', 'to_protocol']]
    seed_extended.columns = df_extended[['from_address', 'from_protocol']].columns
    seed_extended = seed_extended.append(df_extended[['from_address', 'from_protocol']], ignore_index=True).drop_duplicates().dropna()
    seed_extended.columns = ['address', 'protocol']

    protocols_per_cat = json.load(open(path_protocols_category))

    def label_category (row):
        for category in protocols_per_cat:
            if row['protocol'] in protocols_per_cat[category]:
                return category

    seed_extended['category'] = seed_extended.apply(label_category, axis=1)

    result['seed_category'] = pd.Series.to_dict(df_seed['protocol_type'].value_counts())
    result['seed_protocol'] = pd.Series.to_dict(df_seed['protocol'].value_counts())

    result['seed_extended_category'] = pd.Series.to_dict(seed_extended['category'].value_counts())
    result['seed_extended_protocol'] = pd.Series.to_dict(seed_extended['protocol'].value_counts())

    with open(path_results + '/seed-statistics.json', 'w') as fp:
        json.dump(result, fp)

    return

def createNetworks(protocol_name):
    print('Create networks')

    protocols_per_cat = json.load(open(path_protocols_category))

    if protocol_name is not None:
        for category in protocols_per_cat:
            if protocol_name in protocols_per_cat[category]:
                protocols_per_cat = {category: {protocol_name: protocol_name}}
                break

    if not os.path.exists(path_processed_networks):
        os.makedirs(path_processed_networks)

    df = pd.read_csv(path_seed_dataset_extended).drop(columns=['Unnamed: 0'])

    # exclude CHI token from 1inch network
    exclude_address = '0x0000000000004946c0e9f43f4dee607b0ef1fa1c'.lower()
    df = df[(df.to_address != exclude_address) & (df.from_address != exclude_address)]

    for category in protocols_per_cat:
        for protocol_name in protocols_per_cat[category]:

            df_protocol = df.loc[(df['to_protocol'] == protocol_name) & (df['from_protocol'] == protocol_name)]
            G_whole = nx.from_pandas_edgelist(df_protocol, source='from_address', target='to_address',
                                              edge_attr='transaction_count', create_using=nx.DiGraph())

            node_values = dict()

            for index, row in df_protocol.iterrows():
                if (row['from_address'] or row['to_address']) in node_values.keys():
                    continue
                node_values[row['from_address']] = {'label': row['from_label'],
                                                    'protocol': row['from_protocol']}
                node_values[row['to_address']] = {'label': row['to_label'],
                                                  'protocol': row['to_protocol']}

            for key, value in node_values.items():
                G_whole.nodes[key]['label'] = value['label']
                G_whole.nodes[key]['protocol'] = value['protocol']

            comp = max(nx.weakly_connected_components(G_whole), key=len)
            G = G_whole.subgraph(comp)

            # save graph as graphml files
            nx.write_graphml(G, path_processed_networks + protocol_name + '.graphml')

    print('Finished creating protocol networks')
    return


def calc(protocol_name):
    print('Calculate network measures')
    # ------------ Load protocol networks ------------ #
    protocols_per_cat = json.load(open(path_protocols_category))

    if protocol_name is not None:
        for category in protocols_per_cat:
            if protocol_name in protocols_per_cat[category]:
                protocols_per_cat = {category: {protocol_name: protocol_name}}
                break

    # ------------ Calculate measures ------------ #
    graph_measures = {}

    for category in protocols_per_cat:
        for protocol_name in protocols_per_cat[category]:
            prot_time = time.time()
            print(protocol_name)

            graph_measures[protocol_name] = {}

            G = loadGraphFromFile(protocol_name)

            # ------------ Basic Measures ------------ #
            graph_measures = calcBasicMeasures(G, graph_measures, protocol_name)

            # ------------ Centrality Measures ------------ #
            graph_measures = calcCentrality(G, graph_measures, protocol_name)

            print("Calculated " + protocol_name + " in - %s s -" % (time.time() - prot_time))

    # ------------ Add protocols to categories ------------ #
    protocol_results = json.load(open(path_results + '/results.json'))

    for category in protocols_per_cat:
        for protocol in protocols_per_cat[category]:
            if category in protocol_results:
                protocol_results[category][protocol] = graph_measures[protocol]
            else:
                protocol_results[category] = {protocol: graph_measures[protocol]}

    # ------------ Save measures to file ------------ #
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    with open(path_results + '/results.json', 'w') as fp:
        json.dump(protocol_results, fp)

    print('Finished calculating protocol network measures')
    return


def calcBasicMeasures(G, graph_measures, protocol_name):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    degree = G.degree()
    in_degree = G.in_degree()
    out_degree = G.out_degree()

    graph_measures[protocol_name]['comp_count'] = nx.number_weakly_connected_components(G)
    graph_measures[protocol_name]['nodes'] = nodes
    graph_measures[protocol_name]['edges'] = edges
    graph_measures[protocol_name]['density'] = nx.density(G)
    graph_measures[protocol_name]['max_deg'] = sorted(degree, key=lambda x: x[1], reverse=True)[0][1]
    graph_measures[protocol_name]['max_in_deg'] = sorted(in_degree, key=lambda x: x[1], reverse=True)[0][1]
    graph_measures[protocol_name]['max_out_deg'] = sorted(out_degree, key=lambda x: x[1], reverse=True)[0][1]
    graph_measures[protocol_name]['avg_deg'] = sum(dict(degree).values())/nodes

    return graph_measures

def calcCentrality(G, graph_measures, protocol_name):
    centrality_names = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
                        'eigenvector', 'eigenvector_out', 'pagerank', 'pagerank_out', 'katz', 'closeness', 'betweenness']

    degree_names = ['degree', 'in_degree', 'out_degree']

    degrees = {}

    degrees['degree'] = {node:val for (node, val) in G.degree()}
    degrees['in_degree'] = {node:val for (node, val) in G.in_degree()}
    degrees['out_degree'] = {node:val for (node, val) in G.out_degree()}

    centralities = {}

    centralities['degree_centrality'] = nx.degree_centrality(G)
    centralities['in_degree_centrality'] = nx.in_degree_centrality(G)
    centralities['out_degree_centrality'] = nx.out_degree_centrality(G)

    try:
        centralities['eigenvector'] = nx.eigenvector_centrality(G, max_iter=2000)
        centralities['eigenvector_out'] = nx.eigenvector_centrality(G.reverse(), max_iter=2000)
    except Exception as e:
        print(e)

    centralities['pagerank'] = nx.pagerank(G, 0.85, weight='transaction_count')
    centralities['pagerank_out'] = nx.pagerank(G.reverse(), 0.85, weight='transaction_count')
    centralities['closeness'] = nx.closeness_centrality(G)
    centralities['betweenness'] = nx.betweenness_centrality(G)

    phi = max(nx.adjacency_spectrum(G)).real
    if phi == 0:
        phi = 1
    alpha = (1 / phi) * 0.85
    centralities['katz'] = nx.katz_centrality_numpy(G, alpha=alpha)

    for degree in degree_names:
        nx.set_node_attributes(G, degrees[degree], degree)

    for centrality in centrality_names:

        cent_tmp = centralities[centrality]
        cent_tmp_max = max(cent_tmp.values())
        centralities[centrality] = {k: v / cent_tmp_max for k, v in cent_tmp.items()}

        nx.set_node_attributes(G, centralities[centrality], centrality)
        centralities[centrality] = sorted(centralities[centrality].items(), key=lambda item: item[1], reverse=True)
        graph_measures[protocol_name][centrality] = centralities[centrality][0]

    nx.write_graphml(G, path_processed_networks + protocol_name + '.graphml')

    return graph_measures


def etherscanLabelsByCentrality(protocol_name):
    print('Get etherscan labels for highest centrality nodes')

    # ------------ Load protocol networks ------------ #
    protocols_per_cat = json.load(open(path_protocols_category))

    if protocol_name is not None:
        for category in protocols_per_cat:
            if protocol_name in protocols_per_cat[category]:
                protocols_per_cat = {category: {protocol_name: protocol_name}}
                break

    centrality_names = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
                        'eigenvector', 'eigenvector_out', 'pagerank', 'pagerank_out', 'katz', 'closeness', 'betweenness']

    etherscan_names_list = {
        'curvefinance': 'Curve.fi',
        'fei': 'Fei Protocol',
        'renvm': 'RenVM',
        'barnbridge': 'BarnBridge',
        'dydx': 'dYdX',
        'nexus': 'Nexus Mutual',
        'sushiswap': 'SushiSwap',
        'instadapp': 'InstaDApp',
    }

    for category in protocols_per_cat:
        for protocol_name in protocols_per_cat[category]:
            prot_time = time.time()
            print(protocol_name)

            G = loadGraphFromFile(protocol_name)

            for centrality in centrality_names:
                G_sorted = sorted(G.nodes(), key=lambda n: G.nodes[n][centrality], reverse=True)

                count = 0

                for node in G_sorted:
                    if count >= 10:
                        break

                    if G.nodes[node].get("etherscan") is None:
                        prot = protocol_name
                        if protocol_name in etherscan_names_list:
                            prot = etherscan_names_list[protocol_name]
                        time.sleep(1.5)
                        label = checkEtherscan(node, prot)
                        if type(label) is str:
                            G.nodes[node]["etherscan"] = label
                        else:
                            G.nodes[node]["etherscan"] = 'no label'

                    label = G.nodes[node].get("etherscan")
                    if label:
                        prot = protocol_name
                        if protocol_name in etherscan_names_list:
                            prot = etherscan_names_list[protocol_name]
                        G.nodes[node]["etherscan"] = label.replace(prot + ': ', "")
                        G.nodes[node]["etherscan"] = label.replace(prot.capitalize() + ': ', "")
                        G.nodes[node]["etherscan"] = label.replace(prot.capitalize() + ': ', "")
                        G.nodes[node]["etherscan"] = label.replace(prot.capitalize() + ' ', "")

                    label = G.nodes[node].get("label")
                    if label:
                        G.nodes[node]["label"] = label.replace(protocol_name.capitalize(), "")
                        G.nodes[node]["label"] = label.replace(protocol_name, "")

                    count = count + 1

            nx.write_graphml(G, path_processed_networks + protocol_name + '.graphml')
            print("Got etherscan labels for " + protocol_name + " in - %s s -" % (time.time() - prot_time))

    return


# Checks the nametags for an address on Etherscan
def checkEtherscan(addr, protocol_name):
    etherscan_url = "https://etherscan.io/address/"
    hdr = {'User-Agent': 'Mozilla/5.0'}

    sleep(randint(1,5))

    req = Request(etherscan_url + addr, headers=hdr)
    try:
        html = urlopen(req, timeout=5)
    except timeout:
        return 0
    soup = BeautifulSoup(html, features="html.parser")
    labels = set()
    for span in soup.find_all('span', {'class': 'text-truncate'}):
        labels.add(span.contents[0])
    for label in labels:
        if protocol_name in label.lower():
            return str(label.string)
        if label.lower() in protocol_name:
            return str(label.string)
    return False


def degDist(protocol_name):
    print('Plot degree distributions')

    protocols_per_cat = json.load(open(path_protocols_category))

    if protocol_name is not None:
        for category in protocols_per_cat:
            if protocol_name in protocols_per_cat[category]:
                protocols_per_cat = {category: {protocol_name: protocol_name}}
                break

    for category in protocols_per_cat:
        for protocol_name in protocols_per_cat[category]:
            prot_time = time.time()
            print(protocol_name)

            G = loadGraphFromFile(protocol_name)

            # calculate weighted degrees
            degrees_weighted = {'all': [G.degree()[n] for n in G.nodes()],
                                'in': [G.in_degree()[n] for n in G.nodes()],
                                'out': [G.out_degree()[n] for n in G.nodes()]}

            if not os.path.exists(path_plots_degree_dist):
                os.makedirs(path_plots_degree_dist)

            plotDegDistFigure(degrees_weighted, '', protocol_name)

            print("Plotted degDist of " + protocol_name + " in - %s s -" % (time.time() - prot_time))

    print('Finished degree distribution plots')
    return


def plotDegDistFigure(degrees, filename_ext, protocol_name):
    plt.close()

    # number of nodes
    N = len(degrees['all'])

    degrees_sorted = np.sort(degrees['all'])
    max_degree = degrees_sorted[N - 1] + 1
    degree_counts = [0] * degrees_sorted[N - 1]

    # values for x axis degrees
    x = list(range(1, max_degree))

    # count occurrences of each degree value
    for degree in degrees_sorted:
        degree_counts[degree - 1] += 1

    cumulative_degrees = np.cumsum(degree_counts)

    # calculate y axis probabilities
    y = 1 - cumulative_degrees / float(N)

    # plotting
    plt.figure(figsize=(6, 3.7))
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.yscale('log')
    plt.title('CDF degree distribution ' + protocol_name)
    if max_degree > 100:
        plt.xscale('log')
    plt.plot(x, y)
    plt.savefig(path_plots_degree_dist + protocol_name + filename_ext + ".svg")
    plt.close()

    return


def main(args):
    try:
        if args.function in ['create', 'all'] or (args.function in ['create'] and args.protocol):
            createNetworks(args.protocol)

        if args.function in ['calc', 'all'] or (args.function in ['calc'] and args.protocol):
            calc(args.protocol)

        if args.function in ['etherscan', 'all'] or (args.function in ['etherscan'] and args.protocol):
            etherscanLabelsByCentrality(args.protocol)

        if args.function in ['degdist', 'all'] or (args.function in ['degdist'] and args.protocol):
            degDist(args.protocol)

        if args.function in ['seed', 'all']:
            seed()

    except:
        print("Error in executing")


if __name__ == '__main__':
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument('-f', '--function', help='Select function', type=str, required=False)
    parser.add_argument('-p', '--protocol', help='Select protocol', type=str, required=False)
    parser.add_argument('-i', '--include', help='Select include internal/external addresses', type=str, required=False)
    args = parser.parse_args()
    main(args)
    print("Executed in - %s s -" % (time.time() - start_time))
