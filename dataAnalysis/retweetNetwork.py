# encoding:utf-8
import csv
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import community
import pandas as pd
from networkx import exception

csv.field_size_limit(500 * 1024 * 1024)
# write the dataset you want to analyze
csv_file = csv.reader(open('tweets_with_hashtag_nosmoking.csv', 'r', encoding='utf-8'))

userID = []
retweets = []
for stu in csv_file:
    if stu:
        userID.append(stu[0])
        retweets.append(stu[5])
userID.remove(userID[0])
retweets.remove(retweets[0])

G = nx.DiGraph()
# use undirected graph to detect community
# G = nx.Graph()
G.add_nodes_from(userID)

count = 0
i = 0
for user in userID:
    if retweets[i] != 'none':
        if retweets[i] in userID:
            count += 1
            G.add_edge(retweets[i], user)
i += 1
print(count)

G.remove_nodes_from(list(nx.isolates(G)))
G.remove_edges_from(nx.selfloop_edges(G))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='blue', node_size=30, edge_color='grey')
plt.show()


def analyze(G):
    # normal information
    print(nx.info(G))
    # size of giant component
    giant = max(nx.strongly_connected_components(G), key=len)
    print("The length of giant components: " + str(len(giant)))

    # global clustering coefficient
    print("Global clustering coefficient: " + str(nx.average_clustering(G)))
    # average path length
    try:
        print("Average path length: " + str(nx.average_shortest_path_length(G)))
    except exception.NetworkXError:
        print("Graph is not weakly connected.")

    # maximum degree
    print("maximum degree: " + str(max(nx.degree(G))))
    # maximum k-core number
    print("maximum k-core number: " + str(max(nx.k_core(G, k=None, core_number=None))))


def draw_degree_distribution(G, par, scale=""):
    degree = []
    for i in G.nodes():
        if par == "in":
            degree.append(G.in_degree(i))
        elif par == "out":
            degree.append(G.out_degree(i))
    degree = sorted(degree)
    result = list(set(degree))
    result.sort(key=degree.index)

    distribution = {}
    for i in result:
        distribution[i] = degree.count(i)
    print(distribution)
    x = []
    y = []
    for k, v in distribution.items():
        x.append(k)
        y.append(v / len(set(degree)))
    if scale == 'log':
        # draw degree distribution in log-log scale
        plt.loglog(x, y, "g-s", color="blue", linewidth=2)
    else:
        plt.plot(x, y, "g-s", color="blue")

    plt.show()


def cal_community(G):
    # Starting with an initial partition of the graph and running the Louvain algorithm for Community Detection
    partition = community.best_partition(G, weight='MsgCount')
    values = [partition.get(node) for node in G.nodes()]
    list_com = partition.values()

    # Creating a dictionary like {community_number:list_of_participants}
    dict_nodes = {}
    # Populating the dictionary with items
    for each_item in partition.items():
        community_num = each_item[1]
        community_node = each_item[0]
        if community_num in dict_nodes:
            value = dict_nodes.get(community_num) + ' | ' + str(community_node)
            dict_nodes.update({community_num: value})
        else:
            dict_nodes.update({community_num: community_node})

    # Creating a dataframe from the diet, and getting the output into excel
    community_df = pd.DataFrame.from_dict(dict_nodes, orient='index', columns=['Members'])
    community_df.index.rename('Community_Num', inplace=True)
    community_df.to_csv('Community_List_snippet.csv')

    # Creating a new graph to represent the communities created by the Louvain algorithm
    matplotlib.rcParams['figure.figsize'] = [12, 8]
    G_comm = nx.Graph()
    # Populating the data from the node dictionary created earlier
    G_comm.add_nodes_from(dict_nodes)
    # Calculating modularity and the total number of communities
    mod = community.modularity(partition, G)
    print("Modularity: ", mod)
    print("Total number of Communities=", len(G_comm.nodes()))
    # Creating the Graph and also calculating Modularity
    matplotlib.rcParams['figure.figsize'] = [12, 8]
    pos_louvain = nx.spring_layout(G_comm)
    nx.draw_networkx(G_comm, pos_louvain, with_labels=True, node_size=160, font_size=5,
                     label='Modularity =' + str(round(mod, 3)) +
                           '\nCommunities=' + str(len(G_comm.nodes())))
    plt.suptitle('Community structure (Louvain Algorithm)', fontsize=22, fontname='Arial')
    plt.box(on=None)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
    plt.show()


def draw_community(G):
    # first compute the best partition
    partition = community.best_partition(G)

    # drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
                               node_color=str(count / size))

    G.remove_nodes_from(list(nx.isolates(G)))
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


def cliques(G):
    # Finding the Maximal Cliques associated with teh graph
    a = nx.find_cliques(G)
    i = 0
    # For each clique, print the members and also print the total number of communities
    for clique in a:
        # print(clique)
        i += 1
    total_comm_max_cl = i
    print('Total number of communities: ', total_comm_max_cl)
    # Remove "len(clique)>1" if you're interested in maxcliques with 2 or more edges
    cliques = [clique for clique in nx.find_cliques(G) if len(clique) > 1]

    # klist = list(nx.k_clique_communities(G, 5))
    # list of k-cliques in the network. each element contains the nodes that consist the clique.

    # plotting
    pos = nx.spring_layout(G)
    plt.clf()
    nx.draw(G, pos=pos, with_labels=False, node_size=10, edge_coloe='grey')
    nx.draw(G, pos=pos, nodelist=cliques[0], node_color='b', node_size=10, edge_color='grey')
    nx.draw(G, pos=pos, nodelist=cliques[1], node_color='y', node_size=10, edge_color='grey')
    plt.suptitle('Total number of communities: ' + str(total_comm_max_cl), fontsize=10, fontname='Arial')
    plt.show()

# analyze the basic information of the network
# analyze(G)

# draw degree distribution in normal scale
# draw_degree_distribution(G, "in")
# draw_degree_distribution(G, "out")

# draw degree distribution in log-log scale
# draw_degree_distribution(G, "in", "log")
# draw_degree_distribution(G, "out", "log")

# use undirected graph to detect community
# cal_community(G)
# draw_community(G)
# cliques(G)
