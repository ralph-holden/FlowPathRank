# -*- coding: utf-8 -*-
"""
Flow Path and Path Rank

Graph theoretical tools using thermodynamic formalism to detect & understand allosteric pathways

@author: ralph-holden @ Yaliraki Group - Imperial College London

* Simplified and streamlined version to be used with example notebook provided - for full version please contact
"""

# # # IMPORTS # # #
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import heapq
from IPython.utils.io import capture_output
import logging
import os
import csv
import scipy.linalg
from sklearn.linear_model import QuantileRegressor
import copy
from itertools import combinations
import re




# # # aux functions # # #

# used to supress print outputs of functions in high-throughput tasks
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):      # for use w/in jupyter notebook .ipynb
        with capture_output():
            value = func(*args, **kwargs)
        return value
    return func_wrapper



# convert residue ID from 'readable-string format' to 'tuple format'
def filter_residue_string(mystr):

    num_ID = ''
    chain  = mystr.split()[1]

    for i in mystr.split()[0]:
        num_ID += i if i.isnumeric() else ''
    num_ID = int(num_ID)

    return ( num_ID , chain )



def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)



def generate_similars_table(similars_dict, filename_to_save=None):

    # Find all unique entries
    entries = set()
    for key, value in similars_dict.items():
        for entry in value:
            entries.add(entry)
    
    # Create DataFrame
    rows = sorted({entry[1] for entry in entries})
    cols = sorted({entry[0] for entry in entries})
    df = pd.DataFrame(index=rows, columns=cols)
    
    # Populate the DataFrame
    for key, value in similars_dict.items():
        for entry in value:
            df.at[entry[1], entry[0]] = key
    
    # Convert DataFrame to numeric and fill NaN with a specific value
    df = df.apply(pd.to_numeric, errors='coerce').fillna(-1)
    
    # Create a color map
    cmap = plt.cm.get_cmap('coolwarm', np.max(df.values) - np.min(df.values) + 1)
    # Normalize the color map
    norm = plt.Normalize(df.values.min(), df.values.max())
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(df, cmap=cmap, norm=norm)
    
    # Annotate the cells with the DataFrame values
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, int(df.iloc[i, j]) if df.iloc[i, j] != -1 else '',
                    ha='center', va='center', color='black')
    
    # Set tick marks and labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(rows)
    
    ax.set_xlabel('Source Residue')
    ax.set_ylabel('Target Residue')
    ax.set_title('Similarity Groups Table')

    if filename_to_save!=None:
        plt.savefig(filename_to_save+'.pdf')
    
    plt.show()

    

# # # FLOW PATH # # #

class Protein_residue_flow:


    
    def __init__(self, myprot, pdbcode, source_residue, target_residue):
        # inputs
        self.myprot  = myprot
        self.pdbcode = pdbcode
        self.source_residue = source_residue # in form 'abc123 A'
        self.target_residue = target_residue

        self.maxflow_graph_class_dict = {}
        self.residue_path_dict        = {}

        self.is_logged = False

        self.G_res                 = [] 
        self.G_res_flow            = []
        self.G_res_flow_reciprocal = []

        self.best_path = []

        self.is_bottleneck = False

        # exponent scaling factor
        self.beta = 1
        # large beta   -> longer flow paths
        # smaller beta -> shorter flow paths



    # convert BagPype graph from atomistic to residue level 
    def generate_residue_level_graph(self):
        G_res = nx.Graph()

        for interaction in list(self.myprot.graph.edges(data=True)):
        
            atom1 = interaction[0]
            atom2 = interaction[1]
        
            u =  self.myprot.atoms[atom1].res_name + str(self.myprot.atoms[atom1].res_num) + ' ' + self.myprot.atoms[atom1].chain 
            v =  self.myprot.atoms[atom2].res_name + str(self.myprot.atoms[atom2].res_num) + ' ' + self.myprot.atoms[atom2].chain 
        
            if u==v:
                continue
                
            if G_res.has_edge(u, v):
                G_res.add_edge(u, v, weight=G_res[u][v]['weight'] + interaction[2]['weight'], bond_type=G_res[u][v]['bond_type'] + interaction[2]['bond_type'])
            
            else:
                G_res.add_edge(u, v, weight=interaction[2]['weight'], bond_type=interaction[2]['bond_type'])

        self.G_res = G_res

            

    # use NetworkX prebuilt algorithm to find dictionary of maximum flow from source to target residue
    def analyse_flow(self):
        self.max_flow_value , self.max_flow_dict = nx.maximum_flow(self.G_res, self.source_residue, self.target_residue, capacity='weight', flow_func=nx.algorithms.flow.preflow_push)

        

    # FUNCTIONS FOR PATH EXTRACTION
    
    # generate graph from maximum flow dictionary, for path extraction
    def generate_residue_flow_graph(self):
        
        G_res_flow = nx.Graph()
        for u in self.max_flow_dict:
            for v, flow in self.max_flow_dict[u].items():
                
                if flow > 0:  # Ensure only positive flow edges
                    
                    edge_weight_set = flow
        
                    if G_res_flow.has_edge(u , v):
                        G_res_flow.add_edge(u, v, weight= G_res_flow[u][v]['weight'] + edge_weight_set )
                    elif not G_res_flow.has_edge(u, v):
                        G_res_flow.add_edge(u, v, weight= edge_weight_set )

        self.G_res_flow = G_res_flow



    # use Astar algorithm to find path with heuristic based optimum free energy
    def find_optFE_path(self):

        G_res_flow_reciprocal = self.G_res_flow.copy()

        for u,v in list(self.G_res_flow.edges):
            G_res_flow_reciprocal.edges[u,v]['weight'] = self.treat_interaction_weights( self.G_res_flow.edges[u,v]['weight'] ) # assign reciprocal weights
    
        self.best_path = nx.astar_path(G_res_flow_reciprocal, self.source_residue, self.target_residue, heuristic=None, weight='weight')
        
        self.path_flow = min(self.G_res_flow[u][v]['weight'] for u, v in zip(self.best_path[:-1], self.best_path[1:]))

        self.G_res_flow_reciprocal = G_res_flow_reciprocal

    

    def treat_interaction_weights(self, weighting):
        # negative reciprocal so higher weightings have lower transition cost

        return 1/ weighting**self.beta   



    # use priority queue method to find path with optimised flow bottleneck
    def find_optbottleneck_path(self):
        
        path = []
        pq = [(-float('inf'), self.source_residue, [])]
        visited = set()
        
        while pq:
            curr_flow, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]
            if node == self.target_residue:
                return path, -curr_flow
            for neighbour in self.G_res_flow[node]:
                weight = self.G_res_flow[node][neighbour]['weight']
                if neighbour not in visited:
                    heapq.heappush(pq, (max(curr_flow, -weight), neighbour, path))
        
        return None, 0
    


    # FUNCTIONS FOR PATH ANALYSIS


    
    # take best path -> print / log details: edges in path, frequency of edge types, path residues
    def log_path(self):

        # Logging
        log_dir = './log_files/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_filename = f'''log_{self.best_path[0]}-{self.best_path[-1]}.txt'''

        with open(log_dir+log_filename, 'w'): # clear any previous log file of same name
            pass

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up basic configuration for logging to a file
        logging.basicConfig(filename=log_dir+log_filename,
                            level=logging.INFO,
                            format='%(message)s')
    
        start_msg = f'''
RESULTS
Length:            {len(self.best_path)}
        '''
        logging.info(start_msg)
    
        count_master = [['BACKBONE',0],['COVALENT',0],['ELECTROSTATIC',0],['HYDROGEN',0],['HYDROPHOBIC',0],['SALTBRIDGE',0],['STACKED',0]]
    
        title_msg = '''Edges in best path'''
        logging.info(title_msg)
        for i in range(len(self.best_path) - 1):
            u = self.best_path[i]
            v = self.best_path[i + 1]
            for n, tp in enumerate(count_master):
                for b in self.G_res[u][v].get('bond_type'):
                    count_master[n][1] += 1 if b == tp[0] else 0
            edge_msg = f'''EDGE {i} DATA 
    Residues: {self.best_path[i]} , {self.best_path[i+1]}
    Flow:     {self.max_flow_dict[u][v]+self.max_flow_dict[v][u]}
    Weight:   {self.G_res[u][v].get('weight')}
    Type:     {self.G_res[u][v].get('bond_type')}
            '''
            logging.info(edge_msg)
        
        title_msg = f'''Number of bond types in best path by {'A star'}'''
        logging.info(title_msg)
        for entry in count_master:
            entry_msg = f'''{entry[0]}: {entry[1]}'''
            logging.info(entry_msg)
        
        residue_path_list_msg = f'''
    Residues in best path: 
    Length: {len(self.best_path)}
    {self.best_path}
        '''
        logging.info(residue_path_list_msg)



    # Generate pymol command to select and colour specified residues
    def generate_pymol_command(self, lst, colour='<colour>', selection_name='<name>'):
        chains = {}
        formatted_numbers = []
    
        for item in lst:
            # Split the string into the residue number and chain identifier
            parts = item.split()
            residue_chain = parts[0]
            chain_id = parts[1]
            
            # Separate residue number and add to the chain dictionary
            residue_number = ''.join(filter(str.isdigit, residue_chain))
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(residue_number)
            
            # Add the formatted residue number to the list
            formatted_numbers.append(residue_number)
    
        # Format the list with plus signs
        result = '+'.join(formatted_numbers)
    
        # Output the chains and their corresponding residue numbers
        chain_residue_strings = []
        chain_outputs = []
        for chain, residues in chains.items():
            chain_residue_string = f"(chain {chain} and resi {'+'.join(residues)})"
            chain_residue_strings.append(chain_residue_string)
            chain_output = f"Chain {chain}: {'+'.join(residues)}"
            chain_outputs.append(chain_output)
    
        # Generate the PyMOL command
        pymol_command = f"select {selection_name}, {' or '.join(chain_residue_strings)}\nshow sticks, {selection_name}\ncolor {colour}, {selection_name}"
        
        # Join the chain outputs with newlines
        chain_info = '\n'.join(chain_outputs)
    
        return result, pymol_command, chain_info



    # Generate pymol script to colour source+target and flow path, then dim rest of protein 
    def generate_pymol_script(self):
        
        pymol_cmd_list = [f'fetch {self.pdbcode}']
        
        pymol_cmd_list.append( self.generate_pymol_command( self.best_path , colour='red', selection_name='flow_res_path')[1] )
        
        pymol_cmd_sant = 'select s_t, '
        for res in [self.source_residue,self.target_residue]:
            num_ID = ''
            for i in res.split()[0]:
                num_ID += i if i.isnumeric() else ''
            num_ID = int(num_ID)
            pymol_cmd_sant += f'(chain {res.split()[1]} and resi {num_ID})+'
        pymol_cmd_list.append( pymol_cmd_sant[:-1] )
        pymol_cmd_list.append( 'show sticks, s_t' )
        pymol_cmd_list.append( 'color yellow, s_t' )
        
        pymol_cmd_list.append( '''select rest_protein, not flow_res_path and not s_t
set cartoon_transparency, 0.7, rest_protein''' )
        
        pymol_cmd_= '\n'.join(pymol_cmd_list)

        target_dir = './Pymol_scripts/'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        with open(target_dir+f'pymol_cmd_{self.best_path[0]} -- {self.best_path[-1]}.pml', "w") as file: 
            file.write(pymol_cmd_)
        
        #print(f'''Pymol script generated as... pymol_cmd_{self.best_path[0]} -- {self.best_path[-1]}.pml''')



print('Flow Path (residue level) ...')



# # # PATH RANK # # #

class flowpath_RBRW_rank:

    

    def __init__(self, myprot):

        # Bagpype graph
        self.myprot = myprot

        # settings
        self.zeroing_factor    = 1e5                             # make edge-less transitions improbable in transition matrix
        # temperature scale - U / S balance
        self.temperature       = 310.15                          # human body temperature in K
        self.conversion_factor = 1.987e-3 * self.temperature     # kcal/mol per 1 kbT, variable temperature
                                                                 # higher temp -> lesser cost on low edge weights -> shortcuts for shorter paths more likely

        self.U = self.generate_energetics_matrix()
        self.B = self.generate_weighted_adjacency_matrix()

        self.w , self.u , self.v = self.find_dominant_eigs()

    

    def generate_energetics_matrix(self):
        # energetics matrix, U
        # real edges assigned value of negative & reciprocal bagpype graph weight
        # non-edges, ie. zero entries on adjacency matrix A, assigned highly negative value -- acting as an insurmountable cost barrier

        residues = list(set(
            self.myprot.atoms[atom.id].res_name + str(self.myprot.atoms[atom.id].res_num) + ' ' + self.myprot.atoms[atom.id].chain
            for atom in self.myprot.atoms
        ))
        
        num_residues = len(residues)
        U = np.ones((num_residues, num_residues)) * 1/self.zeroing_factor
        
        residue_index_map = {residue: index for index, residue in enumerate(residues)}
    
        for interaction in list(self.myprot.graph.edges(data=True)):
            atom1 = interaction[0]
            atom2 = interaction[1]
    
            residue1 = self.myprot.atoms[atom1].res_name + str(self.myprot.atoms[atom1].res_num) + ' ' + self.myprot.atoms[atom1].chain
            residue2 = self.myprot.atoms[atom2].res_name + str(self.myprot.atoms[atom2].res_num) + ' ' + self.myprot.atoms[atom2].chain
    
            residue_index1 = residue_index_map[residue1]
            residue_index2 = residue_index_map[residue2]
    
            if residue_index1 == residue_index2:
                continue
    
            if U[residue_index1][residue_index2] == 1/self.zeroing_factor:
                U[residue_index1][residue_index2]  = interaction[2]['weight'] #U[atom1][atom2]
                U[residue_index2][residue_index1]  = interaction[2]['weight'] #U[atom1][atom2]
            else:
                U[residue_index1][residue_index2] += interaction[2]['weight'] #U[atom1][atom2]
                U[residue_index2][residue_index1] += interaction[2]['weight'] #U[atom1][atom2]
                
        U = - 1 / ( U * self.conversion_factor) # take elementwise negative reciprocal of SUM of weights

        self.U_df = pd.DataFrame(U, index=residue_index_map.keys(), columns=residue_index_map.keys())
        
        return U


        
    def generate_weighted_adjacency_matrix(self):
        # Energetic adjacency matrix, B
        B = np.exp(self.U)
        
        print(f'''Checking Adjacency matrix, {'B' if True else 'A'}
Symmetric:    {scipy.linalg.issymmetric(B)}
Non-negative: {np.sum(B < 0.0)==0.0}
''')

        return B


    
    def find_dominant_eigs(self):
        # find dominant eigenvalues and left & right eigenvectors
        # dominant eigenvalue (w) and corresponding (left & right) eigenvectors (vl & vr)
        w, u, v  = scipy.linalg.eig(self.B,left=True,right=True)
        dom_u = abs(np.real( u[:, np.argmax(abs(w))] ))
        dom_v = abs(np.real( v[:, np.argmax(abs(w))] ))
        dom_w  = np.real( np.max(abs(w)) )

        # renormalise vectors?
        dom_u /= np.sum(dom_u)
        dom_v /= np.dot(dom_u, dom_v) 

        # Perron–Frobenius theorem
        print(f'''Roulle-Bowen random walk... Checking eigenvalues and eigenvectors...
Is eigenvalue calculated same for left & right:         {np.isclose(dom_w, np.max(w))} , {dom_w} # trivial
Is sum of entries in left eigenvector unity:            {np.isclose(np.sum(dom_u),1)} , {np.sum(dom_u)}
Is sum of product of left and right eigenvectors unity: {np.isclose(np.dot(dom_u,dom_v),1)} , {np.dot(dom_u,dom_v)}
No imaginary parts:                                     {np.sum( np.imag(dom_u+dom_v) )==0.0} , {np.sum( np.imag(dom_u+dom_v) )}
''')
        
        return dom_w , dom_u , dom_v



    def calc_path_probability(self, path):
        # Calculate probability of a path -- for residue level
        
        nsteps = len(path)

        if nsteps==0 or type(path)!=list or 'No pathway found.' in path:
            return 0
        
        U_sum_path = np.sum([ self.U_df[path[i]][path[i+1]] for i in range(len(path)-1) ])

        i_index = self.U_df.columns.get_loc(path[0])
        j_index = self.U_df.columns.get_loc(path[-1])
        
        p_st = self.w**-nsteps * np.exp( U_sum_path ) * self.u[i_index] * self.v[j_index]
        
        return p_st



    def breakdown_path_probability(self, path):

        building_path = []

        for n in path:
        
            building_path += [n]
        
            if len(building_path) < 2: 
                continue
        
            print(f'''Step {len(building_path)}:
Path (so far):          {building_path}
Step energy:            {self.U_df[building_path[-2]][building_path[-1]]}
Step probability:       {self.calc_path_probability(building_path[-2:])}
Total path probability: {self.calc_path_probability(building_path)}
''')



print('Path Rank...')



# # # Putting it together -- Streamlined analysis # # #

def analyse_prot(myprot, myprot_pdb, act_site, allo_site, is_logged=True):
    # Initialise PathRank for scoring paths
    pathrank = flowpath_RBRW_rank( myprot )
    
    # Create dictionary to store results
    paths_dict = {}
    
    # Loop through source & target residue combinations 
    for s_res in act_site:
    
        if s_res not in paths_dict:
            paths_dict[s_res] = {}
    
        print(f'Running Flow Path for source residue {s_res}...')
        
        # Create sub-dictionary entry for target residue
        for j, t_res in enumerate(allo_site):
            paths_dict[s_res][t_res] = {}
    
            # Generate maximum flow graph (residue level)
            prf = Protein_residue_flow(myprot, myprot_pdb, s_res, t_res)
            prf.generate_residue_level_graph()
            prf.analyse_flow()
            prf.generate_residue_flow_graph()
            prf.find_optFE_path()
            
            # find Ruelle-Bowen random walker probability
            RB_prob = pathrank.calc_path_probability( prf.best_path ) 

            # save path data
            if is_logged:
                prf.log_path()              # Log flow path steps and interactions 
                prf.generate_pymol_script() # Generate pymol script for easy visualisation of flow path
             
            # save to output dictionary
            paths_dict[s_res][t_res]['flow_path'] = prf.best_path
            paths_dict[s_res][t_res]['path_l']    = len(prf.best_path)
            paths_dict[s_res][t_res]['RB_prob']   = RB_prob
            
    return paths_dict



# # # POST ANALYSIS # # #

# Retrospective - statistical reduction of data with respect to #steps
class statistical_reduction:


    
    def __init__(self, data_dict, tol=0.02):

        x_path_length = []
        y_RB_prob     = []
        for d1 in data_dict.values():
            for d2 in d1.values():
                x_path_length += [d2['path_l']]
                y_RB_prob     += [d2['RB_prob']]

                if y_RB_prob[-1]==0: # filter for zero results -> replace with near-zero
                    y_RB_prob[-1] = 1e-30        
                    
        self.x_path_length = np.array(x_path_length).reshape(-1,1)
        self.y_RB_prob     = np.log(y_RB_prob)

        self.quantiles = np.linspace(0.01, 0.99, 100)
        
        self.scored_paths_dict = self.quantile_regression( data_dict, tol )


    
    def quantile_regression(self, data_dict, tol):
        
        self.predictions = {}
        self.residuals = {}
        
        scored_paths_dict = copy.deepcopy(data_dict)
        
        print(f"Total number of paths: {len(self.y_RB_prob)}")

        for k1 in data_dict.keys():
            for k2 in data_dict[k1].keys():
                    scored_paths_dict[k1][k2]['RBqs'] = 0.0 # placeholder
            
        for i,q in enumerate(self.quantiles):
            model = QuantileRegressor(quantile=q, alpha=0)  # alpha=0 to avoid regularization
            model.fit(self.x_path_length, self.y_RB_prob)
            self.predictions[q] = model.predict(self.x_path_length)
        
            #print(f"Quantile {q} coefficients: Intercept={model.intercept_}, Slope={model.coef_[0]}")
            
            # Calculate the residuals 
            self.residuals[q] = self.y_RB_prob - self.predictions[q]
        
            #print(f"Number of paths in quantile {round(q,2)}: {np.sum( residuals > 0 ) }")
         
            index_count = 0
            for k1 in data_dict.keys():
                for k2 in data_dict[k1].keys():

                    if self.residuals[q][index_count] > 0:
                        scored_paths_dict[k1][k2]['RBqs'] = round(q, 2)
                    
                    index_count += 1
        
        return scored_paths_dict # data dictionary with Ruelle-Bowen QS entries
    
    


    
    # ADDITIONAL VISUALISATION TOOLS 
    
    def visuals_out(self):

        folder_path = './figures/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        csv_out_path = './summary_csv/'
        if not os.path.exists(csv_out_path):
            os.makedirs(csv_out_path)
        
        self.flow_similars_dict = self.measure_similarity(quantile_threshold=0.75, similarity_threshold=0.5, filename_to_save=folder_path+'RBqs_plot')

        scores_table = self.generate_scores_table(filename_to_save=folder_path+'RBqs_table')
        scores_table.to_csv(csv_out_path+'scores_table.csv', index=True)

        generate_similars_table(self.flow_similars_dict, filename_to_save=folder_path+'similars_table')



    def rank_reduced_scores(self, nrank=10):
        # Ranking code to consider only positive residuals and sort them in descending order
        for q in self.quantiles:
            print(f"Quantile {q} Ranked Scores (greatest positive deviations):")
            
            # Filter for positive residuals (residuals > 0)
            positive_residuals_indices = np.where(self.residuals[q] > 0)[0]  # Indices where residual is positive
            
            # Sort these positive residuals in descending order
            self.sorted_indices = positive_residuals_indices[np.argsort(self.residuals[q][positive_residuals_indices])[::-1]]
            
            # Output the ranked data points
            for i, idx in enumerate(self.sorted_indices[:nrank]):
                print(f"Rank {i+1}: X={[idx][0]}, Actual y={self.y_RB_prob[idx]}, Residual={self.residuals[idx]}")
            
            print("\n")


    
    def plot_residuals(self):
        # Plot the data and the fitted line
        plt.figure(figsize=[5,4*len(self.quantiles)])
        
        for i,q in enumerate(self.quantiles):
            plt.subplot(len(self.quantiles),1,i+1)
            
            plt.plot(self.x_path_length, self.residuals[q], label=f'Quantile {q} Residuals',marker='.',linestyle='')
        
            plt.xlabel('Path length, #res')
            plt.ylabel('Residuals')
            plt.grid(linestyle=':')
            plt.title(f'Residuals of Quantile {q}')
            
        plt.tight_layout(pad=1)
        plt.show()



    def plot_regression(self):
    
        # Plot results
        plt.scatter(self.x_path_length, self.y_RB_prob, alpha=0.5, label="Data", color="gray",marker='.')
        
        for q in self.quantiles:
            plt.plot(self.x_path_length, self.predictions[q], label=f'Quantile {q}')
        
        plt.xlabel("Path length, #res")
        plt.ylabel("Log RB Probability")
        plt.legend()
        plt.title("Quantile Regression")
        plt.show()



    def plot_RBqs(self):

        RBqs_lst = []
        index_count = 0
        for k1 in self.scored_paths_dict.keys():
            for k2 in self.scored_paths_dict[k1].keys():
                RBqs_lst.append( self.scored_paths_dict[k1][k2]['RBqs'] )
                index_count += 1
        
        # Plot the data and the fitted line
        plt.figure(figsize=[8,5])
        plt.plot(self.x_path_length, RBqs_lst, marker='.',linestyle='',color='#ff7f0e' if not self.is_pops else 'b')
        plt.xlabel('Path length, #res')
        plt.ylabel('Quantile Score')
        plt.grid(linestyle=':')
        plt.title('Quantile Scores of Path-Rank')
            
        plt.show()



    def measure_similarity(self, quantile_threshold=0.8, similarity_threshold=0.5, filename_to_save=None):

        quantile_path_keys = []
        RBqs_lst = []
        
        index_count = 0
        for k1 in self.scored_paths_dict.keys():
            for k2 in self.scored_paths_dict[k1].keys():
                RBqs_lst += [ self.scored_paths_dict[k1][k2]['RBqs'] ]
                if self.scored_paths_dict[k1][k2]['RBqs']>quantile_threshold:
                    quantile_path_keys += [ ( k1, k2 ) ]
                index_count += 1
        print(f"Above quantile {quantile_threshold} #paths={np.sum( len(lst) for lst in quantile_path_keys )/2 }")

        
        similars_lsts = [ ]
        
        for i,j in combinations(quantile_path_keys, 2):
        
            pathi = self.scored_paths_dict[i[0]][i[1]]['flow_path']
            pathj = self.scored_paths_dict[j[0]][j[1]]['flow_path']
            
            is_similar = jaccard_similarity( pathi , pathj ) > similarity_threshold
            if is_similar:
        
                in_lst = False
                for lst in similars_lsts:
                    if i in lst and j in lst:
                        in_lst = True
                        break
                    elif i in lst and j not in lst:
                        in_lst = True
                        lst.append( j )
                        break
                    elif j in lst and i not in lst:
                        in_lst = True
                        lst.append( i )
                        break
                        
                if not in_lst:
                    similars_lsts.append( [] )
                    similars_lsts[-1].append( i )
                    similars_lsts[-1].append( j )
        
        for i in quantile_path_keys:
            if np.sum( [ i in lst for lst in similars_lsts ] )==0:
                similars_lsts.append( [] )
                similars_lsts[-1].append( i )       


        plt.figure(figsize=[12,5])

        plt.subplot(1, 2, 1)
        plt.plot(self.x_path_length, RBqs_lst ,marker='.',linestyle='',color='#1f77b4')
        
        plt.xlabel('Path Length, #res')
        plt.ylabel('Quantile Score')
        plt.grid(linestyle=':')
        plt.title('Quantile Score of Path-Rank')
        plt.ylim(0, 1)
        
        plt.subplot(1, 2, 2)
        
        plt.xlabel('Path Length, #res')
        plt.ylabel('Quantile Score')
        plt.grid(linestyle=':')
        plt.title(f'Quantile Scores, Similarity groups > {similarity_threshold}')
        plt.ylim(0, 1)
        
        loner_data_x = []
        loner_data_y = []
        
        for i,grouped_lst in enumerate(similars_lsts):
        
            new_data_x = []
            new_data_y = []
        
            for entry in grouped_lst:
                count_index = 0
                
                for k1 in self.scored_paths_dict.keys():
                    for k2 in self.scored_paths_dict[k1].keys():
                        
                        if k1==entry[0] and k2==entry[1]:
                            new_data_x += [self.x_path_length[count_index]]
                            new_data_y += [RBqs_lst[count_index]]

                            if len(grouped_lst)==1:
                                loner_data_x += [self.x_path_length[count_index]]
                                loner_data_y += [RBqs_lst[count_index]]
                        
                        count_index += 1
        
            if len(grouped_lst)!=1:
                plt.plot(new_data_x, new_data_y, marker='.',linestyle='',label=f'similarity group {i+1}')
        
        if len(loner_data_x)>0:
            plt.plot(loner_data_x, loner_data_y, marker='.',linestyle='',color='black',label='Non similar group') 

        plt.tight_layout(pad=2)
        plt.legend(loc='best')
        if filename_to_save!=None:
            plt.savefig(filename_to_save+'.pdf')
        plt.show()
        
        similars_dict = {}
        for i, lst in enumerate(similars_lsts):
            similars_dict[i] = lst
            
        print(f'Grouped paths (source & target) by similarity {similarity_threshold}')
        
        return similars_dict



    def generate_scores_table(self, filename_to_save=None):
        pruned_data = {}
        for k1 in self.scored_paths_dict.keys():
            
            if k1 not in pruned_data:
                pruned_data[k1] = {}
                
            for k2 in self.scored_paths_dict[k1].keys():
                pruned_data[k1][k2] = self.scored_paths_dict[k1][k2]['RBqs']
        
        df = pd.DataFrame(pruned_data).fillna(0)  # Fill NaNs with zeros

        num_rows, num_cols = df.shape
        fig_width = num_cols * 2  # Width adjustment
        fig_height = num_rows * 0.4  # Height adjustment

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = df.values
        table = plt.table(cellText=table_data, colLabels=df.columns, rowLabels=df.index,
                          cellLoc='center', loc='center', cellColours=plt.cm.coolwarm(table_data / np.max(table_data)))
        
        table.scale(1, 1.5)

        # Add x and y labels
        ax.set_xlabel('Source Residues')
        ax.set_ylabel('Target Residues')
        
        # Add a colorbar
        cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        norm = plt.Normalize(df.values.min(), df.values.max())
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('RB QS')
        #plt.title('Color Coded Table of Dictionary Data')

        if filename_to_save!=None:
            plt.savefig(filename_to_save+'.pdf')
        
        plt.show()
        return df



    
    


class conservation_analysis:


    
        def __init__(self, myprot):
            
            # for conservation scoring -- need to manually overwrite before running methods
            self.conserv_dict = self.extract_consurf_data()
            self.mean_conserv, self.std_conserv = self.find_consurf_stats()
            
            # for identification of weak (functional) residues
            check_prf   = Protein_residue_flow(myprot, 'pdbcode', 's_res', 't_res')
            check_prf.generate_residue_level_graph()
            self.check_G_res = check_prf.G_res
            

            
        def extract_consurf_data(self, force_chain=[]):
            
            conserv_dict = {}
            
            for file in os.listdir('./consurf_dir/'):
                if 'consurf_grades.txt' in file:
            
                    consurf_df = pd.read_csv('./consurf_dir/'+file, header=None,sep="\t",skiprows=29)
                    consurf_df = consurf_df.iloc[:-2]
                    consurf_df.columns = ['POS','SEQ','ATOM','SCORE','COLOR','CONFIDENCE INTERVAL','B/E','F/S','MSA DATA','RESIDUE VARIETY']
                    
                    for n, res in enumerate(consurf_df['ATOM']): 
                        if '-' in res:
                            continue
                        myres_lst = res.split(':')
                        if len(force_chain)==0:
                            res = myres_lst[0].strip() + myres_lst[1] + ' ' + myres_lst[2]
                            conserv_dict[res] = [ float(num) for num in re.findall(r'-?\d*\.?\d+', consurf_df['CONFIDENCE INTERVAL'][n])[:2] ]
                        if len(force_chain)>0:
                            for chain in force_chain:
                                res = myres_lst[0].strip() + myres_lst[1] + ' ' + chain
                                conserv_dict[res] = [ float(num) for num in re.findall(r'-?\d*\.?\d+', consurf_df['CONFIDENCE INTERVAL'][n])[:2] ]
                
            # convert to dataframe and save as csv
            consurf_df_cleaned = pd.DataFrame(conserv_dict)
            consurf_df_cleaned.to_csv("./consurf_dir/consurf_cleaned.csv")
            
            return conserv_dict
        
        
        
        def find_consurf_stats(self):
            conserv_dict_lst = []
            for k in self.conserv_dict.keys():
                conserv_dict_lst += [ np.mean(self.conserv_dict[k]) ]
            return np.mean(conserv_dict_lst), np.std(conserv_dict_lst)
            
        
        
        def find_func_res_conservation(self, scored_paths_dict):
            
            importance_dict = {}

            for k1 in scored_paths_dict.keys():
                for k2 in scored_paths_dict[k1].keys():
            
                    #if k2[-1]!='A': # enforce only allosteric paths
                    #    continue
            
                    mypath = scored_paths_dict[k1][k2]['flow_path']
            
                    path_weaklst    = []
                    for n in range(0, len(mypath)-1):
                        if mypath[n] not in list(self.check_G_res.nodes()):
                            continue
                        if 'COVALENT' not in self.check_G_res[mypath[n]][mypath[n+1]]['bond_type'] and scored_paths_dict[k1][k2]['RBqs']>0.0:
                            for myres in [mypath[n] , mypath[n+1]]:
                                if myres not in path_weaklst:
                                    path_weaklst += [myres]
                                    # don't double count residues!
                                    if myres not in importance_dict:
                                        importance_dict[myres]                  = {}
                                        importance_dict[myres]['raw freq']      = 0
                                        importance_dict[myres]['weighted freq'] = 0
                                        if myres in self.conserv_dict:
                                            importance_dict[myres]['consurf']   = self.conserv_dict[myres]
                                    importance_dict[myres]['raw freq']         += 1
                                    importance_dict[myres]['weighted freq']    += scored_paths_dict[k1][k2]['RBqs']
                                
            # Sort and save dictionary using dataframe
            func_res_df = pd.DataFrame.from_dict(importance_dict, orient='index')
            sorted_df = func_res_df.sort_values('weighted freq',ascending=False)
            sorted_df.to_csv('./summary_csv/functional_residue_conservation.csv')
            
            return sorted_df.iloc[:10]



# # # aux functions 2 # # #
def save_paths_to_csv(scored_paths_dict):
    
    folder_path = './summary_csv/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # save CSV of top scoring flow paths
    with open(folder_path+'paths.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['source_residue', 'target_residue', 'flow_path', 'path_l', 'RB_prob', 'RBqs'])
    
        for k1 in scored_paths_dict.keys():
            for k2 in scored_paths_dict[k1].keys():
                source_residue = k1
                target_residue = k2
                flow_path      = scored_paths_dict[k1][k2]['flow_path']
                path_l         = scored_paths_dict[k1][k2]['path_l']
                RB_prob        = scored_paths_dict[k1][k2]['RB_prob']
                RBqs           = scored_paths_dict[k1][k2]['RBqs']
                writer.writerow([source_residue, target_residue, flow_path, path_l, RB_prob, RBqs])



print('Extra tools...')