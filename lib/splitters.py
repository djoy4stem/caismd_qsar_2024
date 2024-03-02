from itertools import chain, accumulate
from functools import reduce
from math import ceil, floor, isclose
import warnings
import pandas as pd
from pandas.core import series
import numpy as np
from random import Random, seed

from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from rdkit.Chem import (
    AllChem,
    PandasTools,
    MolToInchiKey,
    MolToSmiles,
    MolFromSmiles,
    SanitizeMol,
    rdFingerprintGenerator
)

from rdkit.Chem.AllChem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from typing import Any, List, Union


from lib import utilities

def check_ratios(train_ratio:float=0.8, val_ratio:float=None
                        , test_ratio:float=0.2):

        val_ratio = float(val_ratio or 0)
        # if val_ratio is None:
        #     assert isclose(0.9999999, train_ratio + test_ratio, rel_tol=1e-06, abs_tol=1e-06), f"train_ratio + test_ratio must be equals 1, not {train_ratio + test_ratio}."
        # else:
        assert isclose(0.9999999, train_ratio + val_ratio + test_ratio, rel_tol=1e-06, abs_tol=1e-06), f"train_ratio + val_ratio + test_ratio must be equals 1, not {train_ratio + val_ratio + test_ratio}."


def flatten(mylist:List[Any]):
    return [item for level_2_list in mylist for item in level_2_list]




class ScaffoldSplitter(object):


    @staticmethod
    def get_bemis_murcko_scaffolds(
        molecules: List[Mol],
        include_chirality: bool = False,
        return_as_indices: bool=True,
        sort_by_size:bool=True

        ):

        def bm_scaffold(molecule: Mol, include_chirality:bool=False):
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=molecule, includeChirality=include_chirality
                )

                return scaffold
            except Exception as exp:
                print(f"Could not generate a Bemis-Murck scaffold for the query molecule. \n{exp}")
                # return None

        try:
            scaffold_smiles = [bm_scaffold(mol, include_chirality) for mol in molecules]

            scaffolds={}

            for s in range(len(scaffold_smiles)):
                scaf = scaffold_smiles[s]
                if scaf in scaffolds:
                    scaffolds[scaf].append(s)
                else:
                    scaffolds[scaf] = [s]

            for skey in scaffolds:
                scaffolds[skey].sort()

            if sort_by_size:
                ## Sort by decreasing number of molecules that have a given scaffold
                scaffolds = dict(sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True))

            if return_as_indices:
                return scaffolds
            
            else:
                scaffolds_items = list(scaffolds.items())
                my_scaffolds = {}

                for j in range(len(scaffolds_items)):
                    mols = None
                    if isinstance(mols, series.Series):
                        mols = [molecules.iloc[k] for k in scaffolds_items[j][1]]
                    else:
                        mols = [molecules[k] for k in scaffolds_items[j][1]]


                    my_scaffolds[ scaffolds_items[j][0]]=mols

                return my_scaffolds
        except  Exception as exp:
            print("Could not generate Bemis-Murcko scaffolds for the list of molecules.")
            raise Exception(exp)



    @staticmethod
    def train_val_test_split(molecules:List[Mol], val_ratio:float=None, train_ratio:float=0.8
                            , test_ratio:float=0.2, return_as_indices:bool=False, return_as_clusters:bool=False
                            , include_chirality:bool=False
                            , sort_by_size:bool=True, shuffle_idx:bool=False, random_state:int=1):

        def len_for_list_of_dicts(ldicts:List[dict]):
            # print([len(d[1]) for d in ldicts])
            l = sum([len(d[1]) for d in ldicts])
            return l

        # try:
        if True:

            check_ratios(train_ratio=train_ratio, val_ratio=val_ratio
                            , test_ratio=test_ratio)


            train_size = train_ratio * len(molecules)
            val_size   = float(val_ratio or 0) * len(molecules)
            test_size  = len(molecules) - train_size - val_size


            train_scaffolds,val_scaffolds,test_scaffolds = [],[],[]

            bmscaffolds = ScaffoldSplitter.get_bemis_murcko_scaffolds(
                                        molecules = molecules,
                                        include_chirality = include_chirality,
                                        return_as_indices = return_as_indices,
                                        sort_by_size = sort_by_size)
                       

            # print("bmscaffolds = ", bmscaffolds)
            curr_train_len, curr_val_len, curr_test_len = 0,0,0

            bmscaffolds_items = list(bmscaffolds.items())
            # print(bmscaffolds_items[:10])

            if shuffle_idx:
                a = bmscaffolds_items[10:]
                Random(random_state).shuffle(a)
                bmscaffolds_items = bmscaffolds_items[:10] + a

            
            for bms in bmscaffolds_items:
                # print(train_scaffolds)
                # print(len(bms[1]), bms)
                # print(f"curr_train_len: {curr_train_len}")
                # print(f"len(bms): {len(bms[1])}")
                bms_size = len(bms[1])
                if curr_train_len + bms_size > train_size:
                    if val_size>0:
                        if curr_val_len + bms_size > val_size:
                            # print(f"adding bms to test")
                            test_scaffolds.append(bms)
                            
                        else:
                            # print(f"adding bms to val: {curr_val_len + bms_size}")
                            val_scaffolds.append(bms)
                            curr_val_len = len_for_list_of_dicts(val_scaffolds)
                else:
                    # print(f"adding bms to train: {curr_train_len + bms_size}")
                    train_scaffolds.append(bms)
                    curr_train_len = len_for_list_of_dicts(train_scaffolds)

            if not return_as_clusters:
                train_scaffolds = utilities.flatten_list([t[1] for t in train_scaffolds])
                val_scaffolds   = utilities.flatten_list([t[1] for t in val_scaffolds])
                test_scaffolds  = utilities.flatten_list([t[1] for t in test_scaffolds])



            return train_scaffolds, val_scaffolds, test_scaffolds
        # except Exception as exp:
        #     print(f"Could not split dataset. \n{exp}")


    @staticmethod
    def kfold_split(molecules:List[Mol], n_folds:int=5, return_as_indices:bool=False, include_chirality:bool=False
                            , random_state:int=1, sort_by_size:bool=True):
        
        try:
        # if True:
            fold_size = ceil(len(molecules)/n_folds)
            folds = []
            start_idx = 0
            for i in range(n_folds-1):
                print(f"Fold {i}  :: start: {start_idx} - stop: {start_idx+fold_size-1}")
                if isinstance(molecules, series.Series):
                    folds.append(molecules.iloc[start_idx : start_idx+fold_size-1])
                else:
                    folds.append(molecules[start_idx : start_idx+fold_size-1])
                
                start_idx += fold_size-1
            
            print(f"start_idx: {start_idx}")
            folds.append(molecules[start_idx:])

            return folds
        except Exception as exp:
            print(f"Could not perform k-fold split on dataset. \n{exp}")                





class ClusterSplitter(object):
    
    # _allowable_fptypes = ['morgan', 'atom_pair', 'topoligical']

    @staticmethod
    def generate_fingerprints(molecules:List[AllChem.Mol], fingerprint_type:str='morgan', fp_size:int=1024, radius:int=2
                            , count_bits:bool=True, include_chirality:bool=False):
        fingerprints = []
        fprinter = None
        if fingerprint_type == 'morgan':
            fprinter = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size, countSimulation=count_bits)
        elif fingerprint_type == 'atom_pair':
            fprinter = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_size, countSimulation=count_bits, maxPath=30, includeChirality = include_chirality)
        elif fingerprint_type == 'topological':
            fprinter = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fp_size, countSimulation=count_bits, includeChirality = include_chirality)
        # elif fingerprint_type == 'maccs':
        #     fprinter = rdFingerprintGenerator.GetTopologicalTorsionalGenerator(fpSize=fp_size, countSimulation=count_bits, includeChirality = include_chirality)
        else:
            raise ValueError(f"No implementation for fingerprint type {fingerprint_type}. Allowable types include: {'; '.join(_allowable_fptypes)}")


        if count_bits:
            for mol in molecules:
                try:
                    fingerprints.append(fprinter.GetCountFingerprint(mol))
                except:
                    fingerprints.append(None)
        else:
            for mol in molecules:
                try:
                    fingerprints.append(fprinter.GetFingerprint(mol)) 
                except:
                    fingerprints.append(None)

        return fingerprints

    @staticmethod
    def tanimoto_distance_matrix(fp_list):
        """
        Calculate distance matrix for fingerprint list
        Taken from https://projects.volkamerlab.org/teachopencadd/talktorials/T005_compound_clustering.html
        
        """
        dissimilarity_matrix = []
        # Notice how we are deliberately skipping the first and last items in the list
        # because we don't need to compare them against themselves
        for i in range(1, len(fp_list)):
            # Compare the current fingerprint against all the previous ones in the list
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix

    @staticmethod
    def cluster_fingerprints(fingerprints, sim_cutoff=0.7, return_as_indices:bool=False):
        """Cluster fingerprints
        Parameters:
            fingerprints
            dist_threshold: threshold for the clustering. Molecules with a dissimilarity below (or similarity above) this threshold are grouped into the same cluster.
        """
        # Calculate Tanimoto distance matrix
        distance_matrix = ClusterSplitter.tanimoto_distance_matrix(fingerprints)
        # Now cluster the data with the implemented Butina algorithm:
        clusters = Butina.ClusterData(data=distance_matrix, nPts=len(fingerprints), distThresh=1-sim_cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)

        if return_as_indices:
            return clusters
        else:
            clusters_of_fps = []
            for cluster in clusters:
                culster_m = None
                if isinstance(fingerprints, series.Series):
                    cluster_m = [fingerprints.iloc[i] for i in cluster]
                else:
                    cluster_m = [fingerprints[i] for i in cluster]
                clusters_of_fps.append(cluster_m)

            return clusters_of_fps                
        


    @staticmethod
    def cluster_molecules(molecules:Union[List[AllChem.Mol], pd.Series], fingerprint_type:str='morgan'
                          , fp_size:int=1024, radius:int=2, count_bits:bool=True
                          , include_chirality:bool=False, sim_cutoff=0.7
                          , return_as_indices:bool=False
                        ):
        
        fingerprints = ClusterSplitter.generate_fingerprints(molecules=molecules,fingerprint_type=fingerprint_type
                                                             , fp_size=fp_size, radius=radius, count_bits=count_bits
                                                             , include_chirality=include_chirality
                                                            )
        
        clusters = ClusterSplitter.cluster_fingerprints(fingerprints=fingerprints, sim_cutoff=sim_cutoff
                                                        , return_as_indices=True
                                                    )
        

        if return_as_indices:
            return clusters
        else:
            clusters_of_mols = []
            for cluster in clusters:
                cluster_m = None
                # print(cluster)
                if isinstance(molecules, series.Series):
                    cluster_m = [molecules.iloc[i] for i in cluster]
                else:
                    cluster_m = [molecules[i] for i in cluster]
                clusters_of_mols.append(cluster_m)

            return clusters_of_mols
        


    @staticmethod
    def train_val_test_split(molecules:List[Mol], val_ratio:float=None, train_ratio:float=0.8
                            , test_ratio:float=0.2, return_as_indices:bool=False
                            , return_as_clusters:bool=False
                            , include_chirality:bool=False, fingerprint_type:str='morgan'
                            , fp_size:int=1024, radius:int=2
                            , count_bits:bool=True, sim_cutoff:float=0.2
                            , sort_by_size:bool=True, shuffle_idx:bool=False, random_state:int=1):

        def len_for_list_of_dicts(ldicts:List[dict]):
            # print([len(d[1]) for d in ldicts])
            l = sum([len(d) for d in ldicts])
            return l

        try:
        # if True:

            check_ratios(train_ratio=train_ratio, val_ratio=val_ratio
                            , test_ratio=test_ratio)


            train_size = int(train_ratio * len(molecules))
            val_size   = int(float(val_ratio or 0) * len(molecules))
            test_size  = len(molecules) - train_size - val_size
            print(train_size, val_size, test_size)
 
            train_clusters,val_clusters,test_clusters = [],[],[]

            clusters = ClusterSplitter.cluster_molecules(molecules=molecules, fingerprint_type=fingerprint_type
                                                         , fp_size=fp_size, radius=radius, count_bits=count_bits
                                                         , include_chirality=include_chirality, sim_cutoff=sim_cutoff
                                                         , return_as_indices=return_as_indices
                                                        )
                       

            # print("clusters = ", clusters)
            # print([len(cl) for cl in clusters])
            curr_train_len, curr_val_len, curr_test_len = 0, 0, 0


            # print(clusters[:10])

            if shuffle_idx:
                a = clusters[10:]
                Random(random_state).shuffle(a)
                clusters = clusters[:10] + a

            bms_counter = 0
            # not_added = []
            for bms in clusters:
                bms_counter += 1
                # print(train_clusters)
                # print(len(bms[1]), bms)
                # print(f"bms len: {len(bms)}  -- curr_train_len: {curr_train_len}")
                # print(f"len(bms): {len(bms[1])}")
                bms_size = len(bms)
                if curr_train_len + bms_size > train_size and bms_counter>1:
                    # print("OK")
                    # if  bms_counter == 1:
                    #     train_clusters.append(bms)
                    #     curr_train_len = len_for_list_of_dicts(train_clusters)                        
                    if val_size>0:
                        if curr_val_len + bms_size > val_size:
                            # print(f"adding bms to test")
                            test_clusters.append(bms)
                            
                        else:
                            # print(f"adding bms to val: {curr_val_len + bms_size}")
                            val_clusters.append(bms)
                            curr_val_len = len_for_list_of_dicts(val_clusters)
                    else:
                        # print(f"adding bms to test: {curr_test_len + bms_size}")
                        test_clusters.append(bms)
                        curr_test_len = len_for_list_of_dicts(test_clusters)                        
                else:
                    # print(f"adding bms to train: {curr_train_len + bms_size}")
                    train_clusters.append(bms)
                    curr_train_len = len_for_list_of_dicts(train_clusters)

            if not return_as_clusters:
                train_clusters = utilities.flatten_list(train_clusters)
                val_clusters   = utilities.flatten_list(val_clusters)
                test_clusters  = utilities.flatten_list(test_clusters)



            return train_clusters, val_clusters, test_clusters
        except Exception as exp:
            print(f"Could not split dataset. \n{exp}")


    @staticmethod
    def kfold_split(molecules:List[Mol], n_folds:int = 5, test_ratio:float=0.0                  
                    , return_as_indices:bool=False, return_as_clusters:bool=False
                    , include_chirality:bool=False, sort_by_size:bool=True
                    , random_cluster_pick:bool=True, random_state:int=None
                    , num_repeats:int = 1, fingerprint_type:str='morgan'
                    , fp_size:int=1024, radius:int=2, count_bits:bool=True
                    , sim_cutoff:float=0.2, shuffle_idx:bool=False):
        try:
            def len_cm(my_clusters): #:Union[List[List[Any]],  List[List[Any]]]
                return reduce(lambda x,y:x+y, map(len, [c for c in my_clusters]))

            def idx_to_mols(indices, molecules):
                assert np.greater_equal(len(molecules), len(indices)), "The length of arg 2 must be greater or equal that of arg 1."
                if isinstance(indices, series.Series):
                    return [molecules.iloc[i] for i in indices]
                else:
                    return [molecules[i] for i in indices]

            assert np.greater_equal(n_folds, 2), (
                "Expect the number of splits to br greater or equal to 2", 
                ", got {:4f}".format(n_folds)
            )
            

            selected_clusters = []

            clusters = []
            if isinstance(molecules, (list, np.ndarray, series.Series)) and all([isinstance(molecule, Mol) for molecule in molecules]):
                clusters = ClusterSplitter.cluster_molecules(
                                                                molecules=molecules, fingerprint_type=fingerprint_type
                                                                , fp_size=fp_size, radius=radius
                                                                , count_bits=count_bits, include_chirality=include_chirality
                                                                , sim_cutoff=sim_cutoff, return_as_indices=True
                                                            )
            elif isinstance(molecules, (list, np.ndarray)) and all([isinstance(molecule, list) for molecule in molecules]):
                clusters = ClusterSplitter.cluster_fingerprints(fingerprints=molecules, sim_cutoff=sim_cutoff
                                                                , return_as_indices=True)

            # print(f"num. clusters = {len(clusters)}")
            # print(f"clusters = {clusters}")

            if num_repeats > 1 and not random_cluster_pick:
                warnings.warn(f"random_cluster_pick is set to False while num_repeats > 1({num_repeats}). random_cluster_pick will be set to True.")

            all_folds = []
            total_mol_size = len_cm(clusters)
            test_size      = int(test_ratio * total_mol_size)
            train_val_size = len(molecules) - test_size
            nmols_by_split = floor(train_val_size/n_folds)
            # print(f"nmols_by_split = {nmols_by_split}   -  test size = {test_size}")


            for rcounter in range(num_repeats):
                # print(f"\n\n{rcounter+1} of {num_repeats} repeats\n***************\n")
                if num_repeats > 1 and not random_state is None:
                    random_state += rcounter
                    seed(random_state)

                cluster_mols = clusters.copy()
                cluster_picking_order = list(range(len(cluster_mols))) 
                # print(f"cluster_picking_order = {cluster_picking_order}")

                if random_cluster_pick:
                    Random(random_state).shuffle(cluster_picking_order)
                    # print(f"cluster_picking_order = {cluster_picking_order}")

                test         = None
                train_val    = []
                current_size = 0
                picked       = []
                
                if test_size > 0:
                    test = []
                    for i in cluster_picking_order:
                        item_ = cluster_mols[i]
                        if len(item_) <=5: ## enforce diversity
                            test.append(item_)
                            picked.append(i)
                            current_size += len(item_)
                            if current_size >= test_size:
                                break

                    # test_res = None
                    # print(f"test = {test}")
                    if not return_as_clusters:
                        test = list(chain(*[it for it in test]))
                        # print(f"new test = {test}")
                        if not return_as_indices:
                            test = idx_to_mols(test, molecules)
                    else:
                        if not return_as_indices:
                            test = []
                            if isinstance(molecules, series.Series):
                                test = [molecules.iloc[x] for x in test]
                            else:
                                test = [molecules[x] for x in test]                        
                                    
                    # print(f"test = {test}\n\n")


                train_val = [cluster_mols[k] for k in cluster_picking_order if not k in picked]
                cluster_picking_order = list(range(len(train_val)))

                if random_cluster_pick:
                    Random(random_state).shuffle(cluster_picking_order)

                counters      = list(range(n_folds))
                splits          = [[] for i in counters]
                added_clusters  = []
                current_indices = []

                for i in counters:                
                    split = splits[i]
                    # print(f"\nCounter: {i} - Split = {split}")
                    split_len = len(split)

                    mini_c = []
                    for j in cluster_picking_order:
                        # print(f"j = {j}  :  split_len = {split_len}")
                        
                        if split_len == nmols_by_split:
                                break
                        elif not j in added_clusters:
                            cluster  = train_val[j]
                            
                            curr_len = len(cluster)
                            if split_len + curr_len <= nmols_by_split:
                                split.append(cluster)
                                mini_c.append(j)
                                split_len = len_cm(split)
                                # print(f"cluster ({len(cluster)})= {cluster}")
                                # print(f"mini_c = {mini_c}")

                        cluster_picking_order = [
                            m for m in cluster_picking_order if not m in mini_c
                        ]
                    # print(f"mini_c = {mini_c}")
                    # print(f"cluster_picking_order = {cluster_picking_order}")
                remaining_clusters = [
                    l for l in cluster_picking_order if not l in added_clusters

                ]   
                # print(f"remaining_clusters = {remaining_clusters}")
                # print(f"splits ({len(splits)}): ", [len(s) for s in splits])

                for k in remaining_clusters:
                    splits[-1].append(train_val[k])

                current_folds = []

                if not return_as_clusters:
                    # print([len(s) for s in splits])
                    for s in range(len(splits)):
                        sp = splits[s]
                        sp = flatten([t for t in sp])
                        
                        if return_as_indices:
                            splits[s] = sp
                        else:
                            sp = idx_to_mols(sp, molecules)
                            splits[s] = sp
                            # print(f"splits[s] = {splits[s]}")
                    # print(f"0 splits = {splits} | size = {[len(s) for s in splits]}")

                    for i  in counters:
                        train = list(chain(*[splits[c] for c in counters if c != i]))
                        
                        current_folds.append(tuple([train, splits[i], test])) 
                        
                    all_folds.append(current_folds)

                else:
                    if not return_as_indices: 
                        tag = 0 
                        for s in range(len(splits)): 
                            sp = splits[s] 
                            new_item = [] 
                            # print("sp = ", sp)
                            for clus in sp:
                                new_item.append(idx_to_mols(clus, molecules))
                            tag += 1 

                            splits[s] = new_item 
                            # print(f"splits[{s}] ({len(splits[s])}) = {splits[s]}")                 

                    for i in counters: 
                        if test_ratio > 0:
                            # print("n_folds - 1 ", len(list(chain(*[splits[k] for k in counters if k!=i]))))
                            # print(f"splits[i]: {len(splits[i])}")
                            # print(f"test: {len(test)}")
                            current_folds.append((list(chain(*[splits[k] for k in counters if k!=i])), splits[i], test))
                        else:
                            current_folds.append(tuple(list(chain(*[splits[k] for k in counters if k!=i])), splits[i], None))

                    all_folds.append(tuple(current_folds)) 

            return all_folds 

        except Exception as e: 
            print("Could not perform KFold split") 
            raise Exception(str(e))                 


def mol2fp(mol):
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar



def min_max_train_validate_test_split_df(dataframe, molecule_column, inchikey_column=None, fp_column=None, train_valid_ratios=[0.7, 0.15]
                                , fp_type= "morgan", random_state=1, return_indices=False):
    """
    """
    
    # Store the InChIKeys. These will be used to split the dataframe to ensure no molecule is both in the train and test sets.
    if inchikey_column is None:
        print("Computing and storing the InChiKeys...")
        inchikey_column = "InChIKey"
        dataframe[inchikey_column] = dataframe[molecule_column].apply(lambda x: MolToInchiKey(x))
    
    dataframe.apply(lambda x: x[molecule_column].SetProp("InChIKey", x[inchikey_column]), axis=1)
    
    # Select unique molecules (by InChiKey)
    dataframe_single_ikeys = dataframe.drop_duplicates(subset=[inchikey_column], keep='first')     
    list_of_rdkit_representations = None
    if fp_column is not None:
        list_of_rdkit_representations = dataframe_single_ikeys[fp_column].values.tolist()
    else:
        list_of_rdkit_representations = dataframe_single_ikeys[molecule_column].values.tolist()
    
    # Split datasets
    print("Splitting the dataset...")
    train_validate_test_splits = min_max_train_validate_test_split(list_of_rdkit_representations, train_valid_ratios=train_valid_ratios
                                                 , fp_type=fp_type, random_state=random_state
                                                 , return_indices=True)    
    
    
#     print("Train: {} - Validate: {} - Test: {}".format(train_validate_test_splits[0], train_validate_test_splits[1], train_validate_test_splits[2]))
    train_inchikeys    = list(set(dataframe.iloc[train_validate_test_splits[0]][inchikey_column].values.tolist()))
    validate_inchikeys = list(set(dataframe.iloc[train_validate_test_splits[1]][inchikey_column].values.tolist()))
    test_inchikeys     = list(set(dataframe.iloc[train_validate_test_splits[2]][inchikey_column].values.tolist()))
    
    dataframe_train     = dataframe[dataframe[inchikey_column].isin(train_inchikeys)]
    dataframe_validate  = dataframe[dataframe[inchikey_column].isin(validate_inchikeys)]
    dataframe_test      = dataframe[dataframe[inchikey_column].isin(test_inchikeys)]
    print("Train: {} - Validate: {} - Test: {}".format(dataframe_train.shape, dataframe_validate.shape, dataframe_test.shape))
    print(dataframe_train.columns)
    return dataframe_train, dataframe_validate, dataframe_test


def min_max_train_validate_test_split(list_of_rdkit_representations, train_valid_ratios=[0.7, 0.15] , fp_type= "morgan", random_state=1, return_indices=False):
    """
    fp_types = { "morgan": "GetMorganFingerprint", "atom_pair": "GetAtomPairFingerprint", "top_torso": "GetTopologicalTorsionFingerprint"} 
    """
    try:
        input_mode =list_of_rdkit_representations[0].__class__.__name__
         
        picker = MaxMinPicker()
        fps    = None
        list_of_rdkit_representations = [x for x in list_of_rdkit_representations if not x is None]
        orginal_indices = range(len(list_of_rdkit_representations))
        fps = None
        
        if  input_mode == 'Mol':       
            if fp_type == "morgan":
                fps  = [GetMorganFingerprint(x,3) for x in list_of_rdkit_representations]
            elif fp_type == "atom_pair":
                fps  = [GetAtomPairFingerprint(x) for x in list_of_rdkit_representations]        
            elif fp_type == "top_torso":
                fps  = [GetTopologicalTorsionFingerprint(x) for x in list_of_rdkit_representations]
#             elif fp_type is None and fp_column is not None:
#                 fps = [mol.GetProp(fp_column).strip('][').split(', ') for mol in list_of_rdkit_representations]
#                 for i in fps:
#                     for j in range(len(i)):
#                         i[j] = int(i[j])   
        elif input_mode in ['UIntSparseIntVect', 'SparseIntVect', 'ExplicitBitVect']:            
            fps = list_of_rdkit_representations
            
        if fps is not None:
            nfps = len(fps)
            n_training_compounds = round(nfps*(train_valid_ratios[0]))
            n_valid_compounds    = round(nfps*(train_valid_ratios[1]))
            n_test_compounds     = nfps - n_training_compounds - n_valid_compounds
            print("{} - {} - {}".format(n_training_compounds, n_valid_compounds, n_test_compounds))

            ## Calculate the Dice dissimilarity between compounds
            def distij(i,j,fps=fps):
                return 1-DataStructs.DiceSimilarity(fps[i],fps[j])

            ## Retrieving training indices
            training_indices = list(picker.LazyPick(distij, nfps, n_training_compounds, seed=random_state))
    #         print(training_indices)

            ## Retrieving validation indices
            remaining_indices =  [x for x in orginal_indices if not x in training_indices]
            fps = [fps[j] for j in remaining_indices]
            nfps = len(fps)
    #         print("reamining: {}".format(nfps))        
            val = list(picker.LazyPick(distij, nfps, n_valid_compounds, seed=random_state))
    #         print(val)
            validation_indices = [remaining_indices[k] for k in val]

            ## Retrieving test indices
            test_indices = [l for l in orginal_indices if not l in training_indices + validation_indices]

            print("Indices (training):{} - {}".format(len(training_indices), training_indices[:2]) )
            print("Indices (validation):{} - {}".format(len(validation_indices), validation_indices[:1]) )
            print("Indices (test):{} - {}".format(len(test_indices), test_indices[:1]) )

            if return_indices:
                return training_indices, validation_indices, test_indices
            else:       
                return [list_of_rdkit_representations[i] for i in training_indices], [list_of_rdkit_representations[j] for j in validation_indices], [list_of_rdkit_representations[j] for j in test_indices]
        else:
            raise ValueError("Could not perform clustering and selection.\tFingerprint list = None")
    except Exception as e:
        print("Could not perform clustering and selection.")
        print(e)
        return None
    

    






# def min_max_train_test_split_df(dataframe, molecule_column, inchikey_column, test_ratio=0.2
#                                 , fp_type= "morgan", random_state=1, return_indices=False):
#     """
#     """
    
#     # Store the InChIKeys. These will be used to split the dataframe to ensure
#     # no molecule is both in the train and test sets.
#     if inchikey_column is None:
#         print("Computing and storing the InChiKeys...")
#         inchikey_column = "InChIKey"
#         dataframe[inchikey_column] = dataframe[molecule_column].apply(lambda x: MolToInchiKey(x))
    
#     dataframe.apply(lambda x: x[molecule_column].SetProp("InChIKey", x[inchikey_column]), axis=1)
    
#     # Select unique molecules (by InChiKey)
#     dataframe_single_ikeys = dataframe.drop_duplicates(subset=[inchikey_column], keep='first')     
#     list_of_rdkit_molecules = dataframe_single_ikeys[molecule_column].values.tolist()
    
#     # Split datasets
#     print("Splitting the dataset...")
#     train_test_splits = min_max_train_test_split(list_of_rdkit_molecules, test_ratio=test_ratio
#                                                  , fp_type=fp_type, random_state=random_state
#                                                  , return_indices=False)    
    
#     train_inchikeys   = list(set([mol.GetProp(inchikey_column) for mol in train_test_splits[0]]))
#     test_inchikeys    = list(set([mol2.GetProp(inchikey_column) for mol2 in train_test_splits[1]]))
    
#     print("Train/Test InChiKey Intersection = {}".format([i for i in train_inchikeys if i in test_inchikeys]))
#     print("Unique InChIKeys:: Train: {} - Test: {}".format(len(train_inchikeys), len(test_inchikeys)))
    
#     dataframe_train = dataframe[dataframe[inchikey_column].isin(train_inchikeys)]
#     dataframe_test  = dataframe[dataframe[inchikey_column].isin(test_inchikeys)]
#     print("Train: {} - Test: {}".format(dataframe_train.shape, dataframe_test.shape))
#     print(dataframe_train.columns)
#     return dataframe_train, dataframe_test

# def min_max_train_test_split(list_of_rdkit_molecules, test_ratio, fp_type= "morgan", random_state=1, return_indices=False):
#     """
#     fp_types = { "morgan": "GetMorganFingerprint", "atom_pair": "GetAtomPairFingerprint", "top_torso": "GetTopologicalTorsionFingerprint"} 
#     """
    
#     picker = MaxMinPicker()
#     fps  = None
    
#     if fp_type == "morgan":
#         fps  = [GetMorganFingerprint(x,3) for x in list_of_rdkit_molecules]
#     elif fp_type == "atom_pair":
#         fps  = [GetAtomPairFingerprint(x) for x in list_of_rdkit_molecules]        
#     elif fp_type == "top_torso":
#         fps  = [GetTopologicalTorsionFingerprint(x) for x in list_of_rdkit_molecules]  
                
#     nfps = len(fps)
#     n_training_compounds = round(nfps*(1-test_ratio))
    
#     ## Calculate the Dice dissimilarity between compounds
#     def distij(i,j,fps=fps):
#         return 1-DataStructs.DiceSimilarity(fps[i],fps[j])

#     train_indices = picker.LazyPick(distij, nfps, n_training_compounds ,seed=random_state)   
#     test_indices = [i for i in range(n_training_compounds) if not i in train_indices]
    
#     print("Indices (test): {}".format([x for x in train_indices if x in test_indices]) )
    
#     if return_indices:
#         return train_indices, test_indices
#     else:       
#         return [list_of_rdkit_molecules[i] for i in train_indices], [list_of_rdkit_molecules[j] for j in test_indices]
    