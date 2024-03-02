import os
import sys

from itertools import chain
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib import colors as mcolors


import pickle

import random


from rdkit import Chem, DataStructs
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import rdmolops, Draw
from rdkit.Chem.AllChem import Mol
from rdkit.Chem.rdMolDescriptors import (
    GetMorganFingerprint,
    GetAtomPairFingerprint,
    GetTopologicalTorsionFingerprint,
)
from rdkit.Chem import (
    PandasTools,
    AllChem,
    MolFromSmiles,
    Draw,
    MolToInchiKey,
    MolToSmiles,
)
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from torch import manual_seed, cuda, backends, Generator

from typing import List, Union, Any, Tuple


# from numba import jit
# os.environ['NUMBA_DISABLE_JIT'] = '1'


def set_seeds(seed: int = None):

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # controls the hash seed for hash-based operations so they are reproducible, if seed is not None

    random.seed(seed)
    np.random.seed(seed)

    # # For general torch operations
    # manual_seed(seed)

    # # For operations that happen on the CPU
    # if cuda.is_available():
    #     cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU.
    #     cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs.
    #     Generator().manual_seed(seed)
    #     if not seed is None:
    #         backends.cudnn.deterministic = True # causes cuDNN to only use deterministic convolution algorithms
    #     backends.cudnn.benchmark = False # causes cuDNN to benchmark multiple convolution algorithms and select the fastest

    # # torch.use_deterministic_algorithms(True)


##### MOLECULES
def randomize_smiles(smiles, isomericSmiles=False):
    "Take a SMILES string, and return a valid, randomized, abd equivalent SMILES string"
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    random = Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=isomericSmiles
    )
    return random


def add_numbers_to_mol_atoms(mol):
    for atom in mol.GetAtoms():
        # For each atom, set the property "atomNote" to a index+1 of the atom
        atom.SetProp("atomNote", str(atom.GetIdx() + 1))


def molecule_from_smiles(smiles: str, add_explicit_h: bool = True):
    ### Modified version of the code from
    ### https://keras.io/examples/graph/mpnn-molecular-graphs/
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    return sanitize_molecule(molecule, add_explicit_h)


def sanitize_molecule(molecule: Mol, add_explicit_h: bool = True):
    if not molecule is None:
        # print(Chem.MolToSmiles(molecule))
        # If sanitization is unsuccessful, catch the error, and try again without
        # the sanitization step that caused the error
        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(
                molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag
            )
        if add_explicit_h:
            molecule = Chem.AddHs(molecule)
            # print(Chem.MolToSmiles(molecule))
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def get_largest_fragment_from_smiles(smiles: str, return_as_smiles: bool = False):
    molecule = MolFromSmiles(smiles)
    return get_largest_fragment_from_mol(molecule, return_as_smiles)


def get_largest_fragment_from_mol(molecule: Mol, return_as_smiles: bool = False):
    if molecule is not None:
        try:
            mol_frags = rdmolops.GetMolFrags(molecule, asMols=True)
            largest_mol = max(
                mol_frags, default=molecule, key=lambda m: m.GetNumAtoms()
            )
            if return_as_smiles:
                largest_mol = MolToSmiles(largest_mol)
            return largest_mol
        except Exception as exp:
            print(f"Could not get largest fragment: {exp}")
    else:
        return None


### MEMORY


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.

    Borrowed from https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def flatten_list(mylist: List[Any]):
    return list(chain(*mylist))


def calculate_fingerprints(molecules, fingerprint_type="morgan", radius=2, nBits=1024):

    valid_fingerprints = ["morgan", "avalon", "atom-pair", "maccs"]
    if fingerprint_type not in valid_fingerprints:
        raise ValueError(
            f"Invalid fingerprint type. Choose from {', '.join(valid_fingerprints)}."
        )

    # Create fingerprint generator based on type
    if fingerprint_type == "morgan":

        def fp_generator(mol):  # Pass the molecule as an argument
            return Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=nBits
            )

    elif fingerprint_type == "avalon":

        def fp_generator(mol):
            return Chem.GetAvalonFP(mol, nBits=nBits)

    elif fingerprint_type == "atom-pair":

        def fp_generator(mol):
            return Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)

    elif fingerprint_type == "maccs":

        def fp_generator(mol):
            return Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)

    else:
        raise ValueError(
            f"Internal error: Unknown fingerprint type {fingerprint_type}."
        )

    # Generate fingerprints
    fingerprints = [fp_generator(mol) for mol in molecules]

    return fingerprints


def get_representations(
    dataset,
    feature_list=None,
    fingerprint_type="morgan",
    radius=2,
    nBits=1024,
    as_array=True,
):
    features = None
    if isinstance(dataset, pd.DataFrame):  # If dataset is a DataFrame
        if feature_list is None or len(feature_list) == 0:
            print("Error: feature_list cannot be None or empty for DataFrame input.")
            return
        else:
            features = dataset[feature_list].values
            return features
    elif isinstance(
        dataset, (list, np.ndarray)
    ):  # If dataset is a list of RDKit molecules
        # print("\n\nLIST")
        if all(isinstance(mol, Chem.Mol) for mol in dataset):
            features = calculate_fingerprints(
                molecules=dataset,
                fingerprint_type=fingerprint_type,
                radius=2,
                nBits=nBits,
            )
            # print("features", features)
            if as_array:
                return np.array(features)
            else:
                return features
        elif all(
            isinstance(feats, (list, np.ndarray))
            and isinstance(feats[0], (int, float, complex))
            for feats in dataset
        ):
            # print("\n\nREST")
            return dataset
        else:
            print("dataset[0] = ", dataset[0])
    else:
        print(f"Error: Invalid dataset.")
        return


def avg_tanimoto_coefficient_from_fps(fingerprints):

    total_pairs = len(fingerprints) * (len(fingerprints) - 1) / 2
    similarities = Chem.DataStructs.BulkTanimotoSimilarity(
        fingerprints[0], fingerprints
    )
    avg_sim = sum(similarities[1:]) / total_pairs

    return avg_sim


def avg_tanimoto_dissimilarity_from_fps(fingerprints):

    total_pairs = len(fingerprints) * (len(fingerprints) - 1) / 2
    similarities = Chem.DataStructs.BulkTanimotoSimilarity(
        fingerprints[0], fingerprints
    )
    # Convert similarities to distances (1 - similarity)
    dissimilarities = [(1 - s) for s in similarities[1:]]
    avg_dist = sum(dissimilarities[1:]) / total_pairs

    return avg_dist


def avg_tanimoto_coefficient(
    molecules, fingerprint_type="morgan", radius=2, bitLength=2048
):
    """
    Computes the average Tanimoto coefficient between molecules using specified fingerprints.

    Args:
        molecules (list[rdkit.Chem.Mol]): List of RDKit molecule objects.
        fingerprint_type (str, optional): Type of fingerprint to use.
            Can be "morgan", "avalon", or "atom-pair". Defaults to "morgan".
        radius (int, optional): Radius for Morgan fingerprints. Defaults to 2.
        bitLength (int, optional): Bit length for fingerprints. Defaults to 2048.

    Returns:
        float: Average Tanimoto coefficient between molecules.

    Raises:
        ValueError: If an invalid fingerprint type is specified.
    """

    fingerprints = calculate_fingerprints(
        molecules=molecules,
        fingerprint_type=fingerprint_type,
        radius=radius,
        bitLength=bitLength,
    )

    # Use efficient BulkTanimotoSimilarity
    avg_sim = avg_tanimoto_coefficient_from_fps(fingerprints)

    return avg_sim


def avg_tanimoto_dissimilarity(
    molecules, fingerprint_type="morgan", radius=2, bitLength=2048
):
    """
    Computes the average Tanimoto coefficient between molecules using specified fingerprints.

    Args:
        molecules (list[rdkit.Chem.Mol]): List of RDKit molecule objects.
        fingerprint_type (str, optional): Type of fingerprint to use.
            Can be "morgan", "avalon", or "atom-pair". Defaults to "morgan".
        radius (int, optional): Radius for Morgan fingerprints. Defaults to 2.
        bitLength (int, optional): Bit length for fingerprints. Defaults to 2048.

    Returns:
        float: Average Tanimoto coefficient between molecules.

    Raises:
        ValueError: If an invalid fingerprint type is specified.
    """

    fingerprints = calculate_fingerprints(
        molecules=molecules,
        fingerprint_type=fingerprint_type,
        radius=radius,
        bitLength=bitLength,
    )

    # Use efficient BulkTanimotoSimilarity
    avg_sim = avg_tanimoto_dissimilarity_from_fps(fingerprints)

    return avg_sim


# @jit(nopython=True)
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# @jit(nopython=True)
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


# @jit(nopython=True)
def dot_product(x, y):
    return np.dot(x, y)


# @jit(nopython=True)
def norm(x):
    return np.sqrt(np.dot(x, x))


# @jit(nopython=True)
def cosine_distance(x, y):
    dot_product_xy = dot_product(x, y)
    norm_x = norm(x)
    norm_y = norm(y)
    return 1 - (dot_product_xy / (norm_x * norm_y + 1e-9))


# @jit(nopython=True)
def normalized_euclidean_distance(x, y):
    return euclidean_distance(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))


# @jit(nopython=True)
def normalized_manhattan_distance(x, y):
    return manhattan_distance(x, y) / (np.sum(np.abs(x)) + np.sum(np.abs(y)))


# @jit(nopython=True)
def average_dissimilarity(vectors, metric="euclidean", normalize=False):
    n = len(vectors)
    total_dissimilarity = 0.0

    if metric == "euclidean":
        if normalize:
            distance_func = normalized_euclidean_distance
        else:
            distance_func = euclidean_distance
    elif metric == "manhattan":
        if normalize:
            distance_func = normalized_manhattan_distance
        else:
            distance_func = manhattan_distance
    elif metric == "cosine":
        distance_func = cosine_distance
    else:
        raise ValueError(
            "Invalid metric. Choose from 'euclidean', 'manhattan', or 'cosine'."
        )

    for i in range(n):
        for j in range(i + 1, n):
            total_dissimilarity += distance_func(vectors[i], vectors[j])

    # Average dissimilarity is the total dissimilarity divided by the number of pairwise comparisons
    return total_dissimilarity / (n * (n - 1) / 2)


def save_to_pkl(my_object, filename):
    with open(filename, "wb") as f:
        pickle.dump(my_object, f)
    f.close()


def get_from_pkl(filename):
    my_object = None
    with open(filename, "rb") as f:
        my_object = pickle.load(f)
    return my_object


def get_features_for_folds(kfold_idx, features_df):
    folds_with_features = []
    for fold in kfold_idx:
        # print([len(f) for f in fold])
        dt = [features_df.iloc[f] for f in fold]
        # print([x.shape for x in dt])
        folds_with_features.append(dt)

    return folds_with_features
