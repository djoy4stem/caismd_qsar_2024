import os, sys
import numpy as np
import pandas as pd
from typing import List
from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles


from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle

import re

from rdkit.Chem import Draw, AllChem, PandasTools
from rdkit.Chem import MolFromSmiles, MolToSmiles

import shap

from typing import List, Union, Any, Tuple

from lib import utilities, featurizers, preprocessing


MOL_FEATURIZER_ = featurizers.MoleculeFeaturizer(features=None)
LGBM_classification_model_ = utilities.get_from_pkl(
    f"models/best_desc_lgbm.pkl"
)


def process_molecule(molecule: AllChem.Mol):
    try:
        if not molecule is None:
            molecule = utilities.get_largest_fragment_from_mol(molecule)
            molecule = utilities.sanitize_molecule(molecule, add_explicit_h=True)
            return molecule
        else:
            return None
    except Exception as exp:
        print(f"Failed to process molecules: {exp}")


def featurize_molecules(
    molecules: Union[List[AllChem.Mol], np.ndarray, pd.DataFrame],
    count_unique_bits: bool = False,
    mol_col: str = "Mol",
):

    if not molecules is None:
        features = None
        if isinstance(molecules, (List, np.ndarray)):
            features = pd.DataFrame(molecules, columns=[mol_col])
            x = pd.DataFrame(
                features[mol_col]
                .apply(
                    lambda mol: MOL_FEATURIZER_.compute_properties_for_mols(
                        molecules=[mol], as_dataframe=False, count_unique_bits=False
                    )[0]
                )
                .values.tolist()
            )

            features = pd.concat([features, x], axis=1)

        elif isinstance(molecules, pd.DataFrame):
            x = pd.DataFrame(
                molecules[mol_col]
                .apply(
                    lambda mol: MOL_FEATURIZER_.compute_properties_for_mols(
                        molecules=[mol], as_dataframe=False, count_unique_bits=False
                    )[0]
                )
                .values.tolist()
            )
            features = pd.concat([molecules, x], axis=1)

        features = preprocessing.add_custom_features(
            features, bool_to_int=True, mol_column=mol_col
        )
        features.drop([mol_col], axis=1, inplace=True)

        return features

    else:
        return None


def featurize_and_predict_from_mols(molecules_df: pd.DataFrame, mol_col="Mol"):
    mol_features = featurize_molecules(
        molecules=molecules_df, count_unique_bits=False, mol_col=mol_col
    )

    print("mol_features", mol_features.shape)
    # print(LGBM_classification_model_.feature_name_)
    predictions_probas = [
        x[1] if not x is None else None
        for x in LGBM_classification_model_.predict_proba(
            mol_features[LGBM_classification_model_.feature_name_]
        )
    ]
    predicted_class = [
        pr > 0.5 if not (pr is None) else None for pr in predictions_probas
    ]
    molecules_df["DD2_inbition_proba"] = predictions_probas
    molecules_df["DD2_inbitor"] = predicted_class

    return molecules_df


def featurize_and_predict_from_smiles(
    smiles: Union[List[AllChem.Mol], np.ndarray, pd.DataFrame],
    smiles_col: str = "SMILES",
    **kwargs,
):

    try:
        if isinstance(smiles, (List, np.ndarray)):
            molecules = pd.DataFrame(
                [MolFromSmiles(smi) for smi in smiles], columns=["Mol"]
            )
            return featurize_and_predict_from_mols(
                molecules_df=molecules, mol_col="Mol"
            ).drop([mol_col], axis=1)

        elif isinstance(smiles, pd.DataFrame):
            mol_col = kwargs.get("mol_col", None) or "Mol"
            PandasTools.AddMoleculeColumnToFrame(
                smiles, smilesCol=smiles_col, molCol=mol_col
            )

            return featurize_and_predict_from_mols(
                molecules_df=smiles, mol_col=mol_col
            ).drop([mol_col], axis=1)

    except Exception as exp:
        print(f"Failed to predict DD2 inhibition.\n\t{exp}")

