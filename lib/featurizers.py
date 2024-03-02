import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import HybridizationType
from rdkit.Chem.AllChem import Mol
from rdkit.Chem import (
    Draw,
    AllChem,
    Crippen,
    QED,
    Descriptors,
    GraphDescriptors,
    Fragments,
    MolToSmiles,
    GetPeriodicTable,
)
from rdkit.Chem import rdMolDescriptors as rdmdesc
from typing import List, Any
from lib import utilities
import re
import mordred
from mordred import descriptors, Calculator
from time import time

RDLogger.DisableLog("rdApp.*")


RDKIT_FRAGS = [
    [frag_name, MolToSmiles(eval(f"Fragments.{frag_name}").__defaults__[1])]
    for frag_name in dir(Fragments)
    if not re.match("^fr", frag_name) is None
]


DF_FUNC_GRPS_MINI = pd.DataFrame(
    [
        ["aldehyde", "[$([CX3H][#6]),$([CX3H2])]=[OX1]"],
        ["ketone", "[#6][CX3](=[OX1])[#6]"],
        [
            "carboxylic_ester",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[OX2][#6;!$(C=[O,N,S])]",
        ],
        [
            "ether",
            "[OX2]([#6;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])])[#6;!$(C([OX2])[O,S,#7,#15])]",
        ],
        [
            "thioether",
            "[SX2]([#6;!$(C([SX2])[O,S,#7,#15,F,Cl,Br,I])])[#6;!$(C([SX2])[O,S,#7,#15])]",
        ],
        ["lactone", "[#6][#6X3R](=[OX1])[#8X2][#6;!$(C=[O,N,S])]"],
        [
            "lactam",
            "	[#6R][#6X3R](=[OX1])[#7X3;$([H1][#6;!$(C=[O,N,S])]),$([H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        ],
        ["alcohol", "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]"],
        ["phenol", "[OX2H][c]"],
        ["amine", "[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]"],
        ["arylhalide", "[Cl,F,I,Br][c]"],
        ["alkylhalide", "[Cl,F,I,Br][CX4]"],
        ["urea", "[#7X3;!$([#7][!#6])][#6X3](=[OX1])[#7X3;!$([#7][!#6])]"],
        ["thiourea", "[#7X3;!$([#7][!#6])][#6X3](=[SX1])[#7X3;!$([#7][!#6])]"],
        [
            "sulfoniamide",
            "[SX4;$([H1]),$([H0][#6])](=[OX1])(=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
        ],
    ],
    columns=["name", "SMARTS"],
)

DF_FUNC_GRPS = pd.DataFrame(RDKIT_FRAGS, columns=["name", "SMARTS"])


class MoleculeFeaturizer(object):
    def __init__(
        self,
        features: List[str] = None,
        df_func_gps: pd.DataFrame = DF_FUNC_GRPS,
        label_col: str = "name",
    ):
        self.dim = 0
        self.allowable_calc_features = [
            "CalcExactMolWt",
            "CalcTPSA",
            "CalcNumAromaticRings",
            "CalcNumAliphaticRings",
            "CalcNumAliphaticCarbocycles",
            "CalcNumAliphaticHeterocycles",
            "CalcNumLipinskiHBA",
            "CalcNumLipinskiHBD",
            "CalcNumHBA",
            "CalcNumHBD",
            "CalcNumSpiroAtoms",
            "CalcNumRotatableBonds",
            "CalcNumLipinskiHBD",
            "CalcChi0n",
            "CalcChi0v",
            "CalcChi1n",
            "CalcChi1v",
            "CalcChi2n",
            "CalcChi2v",
            "CalcChi3n",
            "CalcChi3v",
            "CalcChi4n",
            "CalcChi4v",
            "CalcChi0n",
            "CalcChi0v",
            # , 'CalcChiNn', 'CalcChiNv'
        ]  #'CalcNumAtomStereoCenters',
        self.allowable_crippen_features = ["MolLogP"]
        self.allowable_qed_features = ["qed"]
        self.allowable_descriptors = ["MaxPartialCharge", "MinPartialCharge"]
        self.allowable_graph_descriptors = ["HallKierAlpha"]
        self.default_mordred_descs = [
            "ExtendedTopochemicalAtom",
            "Polarizability",
            "ZagrebIndex",
            "MoeType",
        ]
        self.df_func_gps = df_func_gps
        self.fgcp_label_col = label_col
        # print('self.df_func_gps', self.df_func_gps)

        if not features is None:
            self.rdkit_features = [
                f
                for f in features
                if f
                in self.allowable_calc_features
                + self.allowable_crippen_features
                + self.allowable_qed_features
                + self.allowable_descriptors
                + self.allowable_graph_descriptors
                # + self.mordred_descs
            ]

            # print('features', features)
            self.mordred_descs = [
                f
                for f in features
                if f in dir(mordred.descriptors)
                if re.match("^_", f) is None and f != "all"
            ]
        else:
            self.rdkit_features = (
                self.allowable_calc_features
                + self.allowable_crippen_features
                + self.allowable_qed_features
                + self.allowable_descriptors
                + self.allowable_graph_descriptors
            )

            self.mordred_descs = self.default_mordred_descs

        # print('self.rdkit_features ', self.rdkit_features )
        # print('self.mordred_descs ', self.mordred_descs )

    def compute_rdkit_properties(
        self,
        molecule,
        features: List[str] = None,
        label_col: str = "name",
        as_dict: bool = True,
        count_unique_bits: bool = True,
    ):
        properties = {}
        mordred_features, mordred_descs = [], None
        len_func_groups = 0

        failed_com_preps = []

        if features is None:
            features = self.rdkit_features

        if not molecule is None:
            if not (features is None or len(features) == 0):
                for prop in features:
                    try:
                        if prop in self.allowable_calc_features:
                            properties[prop] = eval(f"rdmdesc.{prop}")(molecule)
                        elif prop in self.allowable_crippen_features:
                            properties[prop] = eval(f"Crippen.{prop}")(molecule)
                        elif prop in self.allowable_qed_features:
                            properties[prop] = eval(f"QED.{prop}")(molecule)
                        elif prop in self.allowable_descriptors:
                            properties[prop] = eval(f"Descriptors.{prop}")(molecule)
                        elif prop in self.allowable_graph_descriptors:
                            properties[prop] = eval(f"GraphDescriptors.{prop}")(
                                molecule
                            )
                    except:
                        # print(f"Could not compute molecular property '{prop}'.")
                        failed_com_preps.append(prop)
                        properties[prop] = None

                if len(failed_com_preps) > 0:
                    print(
                        f"Could not compute molecular properties: {'; '.join(failed_com_preps)}"
                    )

            fcgps = None

            if not self.df_func_gps is None:
                fcgps = get_func_groups_pos_from_mol(
                    molecule,
                    df_func_gps=self.df_func_gps,
                    as_dict=True,
                    label_col=self.fgcp_label_col,
                    countUnique=count_unique_bits,
                )

                if fcgps is None and (not features is None):
                    fcgps = dict(
                        zip(
                            self.df_func_gps[self.fgcp_label_col].tolist(),
                            [None] * self.df_func_gps.shape[0],
                        )
                    )

                if not fcgps is None:
                    properties = dict(properties, **fcgps)
                    len_func_groups = len(fcgps)

            # print(f"properties = {properties}")
            if bool(properties):
                if not as_dict:
                    prop_values = []

                    for i in list(properties.values()):
                        if (
                            isinstance(i, float)
                            or isinstance(i, int)
                            or isinstance(i, bool)
                        ):
                            prop_values.append(i)
                        elif (
                            isinstance(i, list)
                            or isinstance(i, tuple)
                            or isinstance(i, set)
                        ):
                            prop_values += list(i)

                    return prop_values
                else:
                    # print("properties 2", properties)
                    return properties
                # print('prop_values', prop_values)
            else:
                return None

        else:
            for f in features:
                properties[f] = None

            if not self.df_func_gps is None:
                properties = dict(
                    properties,
                    **dict(
                        zip(
                            self.df_func_gps[self.fgcp_label_col].tolist(),
                            [None] * self.df_func_gps.shape[0],
                        )
                    ),
                )

            if not as_dict:
                prop_values = []

                for i in list(properties.values()):
                    if (
                        isinstance(i, float)
                        or isinstance(i, int)
                        or isinstance(i, bool)
                    ):
                        prop_values.append(i)
                    elif (
                        isinstance(i, list)
                        or isinstance(i, tuple)
                        or isinstance(i, set)
                    ):
                        prop_values += list(i)

                return prop_values
            else:

                return properties

    def compute_mordred_props(self, molecules, mordred_props: list = None):

        try:
            clean_props = None
            # print('mordred_props', mordred_props)
            if not mordred_props is None:
                clean_props = [eval(f"descriptors.{prop}") for prop in mordred_props]
            elif not len(self.mordred_descs) == 0:
                clean_props = [
                    eval(f"descriptors.{prop}") for prop in self.default_mordred_descs
                ]

            # print('clean_props', clean_props)
            if not (clean_props is None or len(clean_props) == 0):
                # print("Compute")
                return mordred.Calculator(descs=clean_props, ignore_3D=False).pandas(
                    molecules
                )
            else:
                return None
        except Exception as exp:
            print(f"Failed to compute mordred props: + {exp}")
            return None

    def compute_properties_for_mols(
        self, molecules, as_dataframe: bool = True, count_unique_bits: bool = True
    ):
        try:
            # if True:

            mordred_features, mordred_descs = [], None
            mols_df = pd.DataFrame(molecules, columns=["RMol"])
            # print(mols_df.head(2))

            t0 = time()
            rdkit_props = mols_df["RMol"].apply(
                lambda mol: self.compute_rdkit_properties(
                    molecule=mol, count_unique_bits=count_unique_bits
                )
            )
            # print('rdkit_props\n', rdkit_props[0], '\n', rdkit_props.values.tolist())
            t1 = time()
            t_rdkit = t1 - t0
            print(f"RDKIT property calculation: {round(t_rdkit, 3)} seconds.")
            rdkit_props_is_all_none = (
                rdkit_props[0] is None and len(rdkit_props.unique()) == 1
            )
            # print('rdkit_props_is_all_none', rdkit_props_is_all_none)
            # print('self.mordred_descs', self.mordred_descs)
            mordred_props_df = self.compute_mordred_props(molecules, self.mordred_descs)
            t2 = time()
            t_mordred = t2 - t1
            print(f"MORDRED property calculation: {round(t_mordred, 3)} seconds.")
            # print(mordred_props_df is None)
            # if rdkit_props =

            properties = None
            if not (rdkit_props_is_all_none or mordred_props_df is None):
                rdkit_props_df = pd.DataFrame(rdkit_props.values.tolist())
                if as_dataframe:
                    return pd.concat([rdkit_props_df, mordred_props_df], axis=1)
                else:
                    return pd.concat(
                        [rdkit_props_df, mordred_props_df], axis=1
                    ).to_dict(orient="records")

            elif rdkit_props_is_all_none:
                if as_dataframe:
                    return mordred_props_df
                elif not mordred_props_df is None:
                    return mordred_props_df.to_dict(orient="records")
                else:
                    return None

            elif mordred_props_df is None:
                if as_dataframe:
                    if not rdkit_props is None:
                        return pd.DataFrame(rdkit_props.values)
                    else:
                        return None
                else:
                    # print('rdkit_props_df', pd.DataFrame(rdkit_props.values.tolist()).to_dict(orient='records'))
                    # print('rdkit_props', rdkit_props)
                    return rdkit_props.values.tolist()
                    # return pd.DataFrame(rdkit_props.values.tolist()).to_dict(orient='records')

        except Exception as exp:
            print(f"Failed to compute props: {exp}")
            return None


def get_func_groups_pos_from_mol(
    mol,
    df_func_gps: pd.DataFrame = DF_FUNC_GRPS,
    as_dict: bool = False,
    label_col: str = "name",
    countUnique: bool = True,
):
    onehot_func_gps = np.zeros(len(df_func_gps), dtype=int)
    try:
        for i, smiles in enumerate(df_func_gps["SMARTS"]):
            substruct = Chem.MolFromSmarts(smiles)
            match_pos = mol.GetSubstructMatches(substruct)
            if countUnique:
                onehot_func_gps[i] = len(match_pos)
            else:
                onehot_func_gps[i] = int(len(match_pos) > 0)
        if not as_dict:
            return onehot_func_gps
        else:
            return pd.Series(
                onehot_func_gps.tolist(), index=df_func_gps[label_col]
            ).to_dict()
    except:
        for i, smiles in enumerate(df_func_gps["SMARTS"]):
            onehot_func_gps[i] = None
        if not as_dict:
            return onehot_func_gps
        else:
            return pd.Series(
                onehot_func_gps.tolist(), index=df_func_gps[label_col]
            ).to_dict()


def get_func_groups_pos(
    smiles,
    df_func_gps: pd.DataFrame = DF_FUNC_GRPS,
    as_dict: bool = False,
    label_col: str = "name",
):
    mol = Chem.MolFromSmiles(smiles)
    if not mol is None:
        onehot_func_gps = np.zeros(len(df_func_gps), dtype=int)
        for i, smiles in enumerate(df_func_gps["SMARTS"]):
            substruct = Chem.MolFromSmarts(smiles)
            match_pos = mol.GetSubstructMatches(substruct)
            onehot_func_gps[i] = len(match_pos)
        if not as_dict:
            return onehot_func_gps
        else:
            return pd.Series(
                onehot_func_gps.tolist(), index=df_func_gps[label_col]
            ).to_dict()
    else:
        return None


def num_halogen_atoms(molecule: Mol):
    if not molecule:
        return None
    else:
        return len(
            [
                atom
                for atom in molecule.GetAtoms()
                if atom.GetSymbol() in ("Cl", "Br", "I", "F")
            ]
        )


def halogen_weight(molecule: Mol):
    if not molecule:
        return None
    else:
        halogen_atoms = [
            atom
            for atom in molecule.GetAtoms()
            if atom.GetSymbol() in ("Cl", "Br", "I", "F")
        ]
        return sum([atom.GetMass() for atom in halogen_atoms])


def halogen_weight_ratio(molecule: Mol):
    if not molecule:
        return None
    else:
        molecular_weight = Descriptors.MolWt(molecule)
        if molecular_weight == 0.0:
            return None
        return halogen_weight(molecule) / molecular_weight


def num_sp3_carbons(molecule: Mol):
    if molecule is None:
        return None
    else:
        sp3_carbons = [
            atom
            for atom in molecule.GetAtoms()
            if atom.GetSymbol() and atom.GetHybridization() == HybridizationType.SP3
        ]
        return len(sp3_carbons)


def frac_sp3_carbons(molecule: Mol):
    if molecule is None:
        return None
    else:
        num_sp3_carbs = num_sp3_carbons(molecule)
        return num_sp3_carbs / molecule.GetNumAtoms()


def matches_lipinski_ro5(num_hba, num_hbd, molwt, logp):
    return all([num_hba <= 10, num_hbd <= 5, molwt <= 500, logp <= 5])


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
