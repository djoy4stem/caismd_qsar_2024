import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from lib import featurizers, utilities

from rdkit.Chem import Draw, AllChem, PandasTools, MolFromSequence, MolToSmiles, MolToInchiKey, Descriptors, GraphDescriptors

def select_features(df, target_col, model_type='random_forest', threshold=None, **kwargs):
    """
    Perform feature selection using a specified model.

    Args:
    - df (DataFrame): Input DataFrame containing features and target.
    - target_col (str): Name of the target column.
    - model_type (str): Type of model to use for feature selection. Options: 'random_forest', 'lightgbm'.
    - threshold (float): Threshold for feature selection. If None, uses median of feature importances.
    - **kwargs: Additional keyword arguments to pass to the model.

    Returns:
    - feature_importances (Series): Series containing feature importances.
    - selected_features (list): List of selected feature names.
    """
    # Split data into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Initialize the model
    if model_type == 'random_forest':
        model = RandomForestClassifier(**kwargs)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**kwargs)
    else:
        raise ValueError("Invalid model_type. Choose 'random_forest' or 'lightgbm'.")

    # Train the model and get feature importances
    model.fit(X, y)
    
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    elif hasattr(model, 'coef_'):
        feature_importances = pd.Series(np.abs(model.coef_[0]), index=X.columns)
    else:
        raise AttributeError("Model does not have attribute 'feature_importances_' or 'coef_'.")

    # Select features based on threshold or median
    if threshold is None:
        threshold = feature_importances.median()
    
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()

    return feature_importances, selected_features




def remove_conflicting_target_values(dataframe, target, inchikey_column):
    groups = []
    
    for name, group in dataframe.groupby(inchikey_column):
        n_conflicts = 0
        unique_target_values =  group[target].unique().tolist()
        if len(unique_target_values) > 1:
            n_conflicts += 1
            print("InchIKey {} has {} conflicting {} values. All associated samples will be removed.".format(name, len(unique_target_values), target))
        else:
    #         print("{} - {} - {}".format(group.shape, group[target].values, mean_target_value))
            unique_row         = group.drop_duplicates(subset=[inchikey_column], keep='first')
            groups.append(unique_row)
    #     print(groups)
    if n_conflicts > 0:
        print("Number of unique compounds with conflicts: {}".format(n_conflicts))
    return pd.concat(groups, axis=0)    




def clean_features(features_df, columns_to_clean=None, columns_to_scale=None, standardizer=None, strategy_num='mean', strategy_cat='most_frequent'):
    
    if columns_to_clean is None:
        columns_to_clean = features_df.columns
    numeric_cols = [col for col in columns_to_clean if features_df[col].dtype in [int, float]]
    bool_cols    = [col for col in columns_to_clean if features_df[col].dtype == bool]
    # print("columns_to_clean= ", columns_to_clean)
    # print("numeric_cols = ", numeric_cols)
    # print("bool_cols = ", bool_cols)

    cleaned_features_df = features_df
    if len(numeric_cols)>0:
        imputer_num = SimpleImputer(missing_values=np.nan, strategy=strategy_num)
        num_cleaned =  imputer_num.fit_transform(cleaned_features_df[numeric_cols])
        # print(num_cleaned)
        cleaned_features_df[numeric_cols] = pd.DataFrame(num_cleaned, columns = numeric_cols)

    if not strategy_cat is None:
        categorical_cols = [col for col in columns_to_clean if features_df[col].dtype == 'object']
        if len(categorical_cols)>0:
            imputer_cat = SimpleImputer(missing_values=np.nan, strategy=strategy_cat)
            cat_cleaned = imputer_cat.fit_transform(cleaned_features_df[categorical_cols])
            cleaned_features_df[categorical_cols] = pd.DataFrame(cat_cleaned, columns = categorical_cols)

    if not standardizer is None:
        if columns_to_scale is None:
            columns_to_scale = [c for c in columns_to_clean if not c in bool_cols]

        if len(columns_to_scale)>0:
            cleaned_features_df[columns_to_scale] = standardizer.fit_transform(X=cleaned_features_df[columns_to_scale])   
            # print(scaled_)

    cleaned_features_df.dropna(axis=0, inplace=True)

    return cleaned_features_df


def get_features_by_correlation(dataframe, target_column):
    # Calculate correlation with target column
    correlations = dataframe.corr()[target_column].abs().sort_values(ascending=False)
    
    # Remove the target column itself from the list
    correlations = correlations.drop(target_column)
    
    # Get the feature names sorted by correlation
    feature_names = correlations.index.tolist()
    
    return feature_names



def clean_data(dd2_data_df, target_column, smiles_column, mol_column='RMol'):
    print(f"Original shape: {dd2_data_df.shape}")
    print(f"Number of molecules with a missing SMILES string: {dd2_data_df[smiles_column].isna().sum()}/{dd2_data_df.shape[0]}")

    # Removing rows with a missing SMILES string
    dd2_data_df.dropna(subset=[smiles_column], axis=0, inplace=True)


    # Remove rows where 'PUBCHEM_ACTIVITY_OUTCOME' is not defined
    dd2_data_df.dropna(subset=['PUBCHEM_ACTIVITY_OUTCOME'], axis=0, inplace=True)


    # Add a column with 0/1 activity values
    dd2_data_df[target_column] = dd2_data_df['PUBCHEM_ACTIVITY_OUTCOME'].apply(lambda x: 1 if x =='Active' else 0)
    print("\nNumber of row with undefined label: ", dd2_data_df[target_column].isna().sum())


    # Remove rows where target_column is not defined:
    dd2_data_df.dropna(subset=[target_column], axis=0, inplace=True)
    # print(dd2_data_df.isna().sum())

    # Adding molecule (Mol) objects to the dataframe
    PandasTools.AddMoleculeColumnToFrame(dd2_data_df, smiles_column, mol_column)


    dd2_data_df['largest_frag']        = dd2_data_df[mol_column].apply(utilities.get_largest_fragment_from_mol)

    dd2_data_df['largest_frag_smiles'] = dd2_data_df['largest_frag'].apply(lambda mol: MolToSmiles(utilities.sanitize_molecule(mol, add_explicit_h=True)))

    dd2_data_df.dropna(subset=['largest_frag_smiles'], axis=0, inplace=True)

    dd2_data_df = dd2_data_df.reset_index(drop=True) ## Reset the index to avoid issues when manipulating based on indices


    dd2_data_df['largest_frag_ikey'] = dd2_data_df['largest_frag'].apply(MolToInchiKey)
    duplicates = dd2_data_df[dd2_data_df.duplicated(subset=['largest_frag_ikey'], keep=False)].sort_values(by='largest_frag_ikey')
    print(f"ddf: {dd2_data_df.shape}")
    # print(duplicates.shape)
    print("\n\n")
    print(f"Number of molecules with a missing largest fragment SMILES string: {dd2_data_df['largest_frag_smiles'].isna().sum()}/{dd2_data_df.shape[0]}")
    print(f"\nNumber of unique inchikeys: {dd2_data_df['largest_frag_ikey'].unique().size}/{dd2_data_df.shape[0]}")

    number_of_duplicates = duplicates.shape[0]
    print(f"\nNumber of duplicates inchikeys: {duplicates.shape[0]} - e.g.: {duplicates.index[:2]}")



    if number_of_duplicates>0:
        print("\n\nExamples of duplicates", duplicates[['largest_frag_ikey', target_column]].head(10))
        print("\n\nRemoving duplicates...\n")


        #### It can happen that replicates measurements yield values at different ends of the sepctrum
        #### This might be indicate a low reproducibility of the experimental setting, especially if 
        #### this scenario occurs a lot.
        #### In our case, there are a few examples. We will remove cases where the 'PCT_INHIB_DD2'
        #### values for the same compound are apart by more than 20%
        print("Removing replicates with conflicting values.")
        groups = []
        indices_to_remove=[]
        for name, group in duplicates.groupby('largest_frag_ikey'):
            if group[target_column].unique().size>1:
                indices_to_remove.append(list(group.index))
                # print(group[['largest_frag_ikey', 'PCT_INHIB_DD2', target_column]])
                # print(list(group.index))

        # print(f"indices_to_remove = {indices_to_remove}")
        if len(indices_to_remove):
            print(indices_to_remove[0])
            print("Example of f molecule with conflicting replicate measurements...\n")
            print(dd2_data_df[['largest_frag_ikey', 'PCT_INHIB_DD2', target_column]].iloc[indices_to_remove[0]])
            indices_to_remove = utilities.flatten_list(indices_to_remove)
            print(f"\nNumber of samples to remove: = {len(indices_to_remove)}")
            dd2_data_df.drop(indices_to_remove, axis=0, inplace=True)
            dd2_data_df = dd2_data_df.reset_index(drop=True) ## Reset the index to avoid issues when manipulating based on indices

        dd2_data_df = pd.concat([dd2_data_df]+groups, axis=0)
        print("dd2_data_df.shape (after removing conflicting replicates) = ", dd2_data_df.shape)




    # # print(f"Number of molecules with a negative target value: {dd2_data_df[dd2_data_df[target_column]<0].shape[0]}/{dd2_data_df.shape[0]} - e.g.:{dd2_data_df[dd2_data_df[target_column]<0].index[:2]}")
    # # print(f"Number of molecules with target value>100: {dd2_data_df[dd2_data_df[target_column]>100].shape[0]}/{dd2_data_df.shape[0]} - e.g.:{dd2_data_df[dd2_data_df[target_column]>100].index[:2]}")
    # dd2_data_df[['PUBCHEM_CID', 'largest_frag_ikey', target_column]].loc[[8, 9, 272, 458, 11041, 11042],:]

    return dd2_data_df

def add_custom_features(data_df:pd.DataFrame, bool_to_int:bool=True, mol_column='Mol'):
    data_df['num_halo_atoms']    = data_df[mol_column].apply(featurizers.num_halogen_atoms)
    data_df['halo_weight_ratio'] = data_df[mol_column].apply(featurizers.halogen_weight_ratio)
    data_df['num_s3_carbons']    = data_df[mol_column].apply(featurizers.num_sp3_carbons)
    data_df['frac_spe_carbons']  = data_df[mol_column].apply(featurizers.frac_sp3_carbons)
    data_df['lipinski_ro5'] = data_df.apply(lambda x: featurizers.matches_lipinski_ro5(num_hba=x['CalcNumLipinskiHBA'], num_hbd=x['CalcNumLipinskiHBD'], molwt=x['CalcExactMolWt'], logp=x['MolLogP'] ), axis=1)

    if bool_to_int:
        ## Convert boolean to integer. When merging to other data, the column type can be set to 'object', thus causing issues when training models.
        data_df['lipinski_ro5'] = data_df['lipinski_ro5'].apply(lambda x: int(x))


    return data_df