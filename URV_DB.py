import os
from Bio.PDB import PDBParser
from Bio.PDB import PPBuilder
import pandas as pd
import csv
from rdkit import Chem
import os
from sklearn.model_selection import KFold

def get_sequence_from_PDB(PDB_id, URV_protein_folderpath):
    
    
    parser = PDBParser(QUIET=True)
    file= URV_protein_folderpath + '/' + PDB_id + '_protein' + '.pdb'
    structure = parser.get_structure(PDB_id.lower(), file)
    seq=[]

    # print(PDB_id)
    for model in structure:
        for chain in model:
            
            # # Iterate over all residues in the chain
            # for residue in chain:
            #     # Check if the residue is an amino acid
            #     if PDB.is_aa(residue):
            #         # Get the one-letter amino acid code and append to the sequence
            #         sequence += PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(residue.get_resname()))
            if(len(PPBuilder().build_peptides(chain)) > 0):
                seq.append(PPBuilder().build_peptides(chain)[0].get_sequence())
    # print(seq)
    return seq  # The output is a list with the sequences of each chain

def Protein_Generate(URV_affinity_filepath, URV_protein_folderpath, URV_logs_folderpath):
    # Read the Affinity TXT file into a dataframe
    dataframe = pd.read_csv(URV_affinity_filepath,sep=r'\s+',header=None)  # Use the appropriate delimiter if your file is not tab-separated

    # Display the dataframe
    #print(dataframe)

    # Select the first column
    protiens = dataframe.iloc[:, 0]
    protiens = [item.replace("_ligand", "").strip() for item in protiens]
    # print(len(protiens))
    # print(protiens)

    # Loop over the values in the first column
    sequences = []
    protien_df = pd.DataFrame(columns=['protien', 'sequence_length', 'first_sequence', 'first_sequence_length', 'second_sequence', 'second_sequence_length', 'selected_sequence'])
    for protien in protiens:

        sequence = get_sequence_from_PDB(protien, URV_protein_folderpath)
        #print(len(sequence))
        #print(sequence[0])

        if(len(sequence) == 1):
            selected_sequence = sequence[0]
        elif(len(sequence) == 2):
            if(len(sequence[0]) >= len(sequence[1])):
                selected_sequence = sequence[0]
            else: 
                selected_sequence = sequence[1]

        new_row = {'protien': protien, 'sequence_length': len(sequence), 'first_sequence': sequence[0], 'first_sequence_length': len(sequence[0]), 'second_sequence': sequence[1] if len(sequence) == 2 else '' , 'second_sequence_length': len(sequence[1]) if len(sequence) == 2 else '', 'selected_sequence':selected_sequence}
        protien_df = pd.concat([protien_df, pd.DataFrame([new_row])], ignore_index=True)

        # Cast a column as a string
        protien_df = protien_df.astype({'protien': str})
    # print(sequences)
    # print(len(sequences))
    protien_df.to_csv(URV_logs_folderpath + '/protiens_file.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)  # Set index=False to avoid writing row indices
    #protien_df.head(152)

def ligand_Generate(URV_ligandsdf_folderpath, URV_logs_folderpath):

    sdf_ligand_df = pd.DataFrame(columns=['Filepath', 'Id', 'Ligand_state','SMILES', 'Error'])
    for filename in os.listdir(URV_ligandsdf_folderpath):
        sdf_file_path = os.path.join(URV_ligandsdf_folderpath, filename)
        new_row = {'Filepath': sdf_file_path, 'Id': filename[:filename.index('_')], 'Ligand_state': '','Error': '' }
        sdf_ligand_df = pd.concat([sdf_ligand_df, pd.DataFrame([new_row])], ignore_index=True)

        try:
            
            # Read the .sdf file
            supplier = Chem.SDMolSupplier(sdf_file_path, sanitize = False,removeHs=False, strictParsing=False)

            # Iterate over the molecules in the SDF file
            for mol in supplier:
                # Check if the molecule was successfully loaded
                if mol is not None:
                    
                    try:
                        # Perform structure validation
                        Chem.SanitizeMol(mol)
                        sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Ligand_state'] = 'valid'

                    except ValueError as e:
                        print("Error: " + str(e))
                        sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Ligand_state'] = 'invalid'
                        sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Error'] = str(e)
                    
                    # For example, compute 3D coordinates using the AllChem module
                    #AllChem.Compute2DCoords(mol)
                    
                    # Print molecule information
                    #print(Chem.MolToMolBlock(mol))

                    # Convert molecule to SMILES notation
                    smiles = Chem.MolToSmiles(mol)
                    sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'SMILES'] = smiles
                    # print("SMILES notation of ",sdf_file_path," : ", smiles)
                else:
                    #print("Failed to load molecule from .sdf file.")
                    sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Ligand_state'] = 'failed to load'
                    raise ValueError("Failed to parse sdf file. The file may be empty or invalid.")
    
        except FileNotFoundError:
            print("File not found.")
            sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Error'] = "File not found."
        except IOError:
            print("Error reading the file.")
            sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Error'] = "Error reading the file."
        except ValueError as e:
            print("Error:", e)
            sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Error'] = str(e)
        except Exception as e:
            print("An unexpected error occurred:", e)
            sdf_ligand_df.loc[sdf_ligand_df['Filepath'] == sdf_file_path, 'Error'] = str(e)

    # print(sdf_ligand_df.shape)
    sdf_ligand_df.to_csv(URV_logs_folderpath + '/ligands_sdf_file.csv', index=False)  # Set index=False to avoid writing row indices
    # print(sdf_ligand_df.shape)
    condition = sdf_ligand_df['Ligand_state'] == 'valid'  # 
    sdf_ligand_valid_df = sdf_ligand_df[condition]  # Subset of DataFrame where condition is True
    sdf_ligand_invalid_df = sdf_ligand_df[~condition]  # Subset of DataFrame where condition is False
    sdf_ligand_valid_df.to_csv(URV_logs_folderpath + '/ligands_valid_sdf_file.csv', index=False)  # Set index=False to avoid writing row indices
    sdf_ligand_invalid_df.to_csv(URV_logs_folderpath + '/ligands_invalid_sdf_file.csv', index=False)  # Set index=False to avoid writing row indices
    #sdf_ligand_df.head(160)

def train_test_Generate(URV_affinity_filepath, URV_logs_folderpath, URV_output_folderpath, n_folds = 1):
    # Read the Affinity TXT file into a dataframe
    affinity_df = pd.read_csv(URV_affinity_filepath, sep=r'\s+', header=None)  # Use the appropriate delimiter if your file is not tab-separated

    # Define header
    header = ['Id', 'affinity']

    # Rename columns
    affinity_df.rename(columns=dict(enumerate(header)), inplace=True)
    affinity_df['Id'] = affinity_df['Id'].apply(lambda x: x.replace("_ligand", "").strip() if "_ligand" in x else x)

    # Display the dataframe
    # print(affinity_df)

    # Select the protein id first column
    protien_id = affinity_df.iloc[:, 0]

    # Select the affinity second column
    protien_affinity = affinity_df.iloc[:, 1]


    # read the ligands csv file
    ligands_df = pd.read_csv(URV_logs_folderpath + '/ligands_valid_sdf_file.csv')

    # read the protein csv file
    proteins_df = pd.read_csv(URV_logs_folderpath + '/protiens_file.csv')

    # Rename column 'protien' to 'Id'
    proteins_df = proteins_df.rename(columns={'protien': 'Id'})

    # Merge ligands and proteins DataFrames based on common 'Id' column
    combined_df = pd.merge(affinity_df, ligands_df, on='Id')
    combined_df = pd.merge(combined_df, proteins_df, on='Id')

    # Display the combined DataFrame
    #print(df_combined)
    combined_df.to_csv(URV_logs_folderpath + '/combined_file.csv', index=False)  # Set index=False to avoid writing row indices

    # Selecting specific columns
    selected_columns = ['SMILES', 'selected_sequence', 'affinity']  # Replace with your desired column names
    new_combined_df = combined_df[selected_columns].copy()  # Creating a new DataFrame with selected columns

    # Rename columns 
    new_combined_df = new_combined_df.rename(columns={'SMILES': 'compound_iso_smiles', 'selected_sequence': 'target_sequence'})

    if(n_folds <= 1):
        train_df = new_combined_df.copy()
        #train_df.head()
        test_df = new_combined_df.copy()
        #test_df.head()
        train_df.to_csv(URV_output_folderpath + '/urv_train.csv', index=False)  # Set index=False to avoid writing row indices
        test_df.to_csv(URV_output_folderpath + '/urv_test.csv', index=False)  # Set index=False to avoid writing row indices
    else:
        train_test_indices = list(range(1, len(new_combined_df) + 1))

        # Initialize the cross-validation object
        kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)

        i = 0
        # Perform the cross-validation
        for train_index, test_index in kf.split(train_test_indices):

            # print(f"Fold {i}:")
            # print(f"  Train: index={train_index}")
            # print(f"  Test:  index={test_index}")
            train_df = new_combined_df.iloc[train_index]
            test_df = new_combined_df.iloc[test_index]

            trainfile = URV_output_folderpath + '/urv_fold' + str(i + 1) + '_train' + '.csv'
            testfile = URV_output_folderpath + '/urv_fold' + str(i + 1) + '_test' + '.csv'
            train_df.to_csv(trainfile, index=False)  # Set index=False to avoid writing row indices
            test_df.to_csv(testfile, index=False)  # Set index=False to avoid writing row indices

            print(f"file {trainfile} generated.")
            print(f"file {testfile} generated.")
            i = i + 1
        

    

def DB_Generation(URV_datapath, URV_output_folderpath, k_folds = 1):
    URV_ligandsdf_folderpath = URV_datapath + '/Ligand_sdf'
    URV_protein_folderpath   = URV_datapath + '/Protein'
    URV_affinity_filepath    = URV_datapath + '/Affinity.txt'
    URV_logs_folderpath      = URV_datapath + '/logs'

    # generate protien(ligand)  
    ligand_Generate(URV_ligandsdf_folderpath, URV_logs_folderpath)

    Protein_Generate(URV_affinity_filepath, URV_protein_folderpath, URV_logs_folderpath)

    train_test_Generate(URV_affinity_filepath, URV_logs_folderpath, URV_output_folderpath, k_folds)
