import pandas as pd
import numpy as np

try:
    df = pd.read_csv('/data/workplace/jwx/MSF-CPMP/datasets_process/CycPeptMPDB_Peptide_PAMPA.csv', low_memory=False)
except:
    df = pd.read_csv('CycPeptMPDB_Peptide_PAMPA.csv', low_memory=False)

df['Permeability'] = pd.to_numeric(df['Permeability'], errors='coerce')
df = df.dropna(subset=['Permeability', 'SMILES'])

print(f"Num samples: {len(df)}")
print("\nPermeability Stats:")
print(df['Permeability'].describe())

print("\nTop 5 Lowest:")
print(df['Permeability'].sort_values().head())

print("\nTop 5 Highest:")
print(df['Permeability'].sort_values(ascending=False).head())

print("\nChecking for Infs:")
print(np.isinf(df['Permeability']).sum())

print("\nChecking for NaNs:")
print(np.isnan(df['Permeability']).sum())
