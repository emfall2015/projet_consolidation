""" 
Analyse des performances académiques et du risque d'abandon scolaire 
dans l'enseignement supérieur grace à l'IA
"""

import pandas as pd


csv_path = r"data\student_management_dataset.csv"
csv_path_propre = r"data\student_management_dataset_propre.csv"

df = pd.read_csv(csv_path)

print("Nombre de lignes")
print(len(df))

# Découverte du jeu de données
print("Dimensions :", df.shape)
print("\nAperçu :")
print(df.head())

print("\nInformations :")
print(df.info())

print("\nStatistiques :")
print(df.describe())

print("\nValeurs manquantes :")
print(df.isna().sum())

df.to_csv(csv_path_propre, index=False, encoding="utf-8-sig")

