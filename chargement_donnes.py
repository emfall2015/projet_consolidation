import sqlite3
import pandas as pd

# Connexion à la base SQLite
conn = sqlite3.connect("data/students.db")

csv_path_propre = r"data\student_management_dataset_propre.csv"

df = pd.read_csv(csv_path_propre)

# Ajouter le DataFrame dans une table SQLite
df.to_sql(
    "students",
    conn,
    if_exists="replace",   # remplace la table si elle existe déjà
    index=False
)

conn.close()

print("Données enregistrées dans data/students.db")