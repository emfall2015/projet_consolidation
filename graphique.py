import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect("data/students.db")

# Graphique : Taux de décrochage selon le statut socio-économique

df = pd.read_sql_query("""
    SELECT
        Socioeconomic_Status,
        COUNT(*) AS Nombre_Etudiants,
        SUM(Dropout_Risk) AS Nombre_Decrochages,
        ROUND(AVG(Dropout_Risk) * 100, 2) AS Taux_Decrochage
    FROM students
    GROUP BY Socioeconomic_Status
    ORDER BY Taux_Decrochage DESC
    """, conn)

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Socioeconomic_Status",
    y="Taux_Decrochage"
)

plt.title("Taux de décrochage selon le statut socio-économique")
plt.xlabel("Statut socio-économique")
plt.ylabel("Taux de décrochage (%)")

plt.tight_layout()

plt.savefig(
    "graphique/decrochage_socio_eco.png",
    dpi=300,
    bbox_inches="tight"
)

# Taux de décrochage selon le niveau d'éducation des parents
df = pd.read_sql_query("""
    SELECT
        Parental_Education_Level,
        ROUND(AVG(Dropout_Risk) * 100, 2) AS Taux_Decrochage
    FROM students
    GROUP BY Parental_Education_Level
    ORDER BY Taux_Decrochage DESC
""", conn)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Parental_Education_Level",
    y="Taux_Decrochage"
)

plt.title("Taux de décrochage selon le niveau d'éducation des parents")
plt.xlabel("Niveau d'éducation des parents")
plt.ylabel("Taux de décrochage (%)")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("graphique/decrochage_education_parents.png", dpi=300)



conn.close()



