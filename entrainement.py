
import sqlite3
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score



# ==========================================================
# 1. CREATION DES DOSSIERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("graphique", exist_ok=True)


# ==========================================================
# 2. CHARGEMENT DES DONNEES
# ==========================================================

conn = sqlite3.connect("data/students.db")

try:
    df = pd.read_sql_query("SELECT * FROM students", conn)
finally:
    conn.close()

df = df.dropna(subset=["Dropout_Risk"]).copy()

print("Dimensions du dataset :", df.shape)
print("\nRépartition de Dropout_Risk :")
print(df["Dropout_Risk"].value_counts())


# ==========================================================
# 3. MATRICE DE CORRELATION
# ==========================================================

df_numeric = df.select_dtypes(include="number").drop(
    columns=["Student_ID"],
    errors="ignore"
)

correlation = df_numeric.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Matrice de corrélation des variables numériques")
plt.tight_layout()

plt.savefig(
    "graphique/matrice_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

mlflow.set_experiment("Projet_consolidation")


# ==========================================================
# 4. PREPARATION DES VARIABLES
# ==========================================================

X = df.drop(
    columns=["Dropout_Risk", "Student_ID"],
    errors="ignore"
)

y = df["Dropout_Risk"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Le dataset doit contenir les deux classes 0 et 1."
    )

feature_columns = X.columns.tolist()

numeric_columns = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude="number"
).columns.tolist()


# ==========================================================
# 5. PRETRAITEMENT
# ==========================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# ==========================================================
# 6. SEPARATION TRAIN / TEST
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# 7. DEFINITION DES MODELES
# ==========================================================

models = {
    "Baseline naïve": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DummyClassifier(
            strategy="most_frequent"
        ))
    ]),

    "Régression logistique": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),
   
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,          
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=42
            ))
        ])
    }


# Noms des fichiers de modèles
model_files = {
    "Baseline naïve": "baseline_naive.joblib",
    "Régression logistique": "regression_logistique.joblib",
    "Random Forest": "random_forest.joblib"
}



# ==========================================================
# 8. ENTRAINEMENT, MAE, SAUVEGARDE ET MLFLOW
# ==========================================================

results = []

for name, model in models.items():

    print(f"\nEntraînement : {name}")

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        accuracy_ml = accuracy_score(y_test, y_pred)

        print(f"MAE = {mae:.4f}")

        # Enregistrement des paramètres dans MLflow
        mlflow.log_param("modele", name)
        mlflow.log_param("random_state", 42)

        if name == "Random Forest":
            mlflow.log_param("n_estimators", 200)

        # Enregistrement des métriques dans MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("accuracy", accuracy_ml)

        results.append({
            "Modèle": name,
            "MAE": mae
        })

        # Sauvegarder le pipeline et les colonnes attendues
        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns
            },
            os.path.join("models", model_files[name])
        )

        print(
            "Modèle enregistré :",
            os.path.join("models", model_files[name])
        )

# ==========================================================
# 9. COMPARAISON DES MODELES
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE")

print("\n===== COMPARAISON DES MODELES =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    "graphique/comparaison_mae.csv",
    index=False
)


# ==========================================================
# 10. ENREGISTRER LE MODELE AU MAE LE PLUS FAIBLE
# ==========================================================

best_model_name = results_df.iloc[0]["Modèle"]

with open("models/meilleur_modele.txt", "w",
          encoding="utf-8") as file:
    file.write(best_model_name)

print("\nModèle avec le MAE le plus faible :", best_model_name)
print("Tous les modèles ont été enregistrés dans models/")
