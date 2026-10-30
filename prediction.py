
import os
import joblib
import pandas as pd


# ==========================================================
# 1. CHARGER LE MODELE ENREGISTRE
# ==========================================================

model_files = {
    "Baseline naïve": "baseline_naive.joblib",
    "Régression logistique": "regression_logistique.joblib",
    "Random Forest": "random_forest.joblib"
}

with open(
    "models/meilleur_modele.txt",
    "r",
    encoding="utf-8"
) as file:
    best_model_name = file.read().strip()

model_path = os.path.join(
    "models",
    model_files[best_model_name]
)

saved_data = joblib.load(model_path)

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

print("Modèle chargé :", best_model_name)


# ==========================================================
# 2. CARACTERISTIQUES DU NOUVEL ETUDIANT
# ==========================================================

new_student = {
    "Age": 22,
    "Gender": "Female",
    "Ethnicity": "Asian",
    "Socioeconomic_Status": "Medium",
    "Parental_Education_Level": "Bachelor’s",
    "Disability_Status": 0,
    "GPA": 2.8,
    "Past_Academic_Performance": "Average",
    "Current_Semester_Performance": 75,
    "Courses_Failed": 1,
    "Credits_Completed": 45,
    "Study_Hours_per_Week": 15,
    "Attendance_Rate": 80.0,
    "Number_of_Late_Submissions": 2,
    "Class_Participation_Score": 6.0,
    "Online_Learning_Hours": 8.0,
    "Library_Usage_Hours": 5.0,
    "Disciplinary_Actions": 0,
    "Social_Engagement_Score": 6.0,
    "Mental_Health_Score": 7.0,
    "Extracurricular_Activities": 2
}


# ==========================================================
# 3. VERIFIER LES VARIABLES
# ==========================================================

missing_columns = [
    column for column in feature_columns
    if column not in new_student
]

if missing_columns:
    raise ValueError(
        f"Variables manquantes : {missing_columns}"
    )

# Respecter exactement les colonnes et leur ordre
X_new = pd.DataFrame(
    [new_student],
    columns=feature_columns
)


# ==========================================================
# 4. PREDICTION
# ==========================================================

prediction = int(model.predict(X_new)[0])

print("\n===== RESULTAT DE LA PREDICTION =====")

if prediction == 1:
    print("Risque d'abandon détecté.")
else:
    print("Aucun risque d'abandon détecté par le modèle.")


# ==========================================================
# 5. PROBABILITE ESTIMEE
# ==========================================================

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(X_new)[0]

    class_labels = model.named_steps["model"].classes_

    if 1 in class_labels:
        probability = probabilities[
            list(class_labels).index(1)
        ]

        print(
            f"Probabilité estimée de la classe 1 : "
            f"{probability:.2%}"
        )
