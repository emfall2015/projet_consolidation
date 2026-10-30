# Analyse et prédiction du risque de décrochage scolaire

## Présentation du projet

Ce mini-projet de Data Analytics et d'Intelligence Artificielle s'appuie sur un jeu de données relatif au décrochage scolaire, téléchargé depuis [Kaggle](https://www.kaggle.com/datasets).

L'objectif est de développer une application interactive permettant d'explorer les données, d'analyser les facteurs associés au risque de décrochage scolaire et de prédire ce risque à l'aide de modèles de Machine Learning.

Le projet couvre plusieurs étapes du processus de valorisation des données :

- **Stockage et interrogation des données** à l'aide de SQL.
- **Modélisation dimensionnelle** des données selon une architecture en étoile (Star Schema).
- **Analyse exploratoire des données** avec Python.
- **Visualisation des données** pour identifier les tendances et les facteurs associés au décrochage scolaire.
- **Entraînement et évaluation de modèles de Machine Learning**.
- **Développement d'une application interactive** avec Streamlit pour présenter les résultats et effectuer des prédictions.

## Jeu de données

**Source :** [Kaggle – Datasets](https://www.kaggle.com/datasets)

**Lien du dataset :** À compléter avec le lien direct vers le jeu de données utilisé.

### Dictionnaire de données

| Variable | Description |
|---|---|
| `Student_ID` | Identifiant unique de l'étudiant. |
| `Age` | Âge de l'étudiant en années. |
| `Gender` | Genre de l'étudiant. |
| `Ethnicity` | Groupe ethnique de l'étudiant. |
| `Socioeconomic_Status` | Situation socio-économique de l'étudiant. |
| `Parental_Education_Level` | Niveau d'éducation des parents de l'étudiant. |
| `Disability_Status` | Indique si l'étudiant possède un handicap déclaré (0 = non, 1 = oui). |
| `GPA` | Moyenne générale académique de l'étudiant, généralement comprise entre 0 et 4. |
| `Past_Academic_Performance` | Niveau de performance académique lors des périodes précédentes. |
| `Current_Semester_Performance` | Performance académique durant le semestre actuel. |
| `Courses_Failed` | Nombre de cours auxquels l'étudiant a échoué. |
| `Credits_Completed` | Nombre de crédits universitaires validés. |
| `Study_Hours_per_Week` | Nombre d'heures consacrées aux études chaque semaine. |
| `Attendance_Rate` | Taux de présence aux cours, exprimé en pourcentage. |
| `Number_of_Late_Submissions` | Nombre de travaux ou devoirs remis en retard. |
| `Class_Participation_Score` | Score représentant le niveau de participation en classe. |
| `Online_Learning_Hours` | Nombre d'heures consacrées à l'apprentissage en ligne. |
| `Library_Usage_Hours` | Nombre d'heures consacrées à l'utilisation de la bibliothèque. |
| `Disciplinary_Actions` | Nombre d'actions ou de sanctions disciplinaires enregistrées. |
| `Social_Engagement_Score` | Score représentant le niveau d'engagement social de l'étudiant. |
| `Mental_Health_Score` | Indicateur associé à la santé et au bien-être mental de l'étudiant. |
| `Extracurricular_Activities` | Nombre d'activités extrascolaires auxquelles l'étudiant participe. |
| `Dropout_Risk` | Variable cible indiquant le risque de décrochage scolaire (0 = aucun risque indiqué, 1 = risque indiqué). |

### Variable cible et variables explicatives

- **Variable cible (Target) :** `Dropout_Risk`
- **Variables explicatives (Features) :** toutes les autres variables, à l'exception de `Student_ID`, qui sert uniquement d'identifiant.

L'objectif du modèle est de prédire le risque de décrochage scolaire à partir des caractéristiques académiques, personnelles et socio-économiques des étudiants.

## Stack technique

| Technologie | Utilisation |
|---|---|
| **Python** | Langage principal du projet |
| **SQL** | Stockage et interrogation des données |
| **Pandas** | Manipulation et préparation des données |
| **NumPy** | Calcul numérique |
| **Scikit-learn** | Préparation des données et Machine Learning |
| **Matplotlib** | Création de visualisations |
| **Seaborn** | Analyse exploratoire et visualisation statistique |
| **Streamlit** | Développement de l'application interactive |

## Fonctionnalités de l'application

### 1.  Tableau de bord

Présentation d'une vue globale du jeu de données à travers plusieurs indicateurs et visualisations :

- Répartition des étudiants selon leur risque de décrochage.
- Comparaison de la moyenne générale académique (GPA) selon le risque de décrochage.
- Analyse du taux de risque de décrochage selon le statut socio-économique.

### 2. Analyse du risque de décrochage

Étude des principaux indicateurs académiques et comportementaux :

- Taux de présence moyen des étudiants.
- Nombre moyen de cours échoués.
- Nombre de travaux remis en retard.

### 3. Profil des étudiants

Exploration des caractéristiques démographiques et socio-économiques des étudiants :

- Distribution des étudiants par âge.
- Analyse du risque de décrochage selon les caractéristiques démographiques.
- Analyse du risque de décrochage selon le niveau d'éducation des parents (High School, Bachelor, Master).

### 4. Évaluation des modèles de Machine Learning

Entraînement et comparaison de trois modèles de classification :

- **Baseline naïve :** modèle de référence permettant d'établir une première performance.
- **Régression logistique (Logistic Regression) :** modèle de classification statistique.
- **Random Forest :** modèle d'ensemble basé sur plusieurs arbres de décision.

Les performances des modèles sont comparées afin de sélectionner le modèle le plus performant.

**Métrique d'évaluation :** MAE (Mean Absolute Error).

### 5. Prédiction du risque de décrochage

L'application permet de renseigner les caractéristiques d'un étudiant afin de prédire son risque de décrochage scolaire.

Exemple de données utilisées pour une prédiction :

```python
student = {
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
```

À partir de ces informations, le modèle sélectionné estime le risque de décrochage scolaire de l'étudiant.

## Installation et configuration

### 1. Cloner le dépôt GitHub

```bash
git clone https://github.com/emfall2015/projet_consolidation.git

cd projet_consolidation
```

### 2. Créer un environnement virtuel

**Linux / macOS**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv

.venv\\Scripts\\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Lancement de l'application

Depuis la racine du projet, exécuter la commande suivante :

```bash
streamlit run app.py
```

L'application sera ensuite accessible dans le navigateur à l'adresse locale indiquée par Streamlit (généralement `http://localhost:8501`).

## Objectif final

Ce projet permet de mettre en pratique différentes compétences en Data Engineering, Data Analytics et Machine Learning, depuis le stockage et la modélisation des données jusqu'à leur analyse, leur visualisation et leur exploitation dans une application interactive.

Il vise également à illustrer comment les données peuvent contribuer à mieux comprendre les facteurs associés au décrochage scolaire et à identifier les étudiants présentant un risque potentiel.