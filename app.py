import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Analyse du décrochage étudiant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #172033;
    font-weight: 700;
}

h2 {
    color: #26344d;
}

h3 {
    color: #344563;
}

[data-testid="stMetric"] {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
    border: 1px solid #edf0f5;
}

[data-testid="stSidebar"] {
    background-color: #172033;
}

[data-testid="stSidebar"] * {
    color: white;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CONNEXION SQLITE
# =========================================================

@st.cache_data
def load_data():

    conn = sqlite3.connect("data/students.db")

    data = pd.read_sql_query(
        "SELECT * FROM students",
        conn
    )

    conn.close()

    return data


df = load_data()


# =========================================================
# TITRE
# =========================================================

st.markdown(
    """
    ### Analyse et suivi du risque de décrochage étudiant
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "",
    [
        "Tableau de bord",
        "Analyse du risque",
        "Performance académique",
        "Profil des étudiants",
        "Corrélations",
        "Recherche étudiant",
        "Predictions",
        "Evaluation"
    ]
)




# # =========================================================
# # FILTRES GLOBAUX
# # =========================================================

# with st.sidebar.expander("Filtres"):

#     if "Socioeconomic_Status" in df.columns:

#         socio_options = df["Socioeconomic_Status"].dropna().unique()

#         selected_socio = st.multiselect(
#             "Statut socio-économique",
#             socio_options,
#             default=list(socio_options)
#         )

#     else:
#         selected_socio = []

#     if "Gender" in df.columns:

#         gender_options = df["Gender"].dropna().unique()

#         selected_gender = st.multiselect(
#             "Genre",
#             gender_options,
#             default=list(gender_options)
#         )

#     else:
#         selected_gender = []


# df = df.copy()

# if selected_socio:
#     df = df[
#         df["Socioeconomic_Status"].isin(selected_socio)
#     ]

# if selected_gender:
#     df = df[
#         df["Gender"].isin(selected_gender)
#     ]


# =========================================================
# PAGE : TABLEAU DE BORD
# =========================================================

if page == "Tableau de bord":

    st.subheader("Vue générale")

    total_students = len(df)

    # Taux de risque 
    nb_risque = (df["Dropout_Risk"] == 1).sum()
    nb_sans_risque = (df["Dropout_Risk"] == 0).sum()
    
    dropout_rate = (
    nb_risque / len(df) * 100
    if len(df) > 0
    else 0
    )


    avg_attendance = (
        df["Attendance_Rate"].mean()
        if len(df) > 0
        else 0
    )

    avg_failed = (
        df["Courses_Failed"].mean()
        if len(df) > 0
        else 0
    )


    # KPI

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Étudiants",
        f"{total_students:,}"
    )

    col2.metric(
        "Risque de décrochage",
        f"{dropout_rate:.1f}%"
    )


    col3.metric(
        "Taux de Présence au cours",
        f"{avg_attendance:.1f}%"
    )

    col4.metric(
        "Cours échoués en moyenne",
        f"{avg_failed:.2f}"
    )


    st.markdown("---")


    # =====================================================
    # GRAPHIQUE RISQUE
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        risk_df = (
            df["Dropout_Risk"]
            .value_counts()
            .reset_index()
        )

        risk_df.columns = [
            "Dropout_Risk",
            "Nombre"
        ]

        risk_df["Statut"] = risk_df["Dropout_Risk"].map({
            0: "Pas de risque",
            1: "Risque"
        })

        fig = px.pie(
            risk_df,
            names="Statut",
            values="Nombre",
            title="Répartition du risque de décrochage",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # GPA
    # =====================================================

    with col2:

        gpa_df = (
            df
            .groupby("Dropout_Risk")["GPA"]
            .mean()
            .reset_index()
        )

        gpa_df["Statut"] = gpa_df["Dropout_Risk"].map({
            0: "Pas de risque",
            1: "Risque"
        })

        fig = px.bar(
            gpa_df,
            x="Statut",
            y="GPA",
            title="Moyenne générale académique(GPA) moyen selon le risque",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # SOCIO ECONOMIQUE
    # =====================================================

    socio_df = (
        df
        .groupby("Socioeconomic_Status")["Dropout_Risk"]
        .mean()
        .reset_index()
    )

    socio_df["Dropout_Risk"] *= 100

    fig = px.bar(
        socio_df,
        x="Socioeconomic_Status",
        y="Dropout_Risk",
        title="Taux de décrochage selon le statut socio-économique",
        labels={
            "Dropout_Risk": "Taux de décrochage (%)",
            "Socioeconomic_Status": "Statut socio-économique"
        },
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE : ANALYSE DU RISQUE
# =========================================================

elif page == "Analyse du risque":

    # st.subheader("Analyse du risque de décrochage")

    # GPA

    # fig = px.box(
    #     df,
    #     x="Dropout_Risk",
    #     y="GPA",
    #     color="Dropout_Risk",
    #     points="all",
    #     title="GPA selon le risque de décrochage"
    # )

    # fig.update_xaxes(
    #     tickvals=[0, 1],
    #     ticktext=["Pas de risque", "Risque"]
    # )

    # st.plotly_chart(
    #     fig,
    #     use_container_width=True
    # )


    # Attendance

    col1, col2 = st.columns(2)

    with col1:

        attendance_df = (
            df
            .groupby("Dropout_Risk")["Attendance_Rate"]
            .mean()
            .reset_index()
        )

        attendance_df["Statut"] = attendance_df[
            "Dropout_Risk"
        ].map({
            0: "Pas de risque",
            1: "Risque"
        })

        fig = px.bar(
            attendance_df,
            x="Statut",
            y="Attendance_Rate",
            title="Présence moyenne",
            text_auto=".1f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        failed_df = (
            df
            .groupby("Dropout_Risk")["Courses_Failed"]
            .mean()
            .reset_index()
        )

        failed_df["Statut"] = failed_df[
            "Dropout_Risk"
        ].map({
            0: "Pas de risque",
            1: "Risque"
        })

        fig = px.bar(
            failed_df,
            x="Statut",
            y="Courses_Failed",
            title="Nombre moyen de cours échoués",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Late submissions

    late_df = (
        df
        .groupby("Dropout_Risk")[
            "Number_of_Late_Submissions"
        ]
        .mean()
        .reset_index()
    )

    late_df["Statut"] = late_df[
        "Dropout_Risk"
    ].map({
        0: "Pas de risque",
        1: "Risque"
    })

    fig = px.bar(
        late_df,
        x="Statut",
        y="Number_of_Late_Submissions",
        title="Travaux remis en retard",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE : PERFORMANCE ACADEMIQUE
# =========================================================

elif page == "Performance académique":

    # st.subheader("Performance académique")

    # col1, col2 = st.columns(2)

    # with col1:

    #     fig = px.scatter(
    #         df,
    #         x="Study_Hours_per_Week",
    #         y="GPA",
    #         color="Dropout_Risk",
    #         hover_data=[
    #             "Student_ID",
    #             "Attendance_Rate",
    #             "Courses_Failed"
    #         ],
    #         title="Heures d'étude vs GPA"
    #     )

    #     st.plotly_chart(
    #         fig,
    #         use_container_width=True
    #     )


    # with col2:

    #     fig = px.scatter(
    #         df,
    #         x="Attendance_Rate",
    #         y="GPA",
    #         color="Dropout_Risk",
    #         hover_data=[
    #             "Student_ID",
    #             "Courses_Failed"
    #         ],
    #         title="Présence vs GPA"
    #     )

    #     st.plotly_chart(
    #         fig,
    #         use_container_width=True
    #     )


    # Performance passée
    df["Past_Academic_Performance"] = df["Past_Academic_Performance"].replace({
    "Above Average": "Au-dessus de la moyenne",
    "Average": "Moyenne",
    "Below Average": "En dessous de la moyenne"
})
    past_df = (
        df
        .groupby("Past_Academic_Performance")[
            "Dropout_Risk"
        ]
        .mean()
        .reset_index()
    )

    past_df["Dropout_Risk"] *= 100

    fig = px.bar(
        past_df,
        x="Past_Academic_Performance",
        y="Dropout_Risk",
        title="Taux de décrochage selon la performance académique passée",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Risque de décrochage selon la performance académique passée")
    conn = sqlite3.connect("data/students.db")

    # Requête SQL
    query = """
    SELECT
        Past_Academic_Performance,
        COUNT(*) AS Nombre_Etudiants,
        SUM(Dropout_Risk) AS Nombre_A_Risque,
        ROUND(AVG(Dropout_Risk) * 100, 2) AS Taux_Decrochage
    FROM students
    GROUP BY Past_Academic_Performance
    """

    df_performance = pd.read_sql_query(query, conn)

    # Traduction des catégories
    df_performance["Past_Academic_Performance"] = df_performance[
        "Past_Academic_Performance"
    ].replace({
        "Above Average": "Au-dessus de la moyenne",
        "Average": "Moyenne",
        "Below Average": "En dessous de la moyenne"
    })

    # Ligne Total
    total_etudiants = df_performance["Nombre_Etudiants"].sum()
    total_risque = df_performance["Nombre_A_Risque"].sum()
    taux_total = total_risque / total_etudiants * 100

    ligne_total = pd.DataFrame({
        "Past_Academic_Performance": ["Total"],
        "Nombre_Etudiants": [total_etudiants],
        "Nombre_A_Risque": [total_risque],
        "Taux_Decrochage": [round(taux_total, 2)]
    })

    df_performance = pd.concat(
        [df_performance, ligne_total],
        ignore_index=True
    )

    # Renommer les colonnes pour l'affichage
    df_performance = df_performance.rename(columns={
        "Past_Academic_Performance": "Performance passée",
        "Nombre_Etudiants": "Total étudiants",
        "Nombre_A_Risque": "À risque",
        "Taux_Decrochage": "Taux de risque (%)"
    })

    # Affichage
    st.dataframe(
        df_performance,
        use_container_width=True,
        hide_index=True
    )
    conn.close()

# =========================================================
# PAGE : PROFIL
# =========================================================

elif page == "Profil des étudiants":

    st.subheader("Profil des étudiants")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Age",
            color="Dropout_Risk",
            nbins=10,
            barmode="group",
            title="Distribution des étudiants par âge",
            labels={
                "Age": "Âge",
                "Dropout_Risk": "Risque de décrochage"
            }
        )

        fig.update_traces(
            opacity=0.8
        )

        fig.update_layout(
            xaxis=dict(
                dtick=1
            ),
            yaxis_title="Nombre d'étudiants",
            legend_title="Risque de décrochage",
            bargap=0.15
        )

        fig.for_each_trace(
            lambda trace: trace.update(
                name="À risque" if trace.name == "1" else "Sans risque"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        gender_df = (
            df
            .groupby("Gender")["Dropout_Risk"]
            .mean()
            .reset_index()
        )

        gender_df["Dropout_Risk"] *= 100

        fig = px.bar(
            gender_df,
            x="Gender",
            y="Dropout_Risk",
            title="Taux de décrochage selon le genre",
            text_auto=".1f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Education parents

    parent_df = (
        df
        .groupby("Parental_Education_Level")[
            "Dropout_Risk"
        ]
        .mean()
        .reset_index()
    )

    parent_df["Dropout_Risk"] *= 100

    fig = px.bar(
        parent_df,
        x="Parental_Education_Level",
        y="Dropout_Risk",
        title="Taux de décrochage selon le niveau d'éducation des parents",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE : CORRELATIONS
# =========================================================

elif page == "Corrélations":

    st.subheader("Matrice de corrélation")

    numeric_df = (
        df
        .select_dtypes(include=["int64", "float64"])
        .drop(
            columns=["Student_ID"],
            errors="ignore"
        )
    )

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title(
        "Matrice de corrélation des variables numériques"
    )

    st.pyplot(fig)


    # st.markdown("---")

    # st.subheader(
    #     "Corrélations avec le risque de décrochage"
    # )

    # dropout_corr = (
    #     correlation["Dropout_Risk"]
    #     .drop("Dropout_Risk")
    #     .sort_values()
    # )

    # fig = px.bar(
    #     dropout_corr,
    #     orientation="h",
    #     title="Corrélation des variables avec Dropout_Risk",
    #     labels={
    #         "value": "Corrélation",
    #         "index": "Variable"
    #     }
    # )

    # st.plotly_chart(
    #     fig,
    #     use_container_width=True
    # )


# =========================================================
# PAGE : RECHERCHE ETUDIANT
# =========================================================

elif page == "Recherche étudiant":

    st.subheader("Recherche d'un étudiant")

    student_id = st.number_input(
        "Entrez le Student_ID",
        min_value=int(df["Student_ID"].min()),
        max_value=int(df["Student_ID"].max()),
        step=1
    )

    student = df[
        df["Student_ID"] == student_id
    ]

    if len(student) == 0:

        st.warning(
            "Aucun étudiant trouvé."
        )

    else:

        student_data = student.iloc[0]

        risk = int(
            student_data["Dropout_Risk"]
        )

        if risk == 1:

            st.error(
                "Cet étudiant est indiqué comme présentant un risque."
            )

        else:

            st.success(
                "Cet étudiant n'est pas indiqué comme présentant un risque."
            )


        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Moyenne générale académique",
            f"{student_data['GPA']:.2f}"
        )

        col2.metric(
            "Présence",
            f"{student_data['Attendance_Rate']:.1f}%"
        )

        col3.metric(
            "Cours échoués",
            f"{student_data['Courses_Failed']}"
        )

        col4.metric(
            "Heures d'étude",
            f"{student_data['Study_Hours_per_Week']}"
        )


        st.markdown("---")

        st.subheader("Informations détaillées")

        details = student.T

        details.columns = ["Valeur"]

        st.dataframe(
            details,
            use_container_width=True
        )


# =========================================================
# PAGE : Prediction
# =========================================================

elif page == "Predictions":

    st.subheader("Prédire le risque d'abandon d'un étudiant")

    DB_PATH = "data/students.db"
    MODELS_DIR = "models"

    # -----------------------------------------------------
    # 1. Charger les modèles disponibles
    # -----------------------------------------------------

    model_files = {
        "Régression logistique": "regression_logistique.joblib",
        "Random Forest": "random_forest.joblib",
        "Baseline naïve": "baseline_naive.joblib",
    }

    available_models = {
        name: os.path.join(MODELS_DIR, filename)
        for name, filename in model_files.items()
        if os.path.exists(os.path.join(MODELS_DIR, filename))
    }

    if not available_models:
        st.error(
            "Aucun modèle trouvé. Lance d'abord entrainement.py."
        )
        st.stop()

    selected_model_name = st.selectbox(
        "Choisir le modèle de prédiction",
        list(available_models.keys()),
        index=0
    )

    # -----------------------------------------------------
    # 2. Charger le modèle et ses colonnes
    # -----------------------------------------------------

    saved_data = joblib.load(
        available_models[selected_model_name]
    )

    model = saved_data["model"]
    feature_columns = saved_data["feature_columns"]

    # -----------------------------------------------------
    # 3. Lire les données existantes
    # -----------------------------------------------------

    with sqlite3.connect(DB_PATH) as conn:
        df_students = pd.read_sql_query(
            "SELECT * FROM students",
            conn
        )

    # Variables catégorielles connues
    categorical_features = [
        "Gender",
        "Ethnicity",
        "Socioeconomic_Status",
        "Parental_Education_Level",
        "Disability_Status",
        "Extracurricular_Activities",
    ]

    st.markdown(
        "Renseigne les informations de l'étudiant, "
        "puis clique sur **Prédire et enregistrer**."
    )

    # -----------------------------------------------------
    # 4. Formulaire étudiant
    # -----------------------------------------------------
    
    # Traduction des noms des colonnes pour l'interface
    labels_fr = {
        "Age": "Âge",
        "Gender": "Genre",
        "Ethnicity": "Origine ethnique",
        "Socioeconomic_Status": "Statut socioéconomique",
        "Parental_Education_Level": "Niveau d'études des parents",
        "Disability_Status": "Situation de handicap",
        "GPA": "Moyenne générale (GPA)",
        "Past_Academic_Performance": "Résultats scolaires antérieurs",
        "Current_Semester_Performance": "Résultats du semestre actuel",
        "Courses_Failed": "Nombre de cours échoués",
        "Credits_Completed": "Nombre de crédits validés",
        "Study_Hours_per_Week": "Heures d'étude par semaine",
        "Attendance_Rate": "Taux de présence (%)",
        "Number_of_Late_Submissions": "Nombre de devoirs rendus en retard",
        "Class_Participation_Score": "Score de participation en classe",
        "Online_Learning_Hours": "Heures d'apprentissage en ligne",
        "Library_Usage_Hours": "Heures d'utilisation de la bibliothèque",
        "Disciplinary_Actions": "Nombre de sanctions disciplinaires",
        "Social_Engagement_Score": "Score d'engagement social",
        "Mental_Health_Score": "Score de santé mentale",
        "Extracurricular_Activities": "Activités extrascolaires",
    }

    with st.form("form_nouvel_etudiant"):

        st.markdown("### Informations générales")

        form_data = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_columns):

            container = col1 if i % 2 == 0 else col2

            with container:

                if (
                    feature in categorical_features
                    or df_students[feature].dtype == "object"
                ):

                    # Valeurs existantes dans la base
                    options = (
                        df_students[feature]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    options = sorted(options)

                    if not options:
                        options = ["Unknown"]

                    form_data[feature] = st.selectbox(
                    labels_fr.get(feature, feature),
                     options,
                    key=f"input_{feature}"
                    )

                else:

                    # Valeur médiane utilisée comme valeur initiale
                    values = pd.to_numeric(
                        df_students[feature],
                        errors="coerce"
                    ).dropna()

                    if len(values) > 0:
                        default_value = float(values.median())
                        min_value = float(values.min())
                        max_value = float(values.max())
                    else:
                        default_value = 0.0
                        min_value = 0.0
                        max_value = 100.0

                    # Élargir légèrement la plage si nécessaire
                    if min_value == max_value:
                        min_value -= 1
                        max_value += 1

                    form_data[feature] = st.number_input(
                    labels_fr.get(feature, feature),
                    min_value=min_value,
                    max_value=max_value,
                    value=min(
                    max(default_value, min_value),
                    max_value
                    ),
                    key=f"input_{feature}"
                    )

        submitted = st.form_submit_button(
            "Prédire et enregistrer",
            type="primary",
            use_container_width=True
        )

    # -----------------------------------------------------
    # 5. Prédiction et enregistrement
    # -----------------------------------------------------

    if submitted:

        # Les colonnes sont conservées dans l'ordre attendu
        X_new = pd.DataFrame(
            [form_data],
            columns=feature_columns
        )

        try:
            prediction = int(model.predict(X_new)[0])

            # Probabilité associée à la classe 1, si disponible
            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_new)[0]
                classes = model.named_steps["model"].classes_

                if 1 in classes:
                    index_class_1 = list(classes).index(1)
                    probability = float(
                        probabilities[index_class_1]
                    )

            # Créer une table séparée pour ne pas mélanger
            # les prédictions avec les vraies étiquettes
            with sqlite3.connect(DB_PATH) as conn:

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS students_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_data TEXT NOT NULL,
                        predicted_dropout_risk INTEGER NOT NULL,
                        dropout_probability REAL,
                        model_name TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute(
                    """
                    INSERT INTO students_predictions (
                        student_data,
                        predicted_dropout_risk,
                        dropout_probability,
                        model_name,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        X_new.to_json(orient="records"),
                        prediction,
                        probability,
                        selected_model_name,
                        datetime.now().isoformat(timespec="seconds")
                    )
                )

            # Afficher le résultat
            st.divider()
            st.markdown("### Résultat de la prédiction")

            if prediction == 1:
                st.error(
                    "Risque d'abandon détecté par le modèle."
                )
            else:
                st.success(
                    "Le modèle ne détecte pas de risque "
                    "d'abandon pour cet étudiant."
                )

            if probability is not None:
                st.metric(
                    "Probabilité estimée d'abandon",
                    f"{probability:.1%}"
                )

                st.progress(
                    max(0.0, min(1.0, probability))
                )

            st.caption(
                f"Modèle utilisé : {selected_model_name}"
            )

            with st.expander("Voir les informations saisies"):
                st.dataframe(
                    X_new,
                    use_container_width=True
                )

        except Exception as e:
            st.error(
                f"Erreur pendant la prédiction : {e}"
            )

# =========================================================
# PAGE : Évaluation du modèle
# =========================================================

elif page == "Evaluation":

    st.subheader("Comparer les prédictions aux vraies valeurs")

    model_files = {
        "Régression logistique": "models/regression_logistique.joblib",
        "Random Forest": "models/random_forest.joblib",
        "Baseline naïve": "models/baseline_naive.joblib",
    }

    model_name = st.selectbox(
        "Choisir le modèle à évaluer",
        list(model_files.keys())
    )

    if st.button("Lancer l'évaluation", type="primary"):

        # Charger les données
        with sqlite3.connect("data/students.db") as conn:
            df = pd.read_sql_query(
                "SELECT * FROM students",
                conn
            )

        df = df.dropna(subset=["Dropout_Risk"])

        # Préparer X et y
        saved_data = joblib.load(model_files[model_name])
        model = saved_data["model"]
        feature_columns = saved_data["feature_columns"]

        X = df[feature_columns]
        y = df["Dropout_Risk"].astype(int)

        # Reproduire le split utilisé lors de l'entraînement
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Prédictions sur le jeu de test
        y_pred = model.predict(X_test)

        # Métriques
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Exactitude (accuracy)",
            f"{accuracy_score(y_test, y_pred):.1%}"
        )

        col2.metric(
            "Précision - classe abandon",
            f"{precision_score(y_test, y_pred, zero_division=0):.1%}"
        )

        col3.metric(
            "Rappel - classe abandon",
            f"{recall_score(y_test, y_pred, zero_division=0):.1%}"
        )

        col4.metric(
            "F1-score - classe abandon",
            f"{f1_score(y_test, y_pred, zero_division=0):.1%}"
        )

        st.markdown("### Matrice de confusion")

        # Lignes = vraies valeurs, colonnes = prédictions
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        cm_df = pd.DataFrame(
            cm,
            index=["Réel : pas d'abandon", "Réel : abandon"],
            columns=["Prédit : pas d'abandon", "Prédit : abandon"]
        )

        st.dataframe(cm_df, use_container_width=True)

        st.markdown("### Comparaison étudiant par étudiant")

        comparaison = X_test.copy()
        comparaison["Valeur réelle"] = y_test
        comparaison["Valeur prédite"] = y_pred
        comparaison["Résultat"] = (
            comparaison["Valeur réelle"]
            == comparaison["Valeur prédite"]
        ).map({
            True: "Correct",
            False: "Incorrect"
        })

        comparaison["Valeur réelle"] = comparaison[
            "Valeur réelle"
        ].map({
            0: "Pas d'abandon",
            1: "Abandon"
        })

        comparaison["Valeur prédite"] = comparaison[
            "Valeur prédite"
        ].map({
            0: "Pas d'abandon",
            1: "Abandon"
        })

        st.dataframe(
            comparaison,
            use_container_width=True
        )

        # Filtrer les erreurs
        erreurs = comparaison[
            comparaison["Résultat"] == "Incorrect"
        ]

        st.markdown("### Prédictions incorrectes")

        if erreurs.empty:
            st.success("Aucune erreur sur ce jeu de test.")
        else:
            st.dataframe(erreurs, use_container_width=True)

        st.markdown("### Rapport de classification")

        rapport = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Pas d'abandon", "Abandon"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(rapport).transpose(),
            use_container_width=True
        )
