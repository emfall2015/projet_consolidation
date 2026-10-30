/* Etudiant ayant un GPA(moyenne générale) faible et une mauvaise performance */
SELECT
    Student_ID,
    Age,
    GPA,
    Past_Academic_Performance,
    Current_Semester_Performance,
    Courses_Failed,
    Dropout_Risk
FROM students
WHERE GPA < 2.5
  AND Courses_Failed >= 2
ORDER BY GPA ASC;

/* Taux de décrochage selon le statut socio-economic */
SELECT
    Socioeconomic_Status,
    COUNT(*) AS Nombre_Etudiants,
    SUM(Dropout_Risk) AS Nombre_Decrochages,
    ROUND(AVG(Dropout_Risk) * 100, 2) AS Taux_Decrochage
FROM students
GROUP BY Socioeconomic_Status
ORDER BY Taux_Decrochage DESC;

/* l' assiduité moyenne des étudiants selon leur risque de décrochage.*/
SELECT
    Dropout_Risk,
    COUNT(*) AS Nombre_Etudiants,
    ROUND(AVG(Attendance_Rate), 2) AS Assiduite_Moyenne,
    ROUND(AVG(GPA), 2) AS GPA_Moyen,
    ROUND(AVG(Courses_Failed), 2) AS Cours_Echoues_Moyens
FROM students
GROUP BY Dropout_Risk;

/* Etudiant cumulant plusieurs facteurs de risque */

SELECT
    Student_ID,
    GPA,
    Attendance_Rate,
    Courses_Failed,
    Number_of_Late_Submissions,
    Mental_Health_Score,
    Disciplinary_Actions,
    Dropout_Risk
FROM students
WHERE GPA < 2.5
  AND Attendance_Rate < 70
  AND Courses_Failed >= 2
ORDER BY GPA ASC, Attendance_Rate ASC;