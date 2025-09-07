# Prédiction du Churn Client - Projet de Machine Learning
Ce projet analyse les données de clients d'une plateforme e-commerce pour prédire la probabilité de résiliation (churn). L'objectif est de comprendre les facteurs clés qui mènent à la perte de clients et de construire un modèle prédictif déployé via une application web interactive.

Présentation: [https://docs.google.com/presentation/d/1RMP6nOubCtmiaKo5-O1d-b62YopHXdX40PQx8i74r8E/edit?usp=sharing]

## 🎯 Objectif du Projet
L'objectif principal était de répondre à la question suivante :

Quelles caractéristiques et quels comportements clients contribuent le plus à l'attrition, et comment pouvons-nous utiliser ces informations pour la réduire ?

## 📊 Dataset
Le jeu de données utilisé provient de Kaggle : E-Commerce Customer Churn Analysis and Prediction(https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction).

Il contient des informations variées sur les clients, telles que :

Données démographiques : Sexe, statut marital, ville.

Comportement sur la plateforme : Ancienneté (Tenure), appareil de connexion préféré, heures passées sur l'application.

Historique d'achat : Catégorie de produits préférée, nombre de commandes, jours depuis la dernière commande.

Satisfaction et Service Client : Score de satisfaction, réclamations.

# 🛠️ Méthodologie
Le projet a été structuré en plusieurs étapes clés :

## Analyse Exploratoire des Données (EDA) : 
Visualisation des distributions, identification des corrélations et des valeurs manquantes.

## Nettoyage et Prétraitement : 
Standardisation des catégories et gestion des valeurs manquantes.

## Feature Engineering : 
Création de segments clients pertinents en utilisant une approche de clustering RFM (Récence, Fréquence, Monétaire).

## Modélisation : 
Entraînement et comparaison de trois modèles de classification :

Régression Logistique

Arbre de Décision

Random Forest (meilleur modèle retenu)

## Évaluation : 
Le Random Forest a été sélectionné pour ses excellentes performances, notamment sur la métrique PR-AUC (Precision-Recall Area Under Curve), qui est très pertinente pour les datasets déséquilibrés.

## Interprétabilité : 
Utilisation de SHAP (SHapley Additive exPlanations) pour comprendre l'influence de chaque caractéristique sur les prédictions du modèle. Les facteurs les plus importants se sont avérés être l'ancienneté (Tenure), le score de satisfaction, et le fait d'avoir déposé une réclamation.

# 🚀 Application Web Streamlit
Une application web a été développée avec Streamlit pour permettre une interaction simple et intuitive avec le modèle de prédiction.

Lien vers l'application : Application de Prédiction de Churn [https://acb-churn-dudagrandprojetml.streamlit.app/]

L'application permet de :

Saisir les informations d'un client via un formulaire.

Obtenir une prédiction en temps réel sur son risque de churn.

Visualiser la probabilité de churn associée.

# ⚙️ Comment lancer le projet localement
Clonez le dépôt :

git clone [https://github.com/alexandre-cameron-borges/duda_grandprojet_ML.git](https://github.com/alexandre-cameron-borges/duda_grandprojet_ML.git)
cd duda_grandprojet_ML

Installez les dépendances :

pip install -r requirements.txt

Lancez l'application Streamlit :

streamlit run app.py
