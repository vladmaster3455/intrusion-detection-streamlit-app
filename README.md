# Application de Classification d'Intrusions Réseau (NBAI-IOT Dataset)

Ce dépôt contient une application de Machine Learning simple pour la classification d'intrusions réseau, entraînée sur un jeu de données simulé inspiré du dataset NBAI-IOT. L'application est développée avec Streamlit et est conteneurisée à l'aide de Docker pour un déploiement facile.

## Table des matières

- [Description du Projet](#description-du-projet)
- [Algorithmes ML Utilisés](#algorithmes-ml-utilisés)
- [Prétraitement des Données](#prétraitement-des-données)
- [Déploiement Local avec Docker](#déploiement-local-avec-docker)
- [Utilisation de l'Application Streamlit](#utilisation-de-lapplication-streamlit)
- [Déploiement en Ligne (Hugging Face Spaces)](#déploiement-en-ligne-hugging-face-spaces)
- [Fichiers du Projet](#fichiers-du-projet)

## Description du Projet

L'objectif de ce projet est de construire et de déployer un modèle de classification capable d'identifier différentes catégories d'activités réseau (trafic "benign" ou divers types d'attaques comme Mirai ou Gafgyt) basées sur des caractéristiques de flux réseau. Pour des raisons de performance et de facilité de démonstration, le modèle est entraîné sur un jeu de données synthétique qui mime la structure et les caractéristiques du jeu de données original NBAI-IOT.

## Algorithmes ML Utilisés

Trois algorithmes de Machine Learning ont été explorés et comparés, puis le meilleur modèle a été optimisé :

1.  **RandomForest (Forêt Aléatoire)** :
    * Un algorithme d'apprentissage ensembliste qui construit un grand nombre d'arbres de décision. Il est robuste au surapprentissage et efficace sur de grandes bases de données.
2.  **SVM (Support Vector Machine - Machine à Vecteurs de Support)** :
    * Un algorithme puissant pour la classification, basé sur la recherche d'un hyperplan optimal. Il est efficace dans les espaces de grande dimension et peut gérer des séparations non linéaires.
3.  **GradientBoosting (Gradient Boosting Machines)** :
    * Une technique d'apprentissage ensembliste qui construit séquentiellement des modèles faibles pour corriger les erreurs des modèles précédents, visant une haute précision itérative.

Le modèle le plus performant après une évaluation initiale (basée sur le score F1 pondéré et la matrice de confusion) est choisi et optimisé via une recherche sur grille (GridSearchCV) pour affiner ses hyperparamètres.

## Prétraitement des Données

Les données brutes, qu'elles soient d'entraînement ou de nouvelles données pour la prédiction, subissent les étapes de prétraitement suivantes :

-   **Remplacement des sous-catégories vides** : Les entrées vides dans la colonne 'subcategory' sont remplacées par 'benign'.
-   **Normalisation Min-Max** : Toutes les caractéristiques numériques sont normalisées à une échelle de 0 à 1 en utilisant la formule : `(valeur - min) / (max - min)`. Les paramètres `min` et `max` sont appris sur le jeu de données d'entraînement et sauvegardés pour être appliqués de manière cohérente aux nouvelles données.

## Déploiement Local avec Docker

Pour exécuter cette application localement dans un conteneur Docker :

**Prérequis :**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et en cours d'exécution.
- [Python 3.9+](https://www.python.org/downloads/) installé (pour exécuter le script d'entraînement).

**Étapes :**

1.  **Clonez ce dépôt ou téléchargez les fichiers :**
    ```bash
    git clone [https://github.com/vladmaster3455/intrusion-detection-streamlit-app.git](https://github.com/vladmaster3455/intrusion-detection-streamlit-app.git)
    cd intrusion-detection-streamlit-app 
    ```


2.  **Exécutez le script d'entraînement pour générer les modèles `.pkl` :**
    Ce script va entraîner le modèle, l'optimiser et sauvegarder les fichiers `.pkl` nécessaires (`optimized_model.pkl`, `label_encoder.pkl`, `scaler_params.pkl`) dans le répertoire courant.
    ```bash
    python train_and_save_model.py
    ```
    *Assurez-vous monsieur  prof IA que toutes les dépendances sont installées sur votre environnement local avant d'exécuter cette commande. Si vous rencontrez des `ModuleNotFoundError`, exécutez `pip install -r requirements.txt` dans votre environnement Python local en dehors de Docker.*

3.  **Construisez l'image Docker :**
    Assurez-vous que Docker Desktop est lancé et que votre terminal est exécuté en tant qu'administrateur.
    ```bash
    docker build -t intrusion-detection-app .
    ```

4.  **Exécutez le conteneur Docker :**
    ```bash
    docker run -d -p 8501:8501 --name my-intrusion-app intrusion-detection-app
    ```

5.  **Accédez à l'application :**
    Ouvrez votre navigateur web et naviguez vers `http://localhost:8501`.

## Utilisation de l'Application Streamlit

L'interface de l'application Streamlit vous demandera d'entrer **115 valeurs numériques brutes** (non normalisées) séparées par des virgules dans une zone de texte. L'application se chargera de normaliser ces valeurs avant de les passer au modèle pour la prédiction.

### Exemple de données d'entrée brutes (pour tester) :

Vous pouvez exécuter le code Python suivant dans votre terminal pour générer un échantillon de 115 valeurs brutes à copier/coller dans l'application :

```python
import numpy as np
import pandas as pd

def generate_synthetic_data_for_app_test(num_samples=1, num_features=115):
    np.random.seed(123) # Seed pour la reproductibilité
    data = np.random.rand(num_samples, num_features) * 100 # Valeurs entre 0 et 100
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])
    return df

sample_data_for_app = generate_synthetic_data_for_app_test(num_samples=1)
features_string_for_app = ', '.join(map(str, sample_data_for_app.iloc[0].tolist()))
print("Copiez cette chaîne dans la zone de texte de l'application Streamlit :")
print(features_string_for_app)
