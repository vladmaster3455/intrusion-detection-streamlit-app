import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Importer la bibliothèque joblib pour la sauvegarde/chargement des modèles
import os # AJOUTÉ : Importation du module os pour obtenir le répertoire de travail

# --- 1. Simuler un petit device_df pour une exécution plus rapide ---
def generate_synthetic_data(num_samples=10000, num_features=115):
    np.random.seed(42)
    data = np.random.rand(num_samples, num_features) * 100 # Données float aléatoires
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])

    # Ajouter les colonnes 'category' et 'subcategory'
    categories = ['mirai', 'gafgyt', 'benign']
    subcategories = ['udp', 'benign', 'combo', 'udpplain', 'tcp', 'ack', 'scan', 'junk', 'syn']

    # Répartition des catégories et sous-catégories pour simuler des déséquilibres
    df['category'] = np.random.choice(categories, num_samples, p=[0.65, 0.30, 0.05])
    df['subcategory'] = np.random.choice(subcategories, num_samples, p=[0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

    # S'assurer que la catégorie 'benign' a la sous-catégorie 'benign'
    df.loc[df['category'] == 'benign', 'subcategory'] = 'benign'

    return df

# Générer un jeu de données plus petit
device_df = generate_synthetic_data(num_samples=10000) # Nombre réduit d'échantillons
print(f"device_df simulé : {device_df.shape[0]} lignes, {device_df.shape[1]} colonnes")
print("Sous-catégories dans les données simulées :", device_df['subcategory'].unique())

# --- 2. Prétraitement (similaire au notebook original) ---
numeric_cols = device_df.select_dtypes(include='number').columns.tolist()
device_df[numeric_cols] = device_df[numeric_cols].astype('float32')

mins   = device_df[numeric_cols].min()
maxs   = device_df[numeric_cols].max()
ranges = (maxs - mins).replace(0, 1.0)

for col in numeric_cols:
    device_df[col] = (device_df[col] - mins[col]) / ranges[col]

print("\nAprès la normalisation Min-Max (5 premières lignes des colonnes numériques) :\n", device_df[numeric_cols[:5]].head())

# --- Préparation des Données (similaire au notebook original) ---
X = device_df.drop(columns=['category', 'subcategory'])
y = device_df['category']

# Division des données en ensembles d'entraînement, de validation et de test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Encodage des étiquettes de catégorie
le = LabelEncoder().fit(y_train)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

print(f"\nFormes des ensembles de données après division :")
print(f"X_train: {X_train.shape}, y_train: {y_train_enc.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test_enc.shape}")

# Dictionnaire pour stocker les résultats
results = {}

# --- Fonction d'aide pour tracer la matrice de confusion ---
def plot_confusion_matrix(cm, labels, title='Matrice de Confusion'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Étiquette Prédite')
    plt.ylabel('Étiquette Réelle')
    plt.show()

# --- 3. Évaluation Initiale des Modèles de Machine Learning ---

# Modèle 1 : RandomForestClassifier
name = 'RandomForest'
model_rf = RandomForestClassifier(random_state=42)
print(f"\n--- Entraînement et Évaluation Initiale du modèle : {name} ---")
print(f"Entraînement du modèle : {name}...")
model_rf.fit(X_train, y_train_enc)

print(f"Évaluation du modèle : {name} sur l'ensemble de test...")
preds_rf = model_rf.predict(X_test)
pred_labels_rf = le.inverse_transform(preds_rf)

f1_rf = f1_score(y_test, pred_labels_rf, average='weighted')
conf_matrix_rf = confusion_matrix(y_test, pred_labels_rf)

results[name] = {'f1_score': f1_rf, 'confusion_matrix': conf_matrix_rf}

print(f"{name} - Score F1: {f1_rf:.4f}")
print(f"Matrice de Confusion :\n{conf_matrix_rf}")
plot_confusion_matrix(conf_matrix_rf, le.classes_, title=f'Matrice de Confusion {name} (Initiale)')

# Modèle 2 : SVC (Support Vector Machine)
# Remarque : SVC peut être coûteux en calcul même sur des jeux de données plus petits.
# Pour la démonstration, nous utilisons l'ensemble complet comme dans le notebook original.
name = 'SVM'
model_svm = SVC(probability=True, random_state=42)
print(f"\n--- Entraînement et Évaluation Initiale du modèle : {name} ---")
print(f"Entraînement du modèle : {name}...")
model_svm.fit(X_train, y_train_enc)

print(f"Évaluation du modèle : {name} sur l'ensemble de test...")
preds_svm = model_svm.predict(X_test)
pred_labels_svm = le.inverse_transform(preds_svm)

f1_svm = f1_score(y_test, pred_labels_svm, average='weighted')
conf_matrix_svm = confusion_matrix(y_test, pred_labels_svm)

results[name] = {'f1_score': f1_svm, 'confusion_matrix': conf_matrix_svm}

print(f"{name} - Score F1: {f1_svm:.4f}")
print(f"Matrice de Confusion :\n{conf_matrix_svm}")
plot_confusion_matrix(conf_matrix_svm, le.classes_, title=f'Matrice de Confusion {name} (Initiale)')

# Modèle 3 : GradientBoostingClassifier
name = 'GradientBoosting'
model_gb = GradientBoostingClassifier(random_state=42)
print(f"\n--- Entraînement et Évaluation Initiale du modèle : {name} ---")
print(f"Entraînement du modèle : {name}...")
model_gb.fit(X_train, y_train_enc)

print(f"Évaluation du modèle : {name} sur l'ensemble de test...")
preds_gb = model_gb.predict(X_test)
pred_labels_gb = le.inverse_transform(preds_gb)

f1_gb = f1_score(y_test, pred_labels_gb, average='weighted')
conf_matrix_gb = confusion_matrix(y_test, pred_labels_gb)

results[name] = {'f1_score': f1_gb, 'confusion_matrix': conf_matrix_gb}

print(f"{name} - Score F1: {f1_gb:.4f}")
print(f"Matrice de Confusion :\n{conf_matrix_gb}")
plot_confusion_matrix(conf_matrix_gb, le.classes_, title=f'Matrice de Confusion {name} (Initiale)')


# --- 4. Optimisation du Modèle ---

# Identification du meilleur modèle basé sur le Score F1 initial
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
print(f"\n--- Optimisation du Modèle ---")
print(f"Meilleur modèle initial : {best_model_name} avec un Score F1 : {results[best_model_name]['f1_score']:.4f}")

# Définition de la grille de paramètres pour le modèle choisi
param_grid = {}
estimator = None

if best_model_name == 'RandomForest':
    # Paramètres réduits pour une démonstration plus rapide
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    estimator = RandomForestClassifier(random_state=42)
elif best_model_name == 'SVM':
    # Paramètres limités pour une démonstration plus rapide de SVM
    param_grid = {'C': [0.1, 1], 'kernel': ['linear']}
    estimator = SVC(probability=True, random_state=42)
else:  # GradientBoosting
    # Paramètres réduits pour une démonstration plus rapide
    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
    estimator = GradientBoostingClassifier(random_state=42)

print(f"\nLancement de l'optimisation (Grid Search) pour {best_model_name}...")
# cv réduit à 2 pour la vitesse de démonstration
grid_search = GridSearchCV(estimator, param_grid, cv=2, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train_enc)

optimized_model = grid_search.best_estimator_

print(f"\nMeilleurs hyperparamètres trouvés pour {best_model_name} : {grid_search.best_params_}")

# --- Évaluation du Modèle Optimisé ---
print(f"\nÉvaluation du modèle optimisé sur l'ensemble de test...")
preds_opt = optimized_model.predict(X_test)
pred_labels_opt = le.inverse_transform(preds_opt)
f1_opt = f1_score(y_test, pred_labels_opt, average='weighted')

print(f"\nScore F1 après optimisation : {f1_opt:.4f}")
print("Rapport de classification du modèle optimisé :")
print(classification_report(y_test, pred_labels_opt, digits=4))
print("Matrice de confusion du modèle optimisé :")
conf_matrix_opt = confusion_matrix(y_test, pred_labels_opt)
print(conf_matrix_opt)
plot_confusion_matrix(conf_matrix_opt, le.classes_, title=f'Matrice de Confusion {best_model_name} (Optimisée)')

# --- PARTIE AJOUTÉE : Sauvegarde du Modèle Optimisé, LabelEncoder et Scalers ---
# Ces fichiers seront ensuite utilisés pour simuler le déploiement (inférence)
model_filename = 'optimized_model.pkl' # Nom générique
label_encoder_filename = 'label_encoder.pkl'
scaler_params_filename = 'scaler_params.pkl'

print(f"\n--- Début de la sauvegarde des artefacts du modèle pour l'inférence ---")
print(f"Répertoire de travail actuel : {os.getcwd()}") # Ajout pour voir le répertoire

# Sauvegarder le modèle optimisé
try:
    joblib.dump(optimized_model, model_filename)
    print(f"Modèle sauvegardé sous : {model_filename}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde du modèle : {e}")

# Sauvegarder le LabelEncoder
try:
    joblib.dump(le, label_encoder_filename)
    print(f"LabelEncoder sauvegardé sous : {label_encoder_filename}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde du LabelEncoder : {e}")

# Sauvegarder les paramètres de normalisation (mins et ranges)
try:
    scaler_params = {
        'mins': mins.to_dict(),
        'ranges': ranges.to_dict(),
        'numeric_cols': numeric_cols # Sauvegarder aussi les noms des colonnes numériques
    }
    joblib.dump(scaler_params, scaler_params_filename)
    print(f"Paramètres de normalisation sauvegardés sous : {scaler_params_filename}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde des paramètres de normalisation : {e}")

print(f"--- Fin de la sauvegarde des artefacts du modèle ---")


# --- SIMULATION DE DÉPLOIEMENT (INFÉRENCE SUR NOUVELLES DONNÉES DANS LE MÊME NOTEBOOK KAGGLE) ---

print("\n" + "="*80)
print("             SIMULATION DE DÉPLOIEMENT (INFÉRENCE SUR KAGGLE)")
print("="*80 + "\n")

# 1. Charger les artefacts sauvegardés
print("\nÉtape 1: Chargement du modèle, de l'encodeur et des paramètres de normalisation pour l'inférence...")
try:
    # Les fichiers sont sauvegardés dans /kaggle/working/ par défaut
    # Dans un notebook d'inférence séparé, vous devrez ajouter l'output de ce notebook
    # comme source de données et ajuster le chemin si nécessaire.
    # Ex: model_path = '/kaggle/input/nom-de-votre-notebook-output/optimized_model.pkl'
    loaded_model = joblib.load(model_filename)
    loaded_le = joblib.load(label_encoder_filename)
    loaded_scaler_params = joblib.load(scaler_params_filename)
    loaded_mins = pd.Series(loaded_scaler_params['mins'])
    loaded_ranges = pd.Series(loaded_scaler_params['ranges'])
    loaded_numeric_cols = loaded_scaler_params['numeric_cols']

    print("Tous les artefacts nécessaires chargés avec succès à partir de /kaggle/working/.")
except FileNotFoundError as e:
    print(f"Erreur: Fichier non trouvé. Assurez-vous d'avoir exécuté toutes les cellules précédentes pour sauvegarder les fichiers. {e}")
    # Dans un vrai notebook d'inférence, vous auriez besoin d'ajouter les outputs du notebook d'entraînement comme Data Source
    exit()
except Exception as e:
    print(f"Une erreur inattendue est survenue lors du chargement des artefacts : {e}")
    exit()

# 2. Simuler de nouvelles données à prédire (représentant des données de production ou de test)
# Il est crucial que ces nouvelles données aient la même structure (mêmes features)
# que les données sur lesquelles le modèle a été entraîné.
num_new_samples_for_inference = 5 # Simuler 5 nouveaux échantillons pour la démo d'inférence
print(f"\nÉtape 2: Génération de {num_new_samples_for_inference} nouvelles données simulées pour l'inférence...")

# La fonction generate_synthetic_data génère aussi 'category'/'subcategory'.
# Pour l'inférence, nous n'aurions que les features (X).
new_simulated_df = generate_synthetic_data(num_samples=num_new_samples_for_inference)
X_new_raw = new_simulated_df.drop(columns=['category', 'subcategory'])

print(f"Nouvelles données brutes pour l'inférence (premières {num_new_samples_for_inference} lignes, {X_new_raw.shape[1]} features) :")
print(X_new_raw.head())

# 3. Prétraiter les nouvelles données en utilisant les paramètres sauvegardés
print("\nÉtape 3: Prétraitement des nouvelles données en utilisant les paramètres de normalisation chargés...")
X_new_processed = X_new_raw.copy()

# Assurez-vous d'appliquer la normalisation uniquement aux colonnes numériques et d'utiliser les loaded_mins/ranges
for col in loaded_numeric_cols: # Utiliser la liste des colonnes numériques sauvegardées
    if col in X_new_processed.columns: # Vérifier si la colonne existe dans les nouvelles données
        if col in loaded_mins and col in loaded_ranges:
            if loaded_ranges[col] != 0:
                X_new_processed[col] = (X_new_processed[col] - loaded_mins[col]) / loaded_ranges[col]
            else:
                # Gérer les colonnes constantes: si range est 0, la valeur normalisée est 0
                X_new_processed[col] = 0.0
        else:
            print(f"Attention: Paramètres de normalisation manquants pour la colonne {col}. La colonne ne sera pas normalisée.")
    else:
        print(f"Attention: La colonne '{col}' de l'entraînement est absente des nouvelles données.")


print("Nouvelles données prétraitées (premières 5 lignes) :")
print(X_new_processed.head())

# 4. Effectuer les prédictions avec le modèle chargé
print("\nÉtape 4: Effectuation des prédictions sur les nouvelles données...")
predictions_encoded = loaded_model.predict(X_new_processed)

# 5. Inverse-transformer les prédictions en étiquettes originales lisibles
predictions_labels = loaded_le.inverse_transform(predictions_encoded)

print("\nÉtape 5: Résultats des prédictions pour les nouvelles données :")
for i in range(num_new_samples_for_inference):
    print(f"  Échantillon {i+1} : Catégorie prédite = {predictions_labels[i]}")

print("\n" + "="*80)
print("             FIN DE LA SIMULATION DE DÉPLOIEMENT/INFÉRENCE")
print("             (Ces étapes seraient dans un notebook séparé en production)")
print("="*80 + "\n")

# --- 5. Présentation des Algorithmes de Machine Learning Utilisés (inchangé) ---
print("\n--- Présentation des algorithmes de Machine Learning utilisés : ---\n")
print("- **RandomForest (Forêt Aléatoire)** :")
print("  > C'est un algorithme d'apprentissage ensembliste qui construit un grand nombre d'arbres de décision lors de l'entraînement.")
print("  > Pour la classification, il produit la classe qui est le mode des classes (classification) ou la moyenne (régression) des prédictions des arbres individuels.")
print("  > Sa force réside dans sa capacité à gérer de grandes bases de données avec un bon niveau de précision, et sa robustesse au surapprentissage grâce à la diversification des arbres.")

print("\n- **SVM (Support Vector Machine - Machine à Vecteurs de Support)** :")
print("  > Un algorithme puissant pour la classification et la régression, basé sur la recherche d'un hyperplan optimal qui sépare les données en classes.")
print("  > Il maximise la marge entre l'hyperplan et les points de données les plus proches (vecteurs de support).")
print("  > Il est efficace dans les espaces de grande dimension et peut utiliser différentes fonctions de noyau (linéaire, RBF, etc.) pour des séparations complexes non linéaires.")

print("\n- **GradientBoosting (Gradient Boosting Machines)** :")
print("  > Une autre technique d'apprentissage ensembliste qui construit des modèles faibles (souvent des arbres de décision peu profonds) de manière séquentielle.")
print("  > Chaque nouvel arbre corrige les erreurs des arbres précédents en se concentrant sur les résidus (les erreurs non expliquées).")
print("  > Il s'agit d'une technique de 'boosting' qui vise à améliorer la précision de manière itérative, souvent très performante mais potentiellement plus sujette au surapprentissage si mal paramétrée.")