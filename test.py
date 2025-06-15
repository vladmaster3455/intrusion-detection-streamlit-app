import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1, num_features=115):
    np.random.seed(99) # Utiliser une seed différente pour un échantillon unique
    data = np.random.rand(num_samples, num_features) * 100 # Multiplier par 100 pour avoir des valeurs plus "brutes"
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])
    # La fonction originale ajoute aussi 'category' et 'subcategory', mais on n'en a pas besoin ici.
    return df

# Générer un seul échantillon de 115 caractéristiques brutes
sample_data_df = generate_synthetic_data(num_samples=1, num_features=115)

# Convertir le DataFrame en une liste de valeurs, puis en chaîne de caractères séparées par des virgules
features_string = ', '.join(map(str, sample_data_df.iloc[0].tolist()))
print(features_string)