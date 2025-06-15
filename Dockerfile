# Utilise l'image officielle Python 3.9-slim (une version légère de Python)
FROM python:3.9-slim

# Définit le répertoire de travail dans le conteneur Docker
WORKDIR /app

# Copie tous les fichiers de votre répertoire local courant (C:\Users\Vladmaster\Music\IA\)
# vers le répertoire de travail (/app) dans le conteneur.
# Cela inclut app.py, optimized_model.pkl, label_encoder.pkl, scaler_params.pkl, requirements.txt
COPY . /app

# Installe les dépendances Python spécifiées dans requirements.txt
# L'option --no-cache-dir réduit la taille de l'image Docker finale
RUN pip install --no-cache-dir --default-timeout=120 wheel && \
    pip install --no-cache-dir --default-timeout=120 --upgrade --force-reinstall -r requirements.txt
# Expose le port 8501, qui est le port par défaut utilisé par Streamlit
EXPOSE 8501

# Commande à exécuter lorsque le conteneur démarre.
# Elle lance l'application Streamlit.
# --server.port=8501: Spécifie le port interne du conteneur.
# --server.address=0.0.0.0: Rend l'application accessible depuis l'extérieur du conteneur.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]