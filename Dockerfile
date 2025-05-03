# ------------- Stage 1: Base -------------
# Utilise une image Python 3.11 slim officielle basée sur Debian Bookworm
FROM python:3.11-slim-bookworm AS base

# Définir des arguments pour les IDs utilisateur/groupe (bonne pratique)
ARG UID=10001
ARG GID=10001

# Variables d'environnement utiles
ENV PYTHONUNBUFFERED=1 \
    # Chemin pour l'environnement virtuel
    VIRTUAL_ENV=/opt/venv \
    # Empêcher les prompts interactifs apt
    DEBIAN_FRONTEND=noninteractive

# Créer le répertoire pour le venv
RUN mkdir -p $VIRTUAL_ENV

# Créer un groupe et un utilisateur non-root dédiés
RUN groupadd --gid $GID nonroot && \
    useradd --uid $UID --gid $GID --shell /bin/bash --create-home nonroot

# Mettre à jour les paquets et installer les dépendances OS minimales requises
# libgl1 : Souvent requis par opencv-python-headless
# libglib2.0-0 : Fournit libgthread-2.0.so.0, nécessaire pour OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    # Nettoyer pour réduire la taille de l'image
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Créer l'environnement virtuel et activer (pour les étapes suivantes)
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /app

# ------------- Stage 2: Builder -------------
# Cette étape installe les dépendances Python
FROM base AS builder

# Copier uniquement requirements.txt pour utiliser le cache Docker efficacement
COPY --chown=nonroot:nonroot requirements.txt .

# Mettre à jour pip et installer les dépendances sans cache
# Utiliser --no-cache-dir réduit la taille de la couche
# On installe en tant que root mais dans le venv qui sera copié
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------- Stage 3: Final Image -------------
# Recommencer depuis l'image de base légère
FROM base AS final

# Copier l'environnement virtuel avec les dépendances depuis l'étape builder
# S'assurer que l'utilisateur non-root en est propriétaire
COPY --from=builder --chown=nonroot:nonroot $VIRTUAL_ENV $VIRTUAL_ENV

# Copier tout le code de l'application dans le répertoire de travail
# Assure-toi que ton .dockerignore est bien configuré
# Donne la propriété à l'utilisateur non-root
COPY --chown=nonroot:nonroot . .

# Changer pour l'utilisateur non-root pour l'exécution
USER nonroot

# Exposer le port sur lequel Gunicorn écoutera (Cloud Run injecte $PORT)
# 8080 est une convention courante pour Cloud Run
EXPOSE 8080

# Définir la commande pour lancer l'application
# Utilise la forme "shell" pour que $PORT soit interprété
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]