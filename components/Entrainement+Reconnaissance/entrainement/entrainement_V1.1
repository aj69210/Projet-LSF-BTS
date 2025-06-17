import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# --- CONFIGURATION PRINCIPALE ---
# 1. Chemins
DATA_PATH = './data_processed'  # Dossier contenant les .npy (sortie de process_data.py)
SAVE_DIR = './model_trained'   # Dossier où sauvegarder le modèle et les artefacts

# 2. Paramètres des données
SEQUENCE_LENGTH = 27  # Doit correspondre à la longueur de séquence de vos données .npy
# !! IMPORTANT !! Le nombre de features est 1662, pas 1280.
# (33*4 pour la pose) + (468*3 pour le visage) + (21*3 pour main gauche) + (21*3 pour main droite) = 1662
NUM_FEATURES = 1662

# 3. Paramètres du modèle et de l'entraînement
LSTM_UNITS = 128  # Unités dans les couches LSTM (128 est un bon point de départ)
EPOCHS = 300      # Nombre maximal d'époques (EarlyStopping s'en chargera)
BATCH_SIZE = 32   # Taille du lot. Ajustez selon votre VRAM (32 ou 64 sont courants)
VALIDATION_SPLIT_SIZE = 0.20 # 20% des données pour la validation
MIN_SAMPLES_PER_CLASS = 10 # Nombre minimum de séquences pour qu'une classe soit utilisée

# 4. Paramètres des Callbacks
PATIENCE_EARLY_STOPPING = 30
PATIENCE_REDUCE_LR = 15

# 5. Paramètres d'augmentation de données
AUG_NOISE_LEVEL = 0.005 # Ajoute un léger bruit pour la robustesse
AUG_TIME_MASK_MAX_PERCENTAGE = 0.10 # Masque jusqu'à 10% des frames d'une séquence
# --- FIN DE LA CONFIGURATION ---

os.makedirs(SAVE_DIR, exist_ok=True)

def load_sequences_and_labels(data_path):
    """
    Charge les séquences de keypoints (.npy) et leurs labels depuis les sous-dossiers.
    """
    sequences, labels = [], []
    print(f"Chargement des données depuis : {data_path}")
    
    sign_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    for sign in tqdm(sign_folders, desc="Chargement des signes"):
        sign_path = os.path.join(data_path, sign)
        for seq_file in os.listdir(sign_path):
            if seq_file.endswith('.npy'):
                try:
                    res = np.load(os.path.join(sign_path, seq_file))
                    # Validation cruciale de la forme des données
                    if res.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                        sequences.append(res)
                        labels.append(sign)
                    else:
                        print(f"  [!] Ignoré : {seq_file} (forme {res.shape} incorrecte, attendue {(SEQUENCE_LENGTH, NUM_FEATURES)})")
                except Exception as e:
                    print(f"  [!] Erreur chargement {seq_file}: {e}")
                    
    return np.array(sequences), np.array(labels)

def create_model(input_shape, num_classes):
    """
    Crée l'architecture du modèle LSTM Bidirectionnel.
    """
    print("Création de l'architecture du modèle...")
    inputs = Input(shape=input_shape)
    reg_val = 0.0005 # Valeur de régularisation L2 pour éviter le surapprentissage

    # Couche 1
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=l2(reg_val)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Couche 2
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=l2(reg_val)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Couche 3 - La dernière couche LSTM ne retourne pas de séquence
    x = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=False, kernel_regularizer=l2(reg_val)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Couches denses pour la classification
    x = Dense(LSTM_UNITS // 2, activation='relu', kernel_regularizer=l2(reg_val))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print("Modèle compilé.")
    return model

def plot_history_and_confusion_matrix(history, y_true_classes, y_pred_classes, class_names):
    """Affiche l'historique de l'entraînement et la matrice de confusion."""
    # Plot de l'historique
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy Entraînement')
    plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
    plt.title('Accuracy du Modèle')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perte Entraînement')
    plt.plot(history.history['val_loss'], label='Perte Validation')
    plt.title('Perte du Modèle')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'))
    plt.show()

    # Matrice de confusion
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vrai Label')
    plt.xlabel('Label Prédit')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
    plt.show()


# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
print("--- DÉBUT PRÉPARATION DES DONNÉES ---")
X, y = load_sequences_and_labels(DATA_PATH)

if len(X) == 0:
    raise ValueError("Aucune donnée chargée. Vérifiez le DATA_PATH et le contenu des fichiers .npy.")

# Filtrage des classes avec peu d'échantillons
print("\n--- Filtrage des classes ---")
label_counts = Counter(y)
X_filtered, y_filtered = [], []
for x_seq, y_label in zip(X, y):
    if label_counts[y_label] >= MIN_SAMPLES_PER_CLASS:
        X_filtered.append(x_seq)
        y_filtered.append(y_label)

if not y_filtered:
    raise ValueError(f"Aucune classe n'a au moins {MIN_SAMPLES_PER_CLASS} échantillons. Baissez le seuil ou capturez plus de données.")

X = np.array(X_filtered)
y = np.array(y_filtered)
print(f"Données après filtrage : {X.shape[0]} séquences pour {len(set(y))} classes.")

# Encodage des labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
class_names = label_encoder.classes_
num_classes = len(class_names)

# Sauvegarde du LabelEncoder
with open(os.path.join(SAVE_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder sauvegardé. Classes : {class_names}")

# Division des données
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=VALIDATION_SPLIT_SIZE, random_state=42, stratify=y_categorical
)
print(f"\nDonnées divisées : {len(X_train)} pour l'entraînement, {len(X_val)} pour la validation.")

# Normalisation des données
# Très important : calculer la moyenne et l'écart-type UNIQUEMENT sur les données d'entraînement
mean = np.mean(X_train, axis=(0, 1))
std = np.std(X_train, axis=(0, 1))

# Sauvegarde pour l'inférence future
np.save(os.path.join(SAVE_DIR, 'normalization_mean.npy'), mean)
np.save(os.path.join(SAVE_DIR, 'normalization_std.npy'), std)
print("Moyenne et écart-type de normalisation sauvegardés.")

# Appliquer la normalisation
X_train = (X_train - mean) / (std + 1e-8)
X_val = (X_val - mean) / (std + 1e-8)


# --- 2. DÉFINITION DU MODÈLE ET DES CALLBACKS ---
print("\n--- CONFIGURATION DU MODÈLE ---")
model = create_model(input_shape=(SEQUENCE_LENGTH, NUM_FEATURES), num_classes=num_classes)
model.summary()

# Callbacks pour un entraînement intelligent
best_model_path = os.path.join(SAVE_DIR, 'sign_language_model.h5')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE_EARLY_STOPPING, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=PATIENCE_REDUCE_LR, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)


# --- 3. ENTRAÎNEMENT DU MODÈLE ---
print("\n--- DÉBUT DE L'ENTRAÎNEMENT ---")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# --- 4. ÉVALUATION DU MODÈLE ---
print("\n--- ÉVALUATION FINALE DU MEILLEUR MODÈLE ---")
# Le meilleur modèle est déjà chargé grâce à restore_best_weights=True ou sauvegardé par ModelCheckpoint
# Nous chargeons depuis le fichier pour être sûr
best_model = tf.keras.models.load_model(best_model_path)

loss, accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print(f"Perte en validation : {loss:.4f}")
print(f"Précision en validation : {accuracy * 100:.2f}%")

# Rapport de classification détaillé
y_pred_probs = best_model.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

print("\nRapport de Classification :")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))

# Affichage des graphiques
plot_history_and_confusion_matrix(history, y_true_classes, y_pred_classes, class_names)

print("\n--- SCRIPT TERMINÉ ---")
print(f"Le meilleur modèle et les artefacts sont sauvegardés dans le dossier : {os.path.abspath(SAVE_DIR)}")
