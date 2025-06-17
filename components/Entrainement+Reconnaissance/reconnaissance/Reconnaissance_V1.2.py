# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import os
import logging

# --- Configuration Alignée sur l'Entraînement ---
SEQUENCE_LENGTH = 100
THRESHOLD = 0.5  # Ajustez ce seuil selon les performances observées
SAVE_DIR = "data/models"

# --- Chemins des Fichiers Alignés sur la Sortie du Script d'Entraînement ---
MODEL_PATH = os.path.join(SAVE_DIR, 'sign_language_recognition_best.h5')
LABEL_ENCODER_PATH = os.path.join(SAVE_DIR, 'label_encoder.pkl')
MEAN_PATH = os.path.join(SAVE_DIR, 'normalization_mean.npy')
STD_PATH = os.path.join(SAVE_DIR, 'normalization_std.npy')

# --- Configuration des Features (Doit correspondre à l'entraînement) ---
EXPECTED_FEATURES = 258
NUM_HAND_KEYPOINTS = 21
NUM_COORDS_HAND = 3 # x, y, z
NUM_COORDS_POSE = 4 # x, y, z, visibilité
POSE_IDX = [
    0, 11, 12, 13, 14, 15, 16, 23, 24,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    17, 18, 19, 20
]
MOUTH_IDX = [
    13, 14, 78, 308, 0, 17, 61, 291, 146, 375
]

print(f"--- Configuration ---")
print(f"Features Attendues : {EXPECTED_FEATURES}")
print(f"Longueur de Séquence : {SEQUENCE_LENGTH}")
print(f"Seuil de Confiance : {THRESHOLD}")
print(f"Chemin Modèle : {MODEL_PATH}")
print(f"Chemin Encodeur : {LABEL_ENCODER_PATH}")
print(f"Chemin Moyenne Norm. : {MEAN_PATH}")
print(f"Chemin Écart-Type Norm. : {STD_PATH}")
print(f"---------------------")

# --- Initialisation MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic_model = None

# --- Chargement Modèle, Encodeur, et Paramètres de Normalisation ---
model = None
label_encoder = None
actions = []
mean = None
std = None

print("Chargement des ressources...")
try:
    # Charger le Modèle
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Fichier modèle non trouvé : {MODEL_PATH}")
    # Supprimer les avertissements non critiques lors du chargement si nécessaire
    # Optionnel: from tensorflow.python.util import deprecation
    # Optionnel: deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Optionnel: import absl.logging
    # Optionnel: absl.logging.set_verbosity(absl.logging.ERROR)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False) # Mettre compile=False peut éviter l'avertissement sur les métriques
    # Recompiler si nécessaire pour l'inférence (souvent pas indispensable)
    # model.compile() # Ou avec un optimizer si besoin spécifique, mais pas pour predict
    print(f"Modèle chargé avec succès depuis : {MODEL_PATH}")

    # Charger LabelEncoder
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Fichier LabelEncoder non trouvé : {LABEL_ENCODER_PATH}")
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    if not hasattr(label_encoder, 'classes_'):
        raise TypeError(f"L'objet chargé depuis {LABEL_ENCODER_PATH} n'est pas un LabelEncoder valide.")
    actions = label_encoder.classes_
    print(f"LabelEncoder chargé avec succès. Actions : {list(actions)}")

    # Charger les Paramètres de Normalisation (Moyenne/Écart-type)
    if not os.path.exists(MEAN_PATH):
         raise FileNotFoundError(f"Fichier de moyenne de normalisation non trouvé : {MEAN_PATH}")
    if not os.path.exists(STD_PATH):
         raise FileNotFoundError(f"Fichier d'écart-type de normalisation non trouvé : {STD_PATH}")
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    print(f"Paramètres de normalisation bruts chargés (Shape Moyenne : {mean.shape}, Shape Écart-Type : {std.shape})")

    # --- CORRECTION IMPORTANTE : Remodeler mean et std ---
    # Les remodeler en (EXPECTED_FEATURES,) pour un broadcasting correct avec (SEQUENCE_LENGTH, EXPECTED_FEATURES)
    mean = mean.reshape(-1) # Devient (258,)
    std = std.reshape(-1)  # Devient (258,)
    print(f"Paramètres de normalisation remodelés (Shape Moyenne : {mean.shape}, Shape Écart-Type : {std.shape})")
    # --- Fin de la Correction ---

    # Vérifier si les dimensions correspondent aux features attendues après reshape
    if mean.shape[0] != EXPECTED_FEATURES or std.shape[0] != EXPECTED_FEATURES:
         # Cette erreur ne devrait plus se produire avec reshape(-1) si les fichiers npy sont corrects
         print(f"!!! ATTENTION !!! Les dimensions des paramètres de normalisation ({mean.shape[0]}, {std.shape[0]}) après reshape ne correspondent pas aux features attendues ({EXPECTED_FEATURES}).")
         exit()

    # Initialiser le modèle MediaPipe Holistic ici après les vérifications
    print("Initialisation du modèle MediaPipe Holistic...")
    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("Modèle MediaPipe Holistic initialisé.")

except FileNotFoundError as fnf_error:
    print(f"\n--- ERREUR CRITIQUE (Fichier Manquant) ---")
    print(f"{fnf_error}")
    print(f"Veuillez vous assurer que le script d'entraînement s'est exécuté avec succès et a généré tous les fichiers nécessaires dans le dossier '{SAVE_DIR}'.")
    print("-------------------------------------------")
    exit()
except Exception as e:
    print(f"\n--- ERREUR CRITIQUE (Chargement Ressources) ---")
    print(f"{e}")
    print("-----------------------------------------------")
    if holistic_model:
        holistic_model.close()
    exit()


# --- Fonction de Détection MediaPipe ---
def mediapipe_detection(image, modele_holistic):
    """Traite une image avec le modèle MediaPipe Holistic."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = modele_holistic.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

# --- Fonction d'Extraction de Keypoints (Alignée sur l'Entraînement) ---
def extract_keypoints_258(results):
    """Extrait 258 keypoints (Pose, Mains, Bouche) des résultats MediaPipe."""
    # Pose
    if results.pose_landmarks:
        pose_lm = results.pose_landmarks.landmark
        pose = np.array([[pose_lm[i].x, pose_lm[i].y, pose_lm[i].z, pose_lm[i].visibility] for i in POSE_IDX], dtype=np.float32)
    else:
        pose = np.zeros((len(POSE_IDX), NUM_COORDS_POSE), dtype=np.float32)
    pose = pose.flatten() # Shape (92,)

    # Main gauche
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten() \
        if results.left_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS_HAND, dtype=np.float32) # Shape (63,)

    # Main droite
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten() \
        if results.right_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS_HAND, dtype=np.float32) # Shape (63,)

    # Bouche
    if results.face_landmarks:
        face_lm = results.face_landmarks.landmark
        mouth = np.array([[face_lm[i].x, face_lm[i].y, face_lm[i].z, 1.0]
                          for i in MOUTH_IDX], dtype=np.float32)
    else:
        mouth = np.zeros((len(MOUTH_IDX), NUM_COORDS_POSE), dtype=np.float32)
    mouth = mouth.flatten() # Shape (40,)

    # Concaténer
    keypoints = np.concatenate([pose, lh, rh, mouth]) # Shape (258,)

    # Vérification finale (devrait toujours être 258 maintenant)
    if keypoints.shape[0] != EXPECTED_FEATURES:
        logging.warning(f"Extraction keypoints a produit {keypoints.shape[0]} features, attendu {EXPECTED_FEATURES}. Retourne zéros.")
        return np.zeros(EXPECTED_FEATURES, dtype=np.float32)

    return keypoints

# --- Fonction de Dessin des Landmarks ---
def draw_styled_landmarks(image, results):
    """Dessine les landmarks stylisés sur l'image."""
    # Visage
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
    # Main gauche
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
    # Main droite
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

# --- Boucle Principale de Reconnaissance ---
sequence = []
sentence = []
display_action = ""
last_valid_action_time = time.time()

# --- Configuration Capture Vidéo OpenCV ---
WINDOW_NAME = 'Reconnaissance LSF (Alignee Entrainement)'
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\n--- ERREUR CRITIQUE ---")
    print("Erreur : Impossible d'accéder à la caméra.")
    print("--------------------")
    if holistic_model: holistic_model.close()
    exit()

print("\n-----------------------------------------------------")
print(">>> Démarrage de la Reconnaissance LSF en Temps Réel <<<")
print(f"    Appuyez sur 'q' pour quitter.")
print("-----------------------------------------------------")

try:
    while True:
        # 1. Lire une Frame
        ret, frame = cap.read()
        if not ret:
            print("Erreur lecture frame.")
            break

        # 2. Détection MediaPipe
        image, results = mediapipe_detection(frame, holistic_model)

        # 3. Extraire Keypoints
        keypoints = extract_keypoints_258(results) # Shape (258,)

        # 4. Gérer la Séquence
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:] # Garde les N dernières frames

        current_action_display = ""
        current_confidence = 0.0

        # 5. Prédiction (si séquence complète)
        if len(sequence) == SEQUENCE_LENGTH:
            # --- Préparer les Données ---
            input_data = np.array(sequence, dtype=np.float32) # Shape: (100, 258)

            # --- Normalisation ---
            # mean/std ont maintenant shape (258,)
            # Le broadcasting fonctionnera correctement: (100, 258) - (258,) -> (100, 258)
            normalized_input = (input_data - mean) / (std + 1e-8) # Shape: (100, 258)

            # --- Ajouter Dimension Batch ---
            input_batch = np.expand_dims(normalized_input, axis=0) # Shape: (1, 100, 258)
            # La shape est maintenant CORRECTE pour le modèle !

            # --- Prédiction ---
            try:
                prediction_probabilities = model.predict(input_batch, verbose=0)[0]
                predicted_action_idx = np.argmax(prediction_probabilities)
                current_confidence = prediction_probabilities[predicted_action_idx]

                # --- Logique de Décision ---
                if current_confidence > THRESHOLD:
                    predicted_action = actions[predicted_action_idx]
                    current_action_display = f"{predicted_action} ({current_confidence:.2f})"
                    last_valid_action_time = time.time()

                    if not sentence or predicted_action != sentence[-1]:
                        sentence.append(predicted_action)
                        sentence = sentence[-5:]

            except Exception as e:
                print(f"Erreur durant la prédiction modèle : {e}") # Affichera l'erreur détaillée si elle se produit encore
                current_action_display = "Erreur Prediction"

        # 6. Mise à jour Affichage
        if time.time() - last_valid_action_time > 2.0:
            display_action = ""
        elif current_action_display:
            display_action = current_action_display

        # 7. Dessiner Superpositions
        draw_styled_landmarks(image, results)
        cv2.rectangle(image, (0,0), (image.shape[1], 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, display_action, (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 8. Afficher l'Image
        cv2.imshow(WINDOW_NAME, image) # Met à jour la fenêtre unique

        # 9. Quitter avec 'q'
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            print("\nTouche 'q' pressée, sortie.")
            break

# --- Nettoyage ---
except KeyboardInterrupt:
    print("\nInterruption clavier reçue. Sortie...")
finally:
    print("Libération des ressources...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Caméra libérée.")
    cv2.destroyAllWindows()
    print(f"Fenêtre OpenCV '{WINDOW_NAME}' détruite.")
    if holistic_model:
        holistic_model.close()
        print("Modèle MediaPipe Holistic fermé.")
    print("Nettoyage terminé. Programme fini.")
