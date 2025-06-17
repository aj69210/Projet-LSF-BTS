import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
from collections import deque

# --- CONFIGURATION ALIGNÉE SUR LE SCRIPT D'ENTRAÎNEMENT ---
# 1. Chemins
SAVE_DIR = './model_trained'
MODEL_PATH = os.path.join(SAVE_DIR, 'sign_language_model.h5')
LABEL_ENCODER_PATH = os.path.join(SAVE_DIR, 'label_encoder.pkl')
MEAN_PATH = os.path.join(SAVE_DIR, 'normalization_mean.npy')
STD_PATH = os.path.join(SAVE_DIR, 'normalization_std.npy')

# 2. Paramètres des données
SEQUENCE_LENGTH = 27
NUM_FEATURES = 1662

# 3. Paramètres de reconnaissance
PREDICTION_THRESHOLD = 0.80 
PREDICTION_SMOOTHING = 10   
# --- FIN DE LA CONFIGURATION ---

print("--- Chargement des ressources ---")

# --- CHARGEMENT MODÈLE, ENCODEUR, ET PARAMÈTRES DE NORMALISATION ---
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Fichier modèle non trouvé : {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé avec succès.")

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Fichier LabelEncoder non trouvé : {LABEL_ENCODER_PATH}")
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    actions = label_encoder.classes_
    print(f"LabelEncoder chargé. Actions: {list(actions)}")

    if not os.path.exists(MEAN_PATH) or not os.path.exists(STD_PATH):
        raise FileNotFoundError("Fichiers de normalisation (mean/std) non trouvés.")
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    print("Paramètres de normalisation chargés.")

except FileNotFoundError as fnf_error:
    print(f"\n[ERREUR CRITIQUE] : {fnf_error}")
    print("Veuillez vous assurer que le script 'train_model.py' s'est exécuté avec succès.")
    exit()
except Exception as e:
    print(f"\n[ERREUR CRITIQUE] lors du chargement des ressources : {e}")
    exit()

# --- INITIALISATION MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints_1662(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

# --- BOUCLE PRINCIPALE DE RECONNAISSANCE ---
sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=PREDICTION_SMOOTHING)
sentence = []
current_action_display = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\n[ERREUR CRITIQUE] Impossible d'accéder à la caméra.")
    exit()

print("\n--- Démarrage de la reconnaissance en temps réel ---")
print("Appuyez sur 'q' pour quitter.")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints_1662(results)
            sequence.append(keypoints)
            
            # --- NOUVELLE LOGIQUE : VÉRIFICATION DE LA PRÉSENCE DES MAINS ---
            hands_detected_this_frame = results.left_hand_landmarks or results.right_hand_landmarks

            if len(sequence) == SEQUENCE_LENGTH:
                # La prédiction n'est tentée QUE si des mains sont détectées dans la frame actuelle
                if hands_detected_this_frame:
                    input_data = np.array(sequence)
                    normalized_input = (input_data - mean) / (std + 1e-8)
                    
                    res = model.predict(np.expand_dims(normalized_input, axis=0), verbose=0)[0]
                    
                    predicted_class_index = np.argmax(res)
                    predictions.append(predicted_class_index)
                    confidence = res[predicted_class_index]
                    
                    is_stable = (len(predictions) == PREDICTION_SMOOTHING and 
                                 np.unique(predictions)[0] == predicted_class_index)
                    is_confident = confidence > PREDICTION_THRESHOLD
                    
                    if is_stable and is_confident:
                        current_sign = actions[predicted_class_index]
                        current_action_display = f"{current_sign} ({confidence*100:.1f}%)"
                        
                        if not sentence or sentence[-1] != current_sign:
                            sentence.append(current_sign)
                            sentence = sentence[-5:]
                else:
                    # Si aucune main n'est détectée, on réinitialise l'état de prédiction
                    predictions.clear()
                    current_action_display = "--- Montrez vos mains ---"
            
            # --- AFFICHAGE SUR L'IMAGE ---
            draw_styled_landmarks(frame, results)
            
            cv2.rectangle(frame, (0,0), (frame.shape[1], 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, current_action_display, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Reconnaissance LSF', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        print("\nNettoyage...")
        cap.release()
        cv2.destroyAllWindows()
        print("Programme terminé.")
