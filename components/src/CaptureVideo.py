# -*- coding: utf-8 -*-
# Déclare l'encodage UTF-8 pour le fichier, permettant d'utiliser des caractères accentués dans le code et les commentaires.

import cv2  # Importe la bibliothèque OpenCV pour le traitement d'images et de vidéos.
import mediapipe as mp  # Importe MediaPipe pour la détection de points clés (pose, mains, visage).
import numpy as np  # Importe NumPy pour les opérations numériques, notamment sur les tableaux.
import tensorflow as tf  # Importe TensorFlow pour le chargement et l'utilisation du modèle d'apprentissage profond.
import threading  # Importe le module de threading pour exécuter des tâches en parallèle (capture, extraction).
import queue  # Importe le module de file d'attente pour la communication entre threads.
import os  # Importe le module os pour interagir avec le système d'exploitation (chemins de fichiers, etc.).
import logging  # Importe le module de logging pour enregistrer des messages d'information, d'erreur, etc.
import time  # Importe le module time pour mesurer le temps, gérer les pauses.
import csv  # Importe le module csv pour lire et écrire des fichiers CSV.
from collections import deque, Counter  # Importe deque (file à double extrémité) et Counter (comptage d'éléments).
from tqdm import tqdm  # Importe tqdm pour afficher des barres de progression.
import matplotlib.pyplot as plt # Importe Matplotlib pour la création de graphiques.

# --- Configuration du logging ---
# Configure le système de logging pour afficher les messages à partir du niveau INFO,
# avec un format incluant la date, le niveau, le nom du thread et le message.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

# --- ANSI escape codes for colors ---
# Classe utilitaire pour définir des codes d'échappement ANSI pour colorer le texte dans la console.
class Colors:
    RESET = '\x1b[0m'  # Réinitialise la couleur du texte.
    BRIGHT_YELLOW = '\x1b[93m'  # Jaune vif.
    BRIGHT_GREEN = '\x1b[92m'  # Vert vif.
    RED = '\x1b[31m'  # Rouge.
    GREEN = '\x1b[32m' # Vert (utilisé ici pour le démarrage des threads).
    CV_GREEN = (0, 255, 0)  # Couleur verte pour OpenCV (format BGR).
    CV_YELLOW = (0, 255, 255) # Couleur jaune pour OpenCV (format BGR).
    CV_RED = (0, 0, 255)  # Couleur rouge pour OpenCV (format BGR).
    CV_WHITE = (255, 255, 255) # Couleur blanche pour OpenCV (format BGR).

# --- Constants ---
# !!! IMPORTANT: Assurez-vous que MODEL_PATH pointe vers le modèle
#     entraîné avec les features incluant la bouche !!!
MODEL_PATH = "models/model_basic_mouth.h5"  # Chemin vers le fichier du modèle Keras (.h5). Doit être entraîné AVEC les features de la bouche.
VOCABULARY_FILE = "vocabulaire.txt"  # Chemin vers le fichier texte contenant le vocabulaire (mot:index).
FIXED_LENGTH = 46  # Nombre de frames (séquence temporelle) attendu par le modèle. Doit correspondre à l'entraînement.
VIDEOS_DIR = "D:/bonneaup.SNIRW/Test2/video" # CHEMIN ABSOLU vers le dossier contenant les vidéos à traiter (à adapter).

# --- CONFIGURATION ---
PREDICTION_CSV_FILE = "prediction_log_with_mouth.csv" # Nom du fichier CSV pour enregistrer les journaux de prédiction (mis à jour pour la bouche).
CSV_HEADER = ["Timestamp", "VideoFile", "MostFrequentWord", "Frequency", "TotalPredictions", "AvgConfidenceMostFrequent", "MaxConfidenceSeen", "ProcessingTimeSec"] # En-têtes pour le fichier CSV.
SAVE_KEYPOINTS = True # Booléen pour activer/désactiver la sauvegarde des keypoints extraits pendant la capture.
KEYPOINTS_SAVE_DIR = "extracted_keypoints_capture" # Dossier pour sauvegarder les fichiers .npy des keypoints (différent pour capture vs traitement).
TOP_N = 3  # Nombre de meilleures prédictions à afficher.
SMOOTHING_WINDOW_SIZE = 15  # Taille de la fenêtre de lissage pour la prédiction affichée.
CONF_THRESH_GREEN = 0.80  # Seuil de confiance pour afficher la prédiction en vert.
CONF_THRESH_YELLOW = 0.50 # Seuil de confiance pour afficher la prédiction en jaune (sinon rouge).
FRAMES_TO_SKIP = 3  # Nombre de frames à sauter entre chaque frame traitée (ex: 1 = traiter toutes les frames, 2 = traiter 1 frame sur 2).
MAX_FRAME_WIDTH = 1280  # Largeur maximale des frames pour le redimensionnement (None pour désactiver). Si une frame est plus large, elle sera redimensionnée.
DEADLOCK_TIMEOUT = 10 # Délai en secondes avant de considérer un blocage dans le traitement d'une vidéo.

# --- Paramètres d'Extraction (DOIVENT CORRESPONDRE à traitementVideo.py et entrainement.py) ---
# Ces paramètres doivent être identiques à ceux utilisés lors de l'extraction des features pour l'entraînement du modèle.
NUM_POSE_KEYPOINTS = 4  # Nombre de points clés de la pose à extraire (typiquement nez, épaules, hanche du côté visible).
NUM_HAND_KEYPOINTS = 21 # Nombre de points clés par main (standard MediaPipe).
NUM_COORDS = 3 # Nombre de coordonnées par point clé (x, y, z).

# Indices des points des lèvres (DOIVENT ÊTRE IDENTIQUES à ceux utilisés pour l'entraînement)
# Liste des indices des points clés spécifiques à la bouche, selon MediaPipe Face Mesh.
# Ces indices doivent correspondre exactement à ceux utilisés pour générer les données d'entraînement.
MOUTH_LANDMARK_INDICES = sorted(list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    191, 80, 81, 82, 13, 312, 311, 310, 415
])))
NUM_MOUTH_KEYPOINTS = len(MOUTH_LANDMARK_INDICES) # Nombre total de points clés pour la bouche (calculé).

# ---> MISE À JOUR FEATURES_PER_FRAME (DOIT CORRESPONDRE à l'entraînement) <---
# Calcul du nombre total de features (valeurs numériques) extraites par frame.
# Ce calcul inclut maintenant les points de la bouche.
FEATURES_PER_FRAME = (NUM_POSE_KEYPOINTS * NUM_COORDS) + \
                     (NUM_HAND_KEYPOINTS * NUM_COORDS) * 2 + \
                     (NUM_MOUTH_KEYPOINTS * NUM_COORDS) # <-- AJOUT des features de la bouche

# --- Logging des configurations ---
# Affiche les configurations importantes au démarrage pour faciliter le débogage.
logging.info("--- Configuration CaptureVideo (avec Bouche) ---")
logging.info(f"MODEL_PATH: {MODEL_PATH} (Doit être entraîné avec bouche!)")
logging.info(f"VOCABULARY_FILE: {VOCABULARY_FILE}")
logging.info(f"FIXED_LENGTH: {FIXED_LENGTH}")
logging.info(f"VIDEOS_DIR: {VIDEOS_DIR}")
logging.info(f"PREDICTION_CSV_FILE: {PREDICTION_CSV_FILE}")
logging.info(f"SAVE_KEYPOINTS: {SAVE_KEYPOINTS}")
if SAVE_KEYPOINTS: logging.info(f"KEYPOINTS_SAVE_DIR: {KEYPOINTS_SAVE_DIR}")
logging.info(f"TOP_N Predictions: {TOP_N}")
logging.info(f"SMOOTHING_WINDOW_SIZE: {SMOOTHING_WINDOW_SIZE}")
logging.info(f"FRAMES_TO_SKIP: {FRAMES_TO_SKIP}")
if MAX_FRAME_WIDTH: logging.info(f"MAX_FRAME_WIDTH: {MAX_FRAME_WIDTH}")
else: logging.info("MAX_FRAME_WIDTH: Désactivé")
logging.info(f"DEADLOCK_TIMEOUT: {DEADLOCK_TIMEOUT}s")
logging.info(f"Nombre points Pose: {NUM_POSE_KEYPOINTS}, Main: {NUM_HAND_KEYPOINTS}, Bouche: {NUM_MOUTH_KEYPOINTS}")
logging.info(f"FEATURES_PER_FRAME attendu: {FEATURES_PER_FRAME}") # <-- Affichage mis à jour pour inclure la bouche
logging.info("-----------------------------------------------")


# --- Utility Functions ---
def load_vocabulary(filepath):
    """
    Charge le vocabulaire depuis un fichier texte.
    Le fichier doit contenir une ligne par mot, au format "mot:index".
    Retourne un dictionnaire {mot: index}.
    """
    vocabulaire = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1].isdigit(): # Vérifie le format "mot:nombre"
                    vocabulaire[parts[0]] = int(parts[1])
                elif line.strip(): # Si la ligne n'est pas vide mais incorrecte
                    logging.warning(f"Format ligne incorrect vocabulaire: '{line.strip()}'")
    except (FileNotFoundError, ValueError) as e: # Gère les erreurs de fichier non trouvé ou de valeur incorrecte
        logging.error(f"Erreur chargement vocabulaire {filepath}: {e}")
        return {} # Retourne un dictionnaire vide en cas d'erreur
    logging.info(f"Vocabulaire chargé ({len(vocabulaire)} mots) depuis {filepath}")
    return vocabulaire

def extract_keypoints(results):
    """
    Extrait les points clés de la POSE (4) + MAIN GAUCHE (21) + MAIN DROITE (21) + BOUCHE (NUM_MOUTH_KEYPOINTS).
    Retourne un array numpy de taille FEATURES_PER_FRAME ou un array de zéros si une partie des points est manquante ou en cas d'erreur.
    """
    # Pose (NUM_POSE_KEYPOINTS points)
    # Extrait les x, y, z des NUM_POSE_KEYPOINTS premiers points de la pose.
    # Si non détectés ou moins de points que nécessaire, retourne un tableau de zéros.
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_KEYPOINTS]]).flatten() \
        if results.pose_landmarks and len(results.pose_landmarks.landmark) >= NUM_POSE_KEYPOINTS else np.zeros(NUM_POSE_KEYPOINTS * NUM_COORDS)

    # Mains (NUM_HAND_KEYPOINTS points chacune)
    # Extrait les x, y, z des points de la main gauche.
    # Si non détectée ou nombre de points incorrect, retourne un tableau de zéros.
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks and len(results.left_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)
    # Idem pour la main droite.
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks and len(results.right_hand_landmarks.landmark) == NUM_HAND_KEYPOINTS else np.zeros(NUM_HAND_KEYPOINTS * NUM_COORDS)

    # Bouche (NUM_MOUTH_KEYPOINTS points)
    mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS) # Initialise avec des zéros
    if results.face_landmarks: # Si des points du visage sont détectés
        try:
            # Vérifier si tous les indices de points de la bouche demandés existent dans les landmarks détectés
            if all(idx < len(results.face_landmarks.landmark) for idx in MOUTH_LANDMARK_INDICES):
                 # Extrait les points de la bouche spécifiés par MOUTH_LANDMARK_INDICES.
                 mouth_points = [results.face_landmarks.landmark[i] for i in MOUTH_LANDMARK_INDICES]
                 mouth = np.array([[res.x, res.y, res.z] for res in mouth_points]).flatten()
            else:
                 # Si des indices sont hors limites (manquants).
                 logging.warning(f"Indices de bouche manquants dans face_landmarks (détectés: {len(results.face_landmarks.landmark)}). Retour zéros pour la bouche.")
        except IndexError as ie: # Gère les erreurs d'indice spécifiques.
            logging.error(f"Erreur d'indice lors de l'extraction des points de bouche: {ie}. Retour zéros pour la bouche.")
            mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS)
        except Exception as e: # Gère toute autre erreur inattendue.
            logging.error(f"Erreur inattendue lors de l'extraction des points de bouche: {e}. Retour zéros pour la bouche.")
            mouth = np.zeros(NUM_MOUTH_KEYPOINTS * NUM_COORDS)

    # Concaténation de toutes les features (pose, main gauche, main droite, bouche)
    extracted = np.concatenate([pose, lh, rh, mouth]) # <-- 'mouth' est ajouté ici

    # Vérification finale de la taille totale des features
    # S'assure que le nombre de features extraites correspond à FEATURES_PER_FRAME.
    if extracted.shape[0] != FEATURES_PER_FRAME:
        logging.warning(f"L'extraction a produit un nombre inattendu de features ({extracted.shape[0]}), attendu {FEATURES_PER_FRAME}. Retour de zéros.")
        return np.zeros(FEATURES_PER_FRAME) # Retourner des zéros de la bonne taille en cas d'incohérence.

    return extracted

def get_expected_word_from_filename(filename):
    """
    Extrait le mot attendu (label) à partir du nom de fichier vidéo.
    Suppose un format comme "MOT_details.extension" (ex: "BONJOUR_personne1.mp4" -> "bonjour").
    """
    name_without_ext = os.path.splitext(filename)[0] # Enlève l'extension (ex: .mp4)
    parts = name_without_ext.split('_', 1) # Sépare au premier underscore
    expected_word = parts[0] # Le mot est la première partie
    return expected_word.strip().lower() # Nettoie et met en minuscules.


# --- Keypoint Extractor Class ---
# Classe responsable de la capture des frames vidéo et de l'extraction des keypoints en utilisant MediaPipe.
# Utilise des threads séparés pour la capture et l'extraction pour ne pas bloquer le thread principal.
class KeypointExtractor:
    def __init__(self):
        """Constructeur de la classe KeypointExtractor."""
        self.mp_holistic = mp.solutions.holistic # Raccourci pour la solution Holistic de MediaPipe.
        # Files d'attente pour la communication inter-threads:
        self.frame_queue = queue.Queue(maxsize=50) # Stocke les frames capturées, prêtes pour l'extraction.
        self.keypoint_queue = queue.Queue(maxsize=100) # Stocke les keypoints extraits, prêts pour la prédiction.
        self.display_queue = queue.Queue(maxsize=10) # Stocke les frames originales pour l'affichage.
        self.running = threading.Event() # Événement Threading pour contrôler l'exécution des threads (démarrer/arrêter).
        self.video_capture = None # Objet VideoCapture d'OpenCV.
        self.capture_thread = None # Thread pour la fonction capture_frames.
        self.extraction_thread = None # Thread pour la fonction extract_keypoints_loop.
        self.video_path = None # Chemin de la vidéo en cours de traitement.

    def capture_frames(self):
        """Fonction exécutée par le thread de capture de frames."""
        # Cette fonction lit les frames de la vidéo, les redimensionne si nécessaire,
        # saute des frames selon FRAMES_TO_SKIP, et les place dans frame_queue (pour extraction)
        # et display_queue (pour affichage).
        frame_count_read = 0 # Compteur de frames lues depuis la vidéo.
        frame_count_queued_extract = 0 # Compteur de frames mises en file pour extraction.
        frame_count_queued_display = 0 # Compteur de frames mises en file pour affichage.
        capture_start_time = time.time() # Temps de démarrage de la capture.
        self.video_capture = None # Réinitialise au cas où.

        try:
            logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD CAPTURE pour {os.path.basename(self.video_path)}{Colors.RESET}")
            self.video_capture = cv2.VideoCapture(self.video_path) # Ouvre le fichier vidéo.
            if not self.video_capture.isOpened(): # Vérifie si l'ouverture a réussi.
                logging.error(f"Impossible d'ouvrir vidéo : {self.video_path}")
                raise ValueError(f"Impossible d'ouvrir vidéo : {self.video_path}") # Lève une exception.

            # Récupère les propriétés de la vidéo.
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Vidéo ouverte: {os.path.basename(self.video_path)} ({frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames})")

            # Calcule les dimensions cibles pour le redimensionnement si MAX_FRAME_WIDTH est défini.
            target_width = frame_width; target_height = frame_height; resize_needed = False
            if MAX_FRAME_WIDTH and frame_width > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame_width
                target_width = MAX_FRAME_WIDTH
                target_height = int(frame_height * scale)
                resize_needed = True
                logging.info(f"Redimensionnement activé: {frame_width}x{frame_height} -> {target_width}x{target_height}")

            # Boucle de lecture des frames tant que l'événement 'running' est actif.
            while self.running.is_set():
                ret, frame = self.video_capture.read() # Lit une frame.
                if not ret: # Si fin de la vidéo ou erreur de lecture.
                    logging.info(f"Fin vidéo ou erreur lecture après {frame_count_read} frames lues: {os.path.basename(self.video_path)}.")
                    break # Sort de la boucle.
                frame_count_read += 1

                # Logique pour sauter des frames (FRAMES_TO_SKIP).
                # Traite la frame seulement si son index (basé à 0) est un multiple de FRAMES_TO_SKIP.
                if (frame_count_read - 1) % FRAMES_TO_SKIP == 0:
                    frame_to_process = frame
                    if resize_needed: # Redimensionne si nécessaire.
                         try: frame_to_process = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                         except Exception as resize_err: logging.warning(f"[Capture] Erreur redim. frame {frame_count_read}: {resize_err}"); frame_to_process = frame # Utilise l'originale si redim. échoue

                    if frame_to_process is not None:
                         # Met la frame (potentiellement redimensionnée) dans la file pour l'extraction.
                         try: self.frame_queue.put(frame_to_process, timeout=2.0); frame_count_queued_extract += 1
                         except queue.Full: # Si la file est pleine, attend.
                             logging.warning(f"[Capture] Queue frames (extract) pleine. Attente..."); self.frame_queue.put(frame_to_process); frame_count_queued_extract += 1 # Blocage jusqu'à ce qu'il y ait de la place.
                         except Exception as e: logging.exception(f"[Capture] Erreur frame_queue.put : {e}"); self.running.clear(); break # Arrête en cas d'autre erreur.

                         # Met la frame dans la file pour l'affichage (non bloquant).
                         try: self.display_queue.put_nowait(frame_to_process); frame_count_queued_display += 1
                         except queue.Full: # Si la file d'affichage est pleine, remplace la plus ancienne frame.
                             try: self.display_queue.get_nowait(); self.display_queue.put_nowait(frame_to_process)
                             except queue.Empty: pass # Ne rien faire si vide entre-temps.
                             except Exception as e: logging.warning(f"[Capture] Erreur remplacement display queue: {e}")
                         except Exception as e: logging.warning(f"[Capture] Erreur display_queue.put : {e}")
                    else:
                         logging.warning(f"[Capture] frame_to_process est None (frame #{frame_count_read}), skip.")

            logging.debug(f"[Capture] Sortie de la boucle while pour {os.path.basename(self.video_path)}.")
        except ValueError: pass # Erreur déjà loggée (impossibilité d'ouvrir la vidéo).
        except Exception as e_globale_capture: # Capture toute autre exception.
            logging.exception(f"{Colors.RED}!!! ERREUR GLOBALE CAPTURÉE dans capture_frames pour {os.path.basename(self.video_path)} !!! : {e_globale_capture}{Colors.RESET}")
            self.running.clear() # Arrête les opérations en cas d'erreur majeure.
        finally: # Exécuté dans tous les cas (succès, erreur, fin de vidéo).
            logging.debug(f"{Colors.RED}>>> Entrée FINALLY capture_frames pour {os.path.basename(self.video_path)}{Colors.RESET}")
            if self.video_capture:
                 if self.video_capture.isOpened(): self.video_capture.release(); logging.info(f"Capture vidéo relâchée pour {os.path.basename(self.video_path)}") # Libère la ressource vidéo.
            try: self.frame_queue.put(None, timeout=5.0); logging.info("[Capture - Finally] Signal fin (None) envoyé à frame_queue.") # Envoie un signal (None) pour indiquer la fin de la capture.
            except queue.Full: logging.error("[Capture - Finally] Échec envoi signal fin (None) - Queue frames pleine.")
            except Exception as e: logging.error(f"[Capture - Finally] Erreur envoi signal fin (None) : {e}")
            total_time = time.time() - capture_start_time # Calcule le temps total de capture.
            logging.info(f"Thread capture terminé pour {os.path.basename(self.video_path)}. {frame_count_queued_extract}/{frame_count_read} frames vers extraction, {frame_count_queued_display} vers affichage en {total_time:.2f}s.")
            logging.info(f"{Colors.RED}### FIN THREAD CAPTURE pour {os.path.basename(self.video_path)} ###{Colors.RESET}")


    def extract_keypoints_loop(self):
        """Fonction exécutée par le thread d'extraction de keypoints."""
        # Cette fonction récupère les frames de frame_queue, les traite avec MediaPipe Holistic,
        # extrait les keypoints (y compris la bouche) en utilisant la fonction utilitaire extract_keypoints,
        # et place les keypoints dans keypoint_queue.
        logging.info(f"{Colors.GREEN}>>> DÉMARRAGE THREAD EXTRACTION pour {os.path.basename(self.video_path)}{Colors.RESET}")
        frames_processed = 0 # Compteur de frames traitées par MediaPipe.
        extraction_start_time = time.time() # Temps de démarrage de l'extraction.
        holistic_instance = None # Instance de MediaPipe Holistic.

        try:
            # Initialiser Holistic DANS le thread (important pour la compatibilité de MediaPipe avec les threads).
            holistic_instance = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            logging.info("[Extraction] Instance MediaPipe Holistic créée.")

            while True: # Boucle infinie, contrôlée par self.running et le signal 'None'.
                if not self.running.is_set(): # Si l'arrêt est demandé.
                    logging.info("[Extraction] Arrêt demandé (running=False).")
                    break # Sort de la boucle.

                frame = None
                try:
                    # Récupère une frame de la file d'attente, avec un timeout pour ne pas bloquer indéfiniment.
                    frame = self.frame_queue.get(timeout=1.0)
                    if frame is None: # Si le signal de fin (None) est reçu.
                        logging.info(f"[Extraction] Signal fin (None) reçu de frame_queue. Fin normale.")
                        break # Sort de la boucle.
                except queue.Empty: # Si la file est vide après le timeout.
                    # Vérifie si le thread de capture est toujours actif. Si non, et la file est vide, c'est la fin.
                    if self.capture_thread and not self.capture_thread.is_alive():
                        logging.warning("[Extraction] Queue frames vide et thread capture mort. Arrêt extraction.")
                        break
                    else: continue # Continue d'attendre si la capture est encore active.

                # Traitement avec MediaPipe.
                try:
                    process_start = time.time()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe attend des images RGB.
                    frame_rgb.flags.writeable = False # Optimisation: Marque l'image comme non modifiable.
                    results = holistic_instance.process(frame_rgb) # Traitement par MediaPipe.
                    process_time = time.time() - process_start # Temps de traitement MediaPipe (non utilisé ici).

                    # ---> APPEL À LA FONCTION MISE À JOUR <---
                    # Utilise la fonction extract_keypoints définie plus haut, qui inclut la bouche.
                    keypoints = extract_keypoints(results) # Extrait les keypoints formatés.
                    frames_processed += 1

                    # Met les keypoints (avec bouche) dans la file pour la prédiction.
                    try:
                        self.keypoint_queue.put(keypoints, timeout=2.0)
                    except queue.Full: # Si la file est pleine, attend.
                        logging.warning(f"[Extraction] Queue keypoints pleine. Attente...")
                        self.keypoint_queue.put(keypoints) # Blocage.
                    except Exception as e_put_kp: # Gère les erreurs de mise en file.
                        logging.exception(f"[Extraction] Erreur keypoint_queue.put : {e_put_kp}")
                        self.running.clear(); break # Arrête en cas d'erreur.

                except Exception as e_process: # Gère les erreurs de traitement MediaPipe/extraction.
                    logging.exception(f"[Extraction] Erreur traitement MediaPipe/extraction frame {frames_processed + 1}: {e_process}")
                    # On continue potentiellement avec la frame suivante, mais logge l'erreur.
                    pass

            logging.debug(f"[Extraction] Sortie de la boucle while pour {os.path.basename(self.video_path)}.")
        except Exception as e_init_loop: # Gère les erreurs majeures (ex: initialisation Holistic).
             logging.exception(f"[Extraction] Erreur majeure initialisation ou boucle extraction: {e_init_loop}")
             self.running.clear() # Arrête les opérations.
        finally: # Exécuté dans tous les cas.
             logging.debug(f"{Colors.RED}>>> Entrée FINALLY extract_keypoints_loop pour {os.path.basename(self.video_path)}{Colors.RESET}")
             if holistic_instance: # Ferme l'instance MediaPipe si elle a été créée.
                 try: holistic_instance.close(); logging.debug("[Extraction - Finally] Instance Holistic fermée.")
                 except Exception as e: logging.warning(f"[Extraction - Finally] Erreur fermeture Holistic: {e}")
             try: self.keypoint_queue.put(None, timeout=5.0); logging.info("[Extraction - Finally] Signal fin (None) envoyé à keypoint_queue.") # Envoie un signal de fin à la file de keypoints.
             except queue.Full: logging.error("[Extraction - Finally] Échec envoi signal fin (None) keypoints - Queue pleine.")
             except Exception as e: logging.error(f"[Extraction - Finally] Erreur envoi signal fin (None) keypoints : {e}")
             total_time = time.time() - extraction_start_time # Calcule le temps total d'extraction.
             logging.info(f"Fin boucle extraction pour {os.path.basename(self.video_path)}. Traité {frames_processed} frames en {total_time:.2f}s.")
             logging.info(f"{Colors.RED}### FIN THREAD EXTRACTION pour {os.path.basename(self.video_path)} ###{Colors.RESET}")


    def start(self, video_path):
        """Démarre les threads de capture et d'extraction pour une vidéo donnée."""
        # Vérifie si des threads sont déjà actifs pour une vidéo précédente et les arrête si besoin.
        if self.capture_thread is not None and self.capture_thread.is_alive() or \
           self.extraction_thread is not None and self.extraction_thread.is_alive():
            logging.warning(f"{Colors.BRIGHT_YELLOW}Tentative de démarrer alors que threads actifs pour {os.path.basename(self.video_path)}. Appel stop() d'abord...{Colors.RESET}")
            self.stop() # Arrête les threads existants avant d'en démarrer de nouveaux.

        self.video_path = video_path # Stocke le chemin de la nouvelle vidéo.
        self.running.set() # Active l'événement 'running' pour permettre aux boucles des threads de s'exécuter.

        logging.debug("Vidage queues avant démarrage...")
        # Vide toutes les files d'attente pour s'assurer qu'elles sont propres avant de commencer.
        queues_to_clear = [self.frame_queue, self.keypoint_queue, self.display_queue]
        for q in queues_to_clear:
            while not q.empty():
                try: q.get_nowait() # Retire un élément sans bloquer.
                except queue.Empty: break # Sort si la file est vide.
                except Exception as e_clear: logging.warning(f"Erreur vidage queue {q}: {e_clear}")
        logging.debug("Queues vidées.")

        # Crée les nouveaux threads pour la capture et l'extraction.
        # Le nom du thread inclut le nom de la vidéo pour faciliter le débogage.
        self.capture_thread = threading.Thread(target=self.capture_frames, name=f"Capture-{os.path.basename(video_path)}")
        self.extraction_thread = threading.Thread(target=self.extract_keypoints_loop, name=f"Extract-{os.path.basename(video_path)}")
        # Démarre les threads.
        self.extraction_thread.start()
        self.capture_thread.start()
        logging.info(f"Threads démarrés pour {os.path.basename(video_path)}")

    def stop(self):
        """Arrête proprement les threads de capture et d'extraction."""
        video_name = os.path.basename(self.video_path) if self.video_path else "Unknown Video" # Nom de la vidéo pour les logs.
        logging.info(f"{Colors.BRIGHT_YELLOW}>>> Entrée dans stop() pour {video_name}{Colors.RESET}")
        self.running.clear() # Désactive l'événement 'running', signalant aux threads de s'arrêter.
        logging.info(f"Flag 'running' mis à False pour {video_name}.")

        join_timeout_capture = 10 # Timeout pour l'attente du thread de capture.
        join_timeout_extract = 20 # Timeout pour l'attente du thread d'extraction (plus long car il peut attendre la frame_queue).

        # Arrête et attend le thread de capture.
        if self.capture_thread is not None:
            thread_name = self.capture_thread.name
            if self.capture_thread.is_alive(): # S'il est encore en cours d'exécution.
                logging.info(f"Attente fin thread capture '{thread_name}' (max {join_timeout_capture}s)...")
                self.capture_thread.join(timeout=join_timeout_capture) # Attend sa fin.
                if self.capture_thread.is_alive(): logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread capture '{thread_name}' non terminé.") # Si timeout.
                else: logging.info(f"Thread capture '{thread_name}' terminé.")
            else: logging.debug(f"Thread capture '{thread_name}' déjà terminé.")
        else: logging.debug("stop(): Thread capture non trouvé (None).")

        # Arrête et attend le thread d'extraction.
        if self.extraction_thread is not None:
            thread_name = self.extraction_thread.name
            if self.extraction_thread.is_alive():
                logging.info(f"Attente fin thread extraction '{thread_name}' (max {join_timeout_extract}s)...")
                self.extraction_thread.join(timeout=join_timeout_extract) # Attend sa fin.
                if self.extraction_thread.is_alive(): logging.warning(f"{Colors.RED}TIMEOUT!{Colors.RESET} Thread extraction '{thread_name}' non terminé.") # Si timeout.
                else: logging.info(f"Thread extraction '{thread_name}' terminé.")
            else: logging.debug(f"Thread extraction '{thread_name}' déjà terminé.")
        else: logging.debug("stop(): Thread extraction non trouvé (None).")

        # Réinitialise les références aux threads.
        self.capture_thread = None; self.extraction_thread = None
        logging.info(f"Vérification arrêt threads terminée pour {video_name}.")
        logging.info(f"{Colors.BRIGHT_YELLOW}<<< Sortie de stop() pour {video_name}{Colors.RESET}")


# --- Main Function ---
def main():
    """Fonction principale du script."""
    global SAVE_KEYPOINTS # Déclare que la variable globale SAVE_KEYPOINTS peut être modifiée (si dossier non créé).

    # --- Initialisation TF et Modèle ---
    model = None # Variable pour stocker le modèle chargé.
    try:
        # Configuration du GPU pour TensorFlow (Memory Growth) pour éviter d'allouer toute la mémoire GPU d'un coup.
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configuré(s) pour Memory Growth : {gpus}")
            except RuntimeError as e: logging.error(f"Erreur configuration Memory Growth GPU: {e}")
        else: logging.warning("Aucun GPU détecté par TensorFlow.")

        # Charger le modèle Keras (DOIT ÊTRE ENTRAÎNÉ AVEC LA BOUCHE)
        if not os.path.exists(MODEL_PATH): # Vérifie si le fichier modèle existe.
             logging.error(f"Fichier modèle non trouvé : {MODEL_PATH}"); return # Arrête si non trouvé.
        model = tf.keras.models.load_model(MODEL_PATH) # Charge le modèle.
        logging.info(f"Modèle chargé depuis {MODEL_PATH}")
        try:
            # Vérifier la forme d'entrée attendue par le modèle chargé.
            expected_shape = model.input_shape # (None, FIXED_LENGTH, FEATURES_PER_FRAME)
            logging.info(f"Forme entrée attendue par le modèle chargé: {expected_shape}")
            # Valider que la forme du modèle (nombre de frames, nombre de features par frame)
            # correspond aux constantes définies dans le script (FIXED_LENGTH, FEATURES_PER_FRAME).
            # C'est crucial pour éviter les erreurs d'exécution avec le modèle.
            if len(expected_shape) != 3 or \
               expected_shape[1] is not None and expected_shape[1] != FIXED_LENGTH or \
               expected_shape[2] is not None and expected_shape[2] != FEATURES_PER_FRAME:
                 logging.warning(
                     f"{Colors.RED}Incohérence Shape! Modèle attend (batch, {expected_shape[1]}, {expected_shape[2]}), "
                     f"Script Config (batch, {FIXED_LENGTH}, {FEATURES_PER_FRAME}){Colors.RESET}"
                 )
                 # Optionnel: Sortir si l'incohérence est critique, par exemple: return
            else:
                 logging.info("La forme d'entrée du modèle chargé correspond aux constantes du script.")
        except Exception as e:
             logging.warning(f"Impossible de vérifier/valider la forme d'entrée du modèle: {e}")
    except Exception as e: # Gère les erreurs de chargement du modèle.
        logging.exception(f"Erreur majeure lors de l'initialisation TensorFlow/Modèle : {e}"); return # Arrête.

    # --- Chargement Vocabulaire ---
    vocabulaire = load_vocabulary(VOCABULARY_FILE) # Charge le vocabulaire (mot -> index).
    if not vocabulaire: logging.error("Erreur critique : Vocabulaire vide/non chargé. Arrêt."); return # Arrête si échec.
    index_to_word = {i: word for word, i in vocabulaire.items()} # Crée un mapping inverse (index -> mot).

    # --- Vérification Dossier Vidéos et Listing ---
    if not os.path.isdir(VIDEOS_DIR): logging.error(f"Chemin vidéo invalide: {VIDEOS_DIR}"); return # Vérifie si le dossier des vidéos existe.
    try:
        # Liste tous les fichiers vidéo dans le dossier spécifié.
        video_files_to_process = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        video_files_to_process.sort() # Trie les fichiers pour un traitement ordonné.
        logging.debug(f"Fichiers vidéo trouvés: {video_files_to_process}")
    except Exception as e: logging.exception(f"Erreur listage fichiers dans {VIDEOS_DIR}: {e}"); return # Gère les erreurs de listage.
    if not video_files_to_process: logging.info(f"Aucune vidéo trouvée dans {VIDEOS_DIR}."); return # Arrête si aucune vidéo.
    logging.info(f"Trouvé {len(video_files_to_process)} vidéos à traiter.")

    # --- Préparation Sauvegarde Keypoints et CSV ---
    if SAVE_KEYPOINTS: # Si la sauvegarde des keypoints est activée.
        try:
            os.makedirs(KEYPOINTS_SAVE_DIR, exist_ok=True) # Crée le dossier de sauvegarde (s'il n'existe pas).
            logging.info(f"Sauvegarde keypoints activée -> Dossier: '{KEYPOINTS_SAVE_DIR}'")
        except OSError as e: # Gère les erreurs de création de dossier.
            logging.error(f"Impossible créer dossier keypoints '{KEYPOINTS_SAVE_DIR}': {e}")
            SAVE_KEYPOINTS = False # Désactive la sauvegarde si le dossier ne peut être créé.
            logging.warning("Sauvegarde keypoints désactivée.")

    try:
        # Prépare le fichier CSV pour l'enregistrement des résultats.
        # Vérifie si le fichier existe déjà et s'il est vide pour décider d'écrire l'en-tête.
        file_exists = os.path.isfile(PREDICTION_CSV_FILE)
        write_header = not file_exists or os.path.getsize(PREDICTION_CSV_FILE) == 0
        with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile: # Ouvre en mode 'append'.
            writer = csv.writer(csvfile)
            if write_header: writer.writerow(CSV_HEADER); logging.info(f"CSV '{PREDICTION_CSV_FILE}' prêt (en-tête écrit).")
            else: logging.info(f"CSV '{PREDICTION_CSV_FILE}' prêt (ajout données).")
    except IOError as e: # Gère les erreurs d'ouverture/écriture du CSV.
        logging.error(f"Impossible ouvrir/écrire en-tête CSV dans {PREDICTION_CSV_FILE}: {e}")
        # return # Optionnel: sortir si le CSV est critique pour l'application.

    main_start_time = time.time() # Temps de démarrage du traitement principal.
    extractor = None # Instance de KeypointExtractor, initialisée par vidéo.
    word_counts = {}; correct_predictions = {} # Dictionnaires pour le suivi de la précision.

    try:
        # --- Boucle Principale sur les Vidéos ---
        # Traite chaque vidéo trouvée, avec une barre de progression tqdm.
        for video_index, video_file in enumerate(tqdm(video_files_to_process, desc="Traitement Vidéos", unit="video")):
            video_path = os.path.join(VIDEOS_DIR, video_file) # Chemin complet de la vidéo.
            base_video_name = os.path.basename(video_path) # Nom de base du fichier vidéo.
            window_name = f"Video - {base_video_name}" # Nom de la fenêtre d'affichage OpenCV.
            logging.info(f"{Colors.BRIGHT_YELLOW}--- [{video_index+1}/{len(video_files_to_process)}] Début: {base_video_name} ---{Colors.RESET}")
            video_start_time = time.time() # Temps de démarrage du traitement de cette vidéo.

            extractor = KeypointExtractor() # Crée une nouvelle instance pour chaque vidéo.
            extractor.start(video_path) # Démarre les threads de capture et d'extraction.

            # Structures de données pour la vidéo courante
            sequence_window = deque(maxlen=FIXED_LENGTH) # Fenêtre glissante pour stocker la séquence de keypoints pour le modèle.
            all_keypoints_for_video = [] # Liste pour stocker tous les keypoints de la vidéo (si SAVE_KEYPOINTS).
            all_predictions_details = [] # Liste pour stocker les détails de chaque prédiction (index, confiance).
            prediction_display_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE) # Buffer pour lisser la prédiction affichée.
            frame_display_buffer = None # Tampon pour la dernière frame à afficher.
            processing_active = True # Booléen pour contrôler la boucle de traitement de la vidéo.
            frames_processed_main = 0; predictions_made = 0 # Compteurs.
            max_confidence_seen_video = 0.0 # Confiance maximale vue pour cette vidéo.
            last_top_n_text = ["Initialisation..."] # Texte à afficher pour les prédictions.
            last_keypoint_time = time.time(); deadlock_timeout_occurred = False # Pour la détection de deadlock.

            try:
                # --- Boucle Interne: Traitement Keypoints & Affichage ---
                # Boucle tant que le traitement est actif pour la vidéo courante.
                while processing_active:
                    keypoints = None
                    try:
                        # Récupérer les keypoints (qui incluent maintenant la bouche) depuis la queue de l'extracteur.
                        keypoints = extractor.keypoint_queue.get(timeout=0.5) # Timeout pour ne pas bloquer indéfiniment.
                        if keypoints is None: # Si signal de fin (None) reçu.
                            logging.info(f"[Main] Signal fin (None) reçu keypoint_queue. Fin {base_video_name}.")
                            processing_active = False # Arrête la boucle pour cette vidéo.
                        else:
                            frames_processed_main += 1
                            last_keypoint_time = time.time() # Met à jour le temps du dernier keypoint reçu (pour deadlock).
                            if SAVE_KEYPOINTS:
                                all_keypoints_for_video.append(keypoints) # Ajoute les keypoints à la liste de sauvegarde.
                            # Ajouter le vecteur de keypoints (maintenant plus long avec la bouche) à la fenêtre glissante.
                            sequence_window.append(keypoints)

                    except queue.Empty: # Si la queue de keypoints est vide après le timeout.
                        # Logique de détection de fin de traitement ou de deadlock.
                        capture_alive = extractor.capture_thread and extractor.capture_thread.is_alive()
                        extract_alive = extractor.extraction_thread and extractor.extraction_thread.is_alive()
                        # Si l'extracteur est arrêté, la queue est vide, et les threads sont morts -> fin normale.
                        if not extractor.running.is_set() and extractor.keypoint_queue.empty() and not capture_alive and not extract_alive:
                            logging.info(f"[Main] Arrêt/threads terminés et queue vide. Fin {base_video_name}.")
                            processing_active = False
                        # Si capture morte mais extraction vivante et pas de keypoints depuis un certain temps -> deadlock potentiel.
                        elif not capture_alive and extract_alive:
                            time_since_last = time.time() - last_keypoint_time
                            if time_since_last > DEADLOCK_TIMEOUT:
                                logging.error(f"{Colors.RED}[Main] DEADLOCK TIMEOUT ({DEADLOCK_TIMEOUT}s) détecté pour {base_video_name}! Forçage arrêt.{Colors.RESET}")
                                deadlock_timeout_occurred = True # Marque qu'un deadlock est survenu.
                                processing_active = False # Arrête le traitement de cette vidéo.
                        elif extractor.running.is_set() or capture_alive or extract_alive:
                             pass # Continue d'attendre si les threads sont encore actifs ou censés l'être.
                        else: # Si ni running, ni threads vivants, et la queue est vide.
                             if not extractor.keypoint_queue.empty(): logging.info("[Main] Threads morts mais queue pas vide? Tentative vidage...")
                             else: logging.info("[Main] Threads morts et queue vide. Arrêt."); processing_active = False

                    # --- Logique de Prédiction (utilise les keypoints avec bouche) ---
                    if keypoints is not None and not deadlock_timeout_occurred: # Si des keypoints sont disponibles et pas de deadlock.
                        current_sequence_len = len(sequence_window) # Longueur actuelle de la séquence.
                        padded_sequence = None

                        if current_sequence_len > 0:
                            if current_sequence_len < FIXED_LENGTH: # Si la séquence est plus courte que la longueur fixe.
                                # ---> Padding utilise le nouveau FEATURES_PER_FRAME <---
                                # Ajoute du padding (zéros) au début pour atteindre FIXED_LENGTH.
                                # La taille du padding doit correspondre à (frames_manquantes, FEATURES_PER_FRAME).
                                padding = np.zeros((FIXED_LENGTH - current_sequence_len, FEATURES_PER_FRAME))
                                padded_sequence = np.concatenate((padding, np.array(sequence_window)), axis=0)
                            else: # Si la séquence a atteint FIXED_LENGTH.
                                padded_sequence = np.array(sequence_window)

                            # ---> Vérification shape utilise le nouveau FEATURES_PER_FRAME <---
                            # S'assure que la séquence a la bonne forme (FIXED_LENGTH, FEATURES_PER_FRAME) avant la prédiction.
                            if padded_sequence is not None and padded_sequence.shape == (FIXED_LENGTH, FEATURES_PER_FRAME):
                                reshaped_sequence = np.expand_dims(padded_sequence, axis=0) # Ajoute une dimension batch.
                                try:
                                    predict_start = time.time()
                                    # Le modèle chargé DOIT avoir été entraîné avec cette shape d'entrée (incluant la bouche).
                                    res = model.predict(reshaped_sequence, verbose=0)[0] # Prédiction. verbose=0 pour moins de logs TF.
                                    predict_time = time.time() - predict_start # Temps de prédiction (non utilisé).
                                    predictions_made += 1

                                    # Analyse des TOP_N meilleures prédictions.
                                    top_n_indices = np.argsort(res)[-TOP_N:][::-1] # Indices des N meilleures prédictions.
                                    top_n_confidences = res[top_n_indices] # Confiances associées.
                                    top_n_words = [index_to_word.get(idx, f"Idx_{idx}?") for idx in top_n_indices] # Mots correspondants.
                                    top_pred_idx = top_n_indices[0]; top_pred_conf = top_n_confidences[0] # Meilleure prédiction.
                                    all_predictions_details.append((top_pred_idx, top_pred_conf)) # Stocke pour analyse finale.
                                    prediction_display_buffer.append(top_pred_idx) # Ajoute au buffer de lissage.
                                    max_confidence_seen_video = max(max_confidence_seen_video, top_pred_conf) # Met à jour la confiance max.
                                    # Prépare le texte pour l'affichage.
                                    last_top_n_text = [f"{w} ({c:.2f})" for w, c in zip(top_n_words, top_n_confidences)]

                                except tf.errors.InvalidArgumentError as e_tf_shape:
                                     # Cette erreur survient typiquement si la forme des données d'entrée (script)
                                     # ne correspond pas à la forme attendue par le modèle chargé.
                                     logging.error(f"[Main] {Colors.RED}Erreur TensorFlow (mauvaise shape modèle?): {e_tf_shape}. Shape fournie: {reshaped_sequence.shape}. Modèle attendait probablement autre chose.{Colors.RESET}")
                                     last_top_n_text = ["Erreur Shape Modèle?"]
                                     processing_active = False # Arrêter si le modèle est incompatible.
                                except Exception as e_pred: # Gère toute autre erreur de prédiction.
                                    logging.exception(f"[Main] Erreur inconnue model.predict: {e_pred}")
                                    last_top_n_text = ["Erreur Prediction"]
                            else:
                                # Erreur si la séquence n'a pas la forme attendue (problème de logique de padding/fenêtrage).
                                logging.warning(f"[Main] Shape incorrecte ({padded_sequence.shape if padded_sequence is not None else 'None'}) avant prédiction. Attendu: ({FIXED_LENGTH}, {FEATURES_PER_FRAME})")
                                last_top_n_text = ["Erreur Seq Shape"]

                    # --- Affichage (inchangé par rapport à la version sans bouche) ---
                    try:
                        # Récupère une nouvelle frame depuis la queue d'affichage (non bloquant).
                        new_frame = extractor.display_queue.get_nowait()
                        if new_frame is not None: frame_display_buffer = new_frame # Met à jour le buffer d'affichage.
                    except queue.Empty: pass # Ne rien faire si la queue est vide.

                    if frame_display_buffer is not None and frame_display_buffer.size > 0: # Si une frame est disponible.
                        display_frame = frame_display_buffer.copy() # Copie pour ne pas modifier l'original.
                        try:
                            # Détermine la couleur du texte de la meilleure prédiction en fonction de sa confiance.
                            top_conf = all_predictions_details[-1][1] if all_predictions_details else 0.0
                            text_color = Colors.CV_RED
                            if top_conf >= CONF_THRESH_GREEN: text_color = Colors.CV_GREEN
                            elif top_conf >= CONF_THRESH_YELLOW: text_color = Colors.CV_YELLOW

                            # Affiche les TOP_N prédictions.
                            y_offset = 30
                            for line_idx, line in enumerate(last_top_n_text):
                                current_color = text_color if line_idx == 0 else Colors.CV_WHITE # Seule la 1ère ligne est colorée par confiance.
                                cv2.putText(display_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2, cv2.LINE_AA)
                                y_offset += 25

                            # Affiche la prédiction lissée (mot le plus fréquent dans le buffer de lissage).
                            if prediction_display_buffer:
                                try:
                                    smoothed_index = Counter(prediction_display_buffer).most_common(1)[0][0]
                                    smoothed_word = index_to_word.get(smoothed_index, "?")
                                    cv2.putText(display_frame, f"Lisse ({SMOOTHING_WINDOW_SIZE}f): {smoothed_word}", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.CV_WHITE, 2, cv2.LINE_AA)
                                except IndexError: pass # Si buffer vide.

                            cv2.imshow(window_name, display_frame) # Affiche la frame.
                            key = cv2.waitKey(1) & 0xFF # Attend une touche (1ms).
                            if key == ord('q'): # Si 'q' est pressée, arrêt global.
                                logging.info("Touche 'q' pressée, arrêt global.")
                                processing_active = False
                                extractor.running.clear() # Arrête les threads de l'extracteur.
                                raise KeyboardInterrupt("Arrêt utilisateur via 'q'") # Lève une exception pour sortir proprement.

                        except cv2.error as e_cv: # Gère les erreurs OpenCV (ex: fenêtre fermée).
                            if "NULL window" in str(e_cv) or "Invalid window handle" in str(e_cv):
                                logging.warning(f"[Main] Fenêtre '{window_name}' fermée? Arrêt vidéo.")
                                processing_active = False # Arrête le traitement de cette vidéo.
                            else: logging.warning(f"[Main] Erreur cv2: {e_cv}")
                        except Exception as e_show: # Gère toute autre erreur d'affichage.
                             logging.exception(f"[Main] Erreur affichage/texte: {e_show}")

                    if not processing_active: # Si le traitement doit s'arrêter pour cette vidéo.
                         logging.debug(f"[Main] processing_active=False, sortie boucle vidéo {base_video_name}.")
                         break # Sort de la boucle interne.
                # --- Fin boucle interne (traitement de la vidéo courante) ---
                logging.info(f"Fin boucle traitement principale pour {base_video_name}.")

            except KeyboardInterrupt: # Si Ctrl+C ou 'q' a été pressé.
                 logging.info(f"KeyboardInterrupt pendant {base_video_name}. Arrêt...")
                 if extractor: extractor.running.clear() # S'assure que les threads de l'extracteur sont signalés.
                 raise # Propage l'exception pour arrêter le script.
            except Exception as e_inner_loop: # Gère toute autre erreur inattendue dans la boucle interne.
                logging.exception(f"Erreur inattendue boucle interne {base_video_name}: {e_inner_loop}")
                if extractor: extractor.running.clear() # Signale l'arrêt à l'extracteur.
            finally:
                # --- Nettoyage après chaque vidéo ---
                logging.info(f"{Colors.BRIGHT_YELLOW}--- Nettoyage pour {base_video_name}... ---{Colors.RESET}")
                if extractor: extractor.stop() # Arrête proprement les threads de l'extracteur.
                try:
                     # Ferme la fenêtre OpenCV si elle est encore visible.
                     if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                         cv2.destroyWindow(window_name); logging.info(f"Fenêtre '{window_name}' fermée.")
                     cv2.waitKey(1) # Petit délai pour s'assurer que la fenêtre est bien fermée.
                except Exception as e_close: # Gère les erreurs de fermeture de fenêtre.
                    logging.warning(f"Erreur fermeture fenêtre '{window_name}': {e_close}"); cv2.waitKey(1)

                video_end_time = time.time(); processing_time_sec = video_end_time - video_start_time # Calcule le temps de traitement.
                logging.info(f"Vidéo {base_video_name}: {frames_processed_main} keypoints traités, {predictions_made} prédictions.")
                logging.info(f"Temps traitement vidéo: {processing_time_sec:.2f} sec.")

                # --- Sauvegarde Keypoints (.npy) si activée ---
                if SAVE_KEYPOINTS and all_keypoints_for_video: # Si la sauvegarde est active et des keypoints ont été collectés.
                    # Nom du fichier .npy, avec un suffixe "_capture" pour distinguer des keypoints de traitement offline.
                    npy_filename = os.path.splitext(base_video_name)[0] + "_capture.npy"
                    npy_filepath = os.path.join(KEYPOINTS_SAVE_DIR, npy_filename)
                    try:
                        np.save(npy_filepath, np.array(all_keypoints_for_video)) # Sauvegarde en format NumPy.
                        logging.info(f"Keypoints sauvegardés: {npy_filepath} ({len(all_keypoints_for_video)} frames)")
                    except Exception as e_save: # Gère les erreurs de sauvegarde.
                        logging.error(f"Erreur sauvegarde keypoints {npy_filepath}: {e_save}")

                # --- Analyse Prédictions & Log CSV & Suivi Précision ---
                # (Cette logique est largement inchangée par rapport à une version sans bouche,
                #  elle opère sur les résultats des prédictions.)
                final_word = "N/A"; final_word_freq = 0; avg_conf_final_word = 0.0; is_correct = False
                expected_word = get_expected_word_from_filename(base_video_name) # Extrait le mot attendu du nom de fichier.
                if not expected_word: logging.warning(f"Mot attendu non extrait de '{base_video_name}'")
                else: word_counts[expected_word] = word_counts.get(expected_word, 0) + 1 # Compte les occurrences de chaque mot attendu.

                if deadlock_timeout_occurred: # Si un deadlock a été détecté pour cette vidéo.
                     final_word = "TIMEOUT_DEADLOCK"; final_word_freq = 0; avg_conf_final_word = 0.0
                     logging.warning(f"Enregistrement CSV: {Colors.RED}{final_word}{Colors.RESET} pour {base_video_name}.")
                elif all_predictions_details: # Si des prédictions ont été faites.
                    try:
                        prediction_indices = [idx for idx, conf in all_predictions_details] # Liste des indices prédits.
                        if prediction_indices:
                            # Trouve le mot le plus fréquent et sa fréquence.
                            index_counts = Counter(prediction_indices)
                            most_common_index, final_word_freq = index_counts.most_common(1)[0]
                            final_word = index_to_word.get(most_common_index, f"Idx_{most_common_index}?") # Mot final.
                            # Calcule la confiance moyenne pour ce mot final.
                            confidences_for_final_word = [conf for idx, conf in all_predictions_details if idx == most_common_index]
                            if confidences_for_final_word: avg_conf_final_word = sum(confidences_for_final_word) / len(confidences_for_final_word)

                            # Vérifie si la prédiction est correcte.
                            if expected_word and final_word.lower() == expected_word:
                                is_correct = True
                                correct_predictions[expected_word] = correct_predictions.get(expected_word, 0) + 1 # Compte les prédictions correctes.
                                logging.info(f"-> Mot final: {Colors.BRIGHT_GREEN}{final_word}{Colors.RESET} ({final_word_freq}/{predictions_made}, conf avg: {avg_conf_final_word:.2f}) - CORRECT")
                            else:
                                logger_func = logging.info if expected_word else logging.warning # Log en warning si pas de mot attendu.
                                logger_func(f"-> Mot final: {Colors.RED}{final_word}{Colors.RESET} ({final_word_freq}/{predictions_made}, conf avg: {avg_conf_final_word:.2f}) - INCORRECT (Attendu: {expected_word if expected_word else 'N/A'})")
                        else: final_word = "Erreur_Analyse_Indices" # Si pas d'indices de prédiction.
                    except Exception as e_analyze: # Gère les erreurs d'analyse.
                        logging.exception(f"Erreur analyse finale prédictions {base_video_name}: {e_analyze}")
                        final_word = "Erreur_Analyse_Exception"
                else: # Si aucune prédiction n'a été générée (et pas de deadlock).
                    logging.warning(f"{Colors.RED}-> Aucune prédiction générée pour {base_video_name} (pas de deadlock).{Colors.RESET}")

                # Écrit les résultats dans le fichier CSV.
                try:
                    current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S") # Horodatage actuel.
                    with open(PREDICTION_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([current_timestamp, base_video_name, final_word, final_word_freq, predictions_made,
                                         f"{avg_conf_final_word:.4f}", f"{max_confidence_seen_video:.4f}", f"{processing_time_sec:.2f}"])
                    logging.info(f"Résultat '{final_word}' ajouté à {PREDICTION_CSV_FILE}")
                except IOError as e_io_csv: logging.error(f"Impossible écrire CSV {PREDICTION_CSV_FILE}: {e_io_csv}")
                except Exception as e_csv: logging.exception(f"Erreur écriture CSV: {e_csv}")

                logging.info(f"{Colors.BRIGHT_YELLOW}--- Fin traitement complet {base_video_name} ---{Colors.RESET}")

        # --- Fin boucle principale vidéos ---
        total_main_time = time.time() - main_start_time # Temps total de traitement de toutes les vidéos.
        logging.info(f"{Colors.BRIGHT_GREEN}=== Traitement des {len(video_files_to_process)} vidéos terminé en {total_main_time:.2f} secondes. ===")

        # --- Calcul & Affichage Précision Globale & par Mot ---
        # (Logique inchangée, basée sur word_counts et correct_predictions)
        total_videos_processed_for_accuracy = sum(word_counts.values())
        total_correct_overall = sum(correct_predictions.values())
        if total_videos_processed_for_accuracy > 0:
            overall_accuracy = (total_correct_overall / total_videos_processed_for_accuracy) * 100
            logging.info(f"=== Précision Globale: {total_correct_overall}/{total_videos_processed_for_accuracy} ({overall_accuracy:.2f}%) ===")
        else: logging.info("=== Aucune vidéo traitable pour calculer précision globale. ===")

        word_accuracies = {} # Dictionnaire pour stocker la précision de chaque mot.
        logging.info("--- Précision par Mot ---")
        sorted_expected_words = sorted(word_counts.keys()) # Mots attendus triés.
        if not sorted_expected_words: logging.info("Aucun mot attendu extrait.")
        else:
            for word in sorted_expected_words:
                total = word_counts[word] # Nombre total d'apparitions du mot.
                correct = correct_predictions.get(word, 0) # Nombre de fois où il a été correctement prédit.
                accuracy = (correct / total) * 100 if total > 0 else 0 # Précision pour ce mot.
                word_accuracies[word] = accuracy
                logging.info(f"- {word}: {correct}/{total} ({accuracy:.2f}%)")
        logging.info("------------------------")

        # --- Génération Graphique (inchangé) ---
        # Si des données de précision par mot existent, génère un graphique à barres.
        if word_accuracies:
            try:
                words = list(word_accuracies.keys()); accuracies = list(word_accuracies.values())
                plt.figure(figsize=(max(10, len(words) * 0.8), 6)) # Taille de la figure adaptable.
                bars = plt.bar(words, accuracies, color='skyblue') # Crée les barres.
                plt.xlabel("Mot Attendu (normalisé)"); plt.ylabel("Précision (%)"); plt.title("Précision par Mot")
                plt.ylim(0, 105); plt.xticks(rotation=45, ha='right') # Mise en forme des axes.
                # Ajoute le pourcentage au-dessus de chaque barre.
                for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center', fontsize=9)
                plt.tight_layout() # Ajuste la mise en page.
                graph_filename = "prediction_accuracy_per_word_capture.png" # Nom du fichier graphique.
                plt.savefig(graph_filename); logging.info(f"Graphique précision sauvegardé: '{graph_filename}'")
                # plt.show() # Décommenter pour afficher le graphique interactivement.
            except Exception as e_plot: logging.error(f"Erreur génération graphique: {e_plot}")
        else: logging.info("Aucune donnée pour générer graphique.")

    except KeyboardInterrupt: # Gère l'arrêt par Ctrl+C (ou 'q').
         logging.info(f"{Colors.RED}Arrêt programme (KeyboardInterrupt).{Colors.RESET}")
         if extractor is not None: logging.info("Arrêt propre threads..."); extractor.stop() # Arrête l'extracteur si actif.
    except Exception as e_main_loop: # Gère toute autre exception non gérée dans la boucle principale.
         logging.exception(f"{Colors.RED}Erreur non gérée boucle principale: {e_main_loop}{Colors.RESET}")
         if extractor is not None: logging.info("Arrêt propre threads après erreur..."); extractor.stop() # Arrête l'extracteur.
    finally:
         # --- Nettoyage final global ---
         logging.info("Nettoyage final: Fermeture fenêtres OpenCV...")
         cv2.destroyAllWindows(); cv2.waitKey(1); cv2.waitKey(1) # Ferme toutes les fenêtres OpenCV.
         if 'model' in locals() and model is not None: # Si le modèle a été chargé.
             try:
                 logging.debug("Libération session Keras globale...");
                 tf.keras.backend.clear_session() # Libère la session Keras/TensorFlow pour libérer la mémoire.
                 del model # Supprime la référence au modèle.
                 logging.debug("Session Keras libérée.")
             except Exception as e: logging.warning(f"Erreur nettoyage final Keras: {e}")
         time.sleep(0.5) # Petite pause pour permettre aux logs de s'afficher.
         logging.info(f"{Colors.BRIGHT_GREEN}Programme terminé.{Colors.RESET}")

# Point d'entrée principal du script.
if __name__ == "__main__":
    try:
        main() # Appelle la fonction principale.
    except Exception as e: # Capture toute exception non gérée qui pourrait survenir avant ou après main().
        logging.exception(f"{Colors.RED}Erreur non gérée au niveau __main__: {e}{Colors.RESET}")
    finally: # S'assure que les fenêtres OpenCV sont fermées même en cas d'erreur très précoce.
        cv2.destroyAllWindows(); cv2.waitKey(1)
        logging.info("Sortie finale script.")
