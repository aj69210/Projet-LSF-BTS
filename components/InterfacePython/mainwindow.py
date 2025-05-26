# -*- coding: utf-8 -*-
# mainwindow.py

# --- PyQt Imports ---
import sys
print("DEBUG: Importing sys")
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale, # type: ignore
                            QMetaObject, QObject, QPoint, QPointF, QRect,
                            QSize, QTime, QUrl, Qt, Signal, QThread, Slot, QTimer)
print("DEBUG: Imported QtCore")
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, # type: ignore
                         QFont, QFontDatabase, QGradient, QIcon,
                         QImage, QKeySequence, QLinearGradient, QPainter,
                         QPalette, QPixmap, QRadialGradient, QTransform, QTextCursor)
print("DEBUG: Imported QtGui")
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QFrame,  # type: ignore
                             QGridLayout, QHBoxLayout, QLabel, QLayout,
                             QMainWindow, QMenuBar, QPushButton, QSizePolicy,
                             QSpacerItem, QStatusBar, QTextEdit, QVBoxLayout,
                             QWidget, QMessageBox, QButtonGroup)
print("DEBUG: Imported QtWidgets")

# --- Imports pour le traitement vidéo ---
print("DEBUG: Importing OpenCV...")
try:
    import cv2
    print("DEBUG: OpenCV imported successfully.")
except ImportError:
    print("ERREUR: Le module 'cv2' (OpenCV) n'est pas installé.")
    print("Veuillez l'installer avec : pip install opencv-python")
    sys.exit(1)

print("DEBUG: Importing Mediapipe...")
try:
    import mediapipe as mp
    print("DEBUG: Mediapipe imported successfully.")
except ImportError:
    print("ERREUR: Le module 'mediapipe' n'est pas installé.")
    print("Veuillez l'installer avec : pip install mediapipe")
    mp = None

print("DEBUG: Importing NumPy and TensorFlow...")
import numpy as np
import tensorflow as tf
print(f"DEBUG: TensorFlow imported. Version: {tf.__version__}")
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
print("DEBUG: Keras imports complete.")

import os
import logging
import time
from collections import deque, Counter
# import traceback

print("DEBUG: Standard library imports complete.")

# --- Import Config (pour la partie vidéo) ---
print("DEBUG: Importing config...")
try:
    import config # Assurez-vous que config.py est dans le même dossier ou PYTHONPATH
    print("DEBUG: config imported successfully.")
except ImportError:
    print("ERREUR: Impossible d'importer config.py. Assurez-vous qu'il existe et qu'il est accessible.")
    sys.exit(1)
except Exception as e:
    print(f"ERREUR: Problème lors de l'import de config.py: {e}")
    sys.exit(1)

# --- Imports pour le Text-To-Speech (TTS) ---
print("DEBUG: Importing TTS modules...")
import requests
import base64
import re
import queue # Python's standard queue
import threading
import tempfile
from json import load as json_load, dump as json_dump # Renamed to avoid conflict
from typing import Dict, List, Optional, Tuple # No, Tuple not used here. Kept for consistency with original.
from enum import Enum

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    print("DEBUG: gTTS imported successfully.")
except ImportError:
    print("AVERTISSEMENT: Le module 'gTTS' n'est pas installé. La voix féminine française TTS sera désactivée.")
    print("Veuillez l'installer avec : pip install gTTS")
    GTTS_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    PYDUB_AVAILABLE = True
    print("DEBUG: Pydub imported successfully.")
except ImportError:
    print("AVERTISSEMENT: Le module 'pydub' n'est pas installé. La lecture audio TTS sera désactivée.")
    print("Veuillez l'installer avec : pip install pydub")
    PYDUB_AVAILABLE = False
print("DEBUG: TTS module imports complete.")


# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
print("DEBUG: Logging configured.")


# --- Définitions TTS (intégrées depuis custom_tts.py) ---
class Voice(Enum):
    FR_MALE = "fr_002"    # TikTok API voice ID
    EN_MALE = "en_us_006"  # TikTok API voice ID
    EN_FEMALE = "en_us_001" # TikTok API voice ID
    # Add more TikTok API voices here if known, e.g.:
    # FR_FEMALE_TIKTOK = "fr_001"

api_voice_keys: Dict[str, Voice] = {v.name.lower(): v for v in Voice}
# "fr_female" is a special key for local gTTS generation
valid_voice_keys: List[str] = list(api_voice_keys.keys()) + ["fr_female"]

TTS_PYDUB_OK = PYDUB_AVAILABLE # Global flag for TTS playback readiness
TTS_GTTS_OK = GTTS_AVAILABLE   # Global flag for gTTS French Female voice

def _load_endpoints() -> List[Dict[str, str]]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, 'data', 'config.json')
    if not os.path.exists(json_file_path):
        print(f"DEBUG (TTS): Fichier config.json non trouvé à '{json_file_path}'. L'API TTS ne fonctionnera pas sans endpoints.")
        return []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json_load(file)
    except Exception as e:
        print(f"ERREUR (TTS): Erreur de chargement des endpoints depuis {json_file_path}: {e}")
        return []

def _split_text(text: str) -> List[str]:
    merged_chunks: List[str] = []
    separated_chunks: List[str] = re.findall(r'.*?[.,!?:;-]|.+', text)
    character_limit: int = 300 # TikTok API limit
    processed_chunks: List[str] = []
    for chunk in separated_chunks:
        if len(chunk.encode("utf-8")) > character_limit:
             sub_chunks = re.findall(r'.*?[ ]|.+', chunk)
             processed_chunks.extend(sub_chunks)
        else:
            processed_chunks.append(chunk)
    current_chunk: str = ""
    for separated_chunk in processed_chunks:
        if len((current_chunk + separated_chunk).encode("utf-8")) <= character_limit:
            current_chunk += separated_chunk
        else:
            if current_chunk:
                 merged_chunks.append(current_chunk)
            if len(separated_chunk.encode("utf-8")) <= character_limit:
                current_chunk = separated_chunk
            else:
                print(f"ATTENTION (TTS): Segment trop long même après découpage: {separated_chunk[:50]}...")
                merged_chunks.append(separated_chunk) # Send as is, API might reject
                current_chunk = ""
    if current_chunk:
        merged_chunks.append(current_chunk)
    return [chunk for chunk in merged_chunks if chunk and not chunk.isspace()]

def _fetch_audio_bytes_from_api(endpoint: Dict[str, str], text_chunk: str, voice_id: str) -> Optional[str]:
    try:
        api_url = endpoint.get("url")
        response_key = endpoint.get("response")
        if not api_url or not response_key:
            print(f"ERREUR (TTS API): Configuration d'endpoint invalide: {endpoint}")
            return None
        print(f"  [TTS API] Envoi du segment à {api_url} (Voix: {voice_id}): {text_chunk[:30]}...")
        response = requests.post(api_url, json={"text": text_chunk, "voice": voice_id}, timeout=15)
        response.raise_for_status()
        json_response = response.json()
        if response_key in json_response:
             print(f"  [TTS API] Données du segment reçues.")
             return json_response[response_key]
        else:
             print(f"  [TTS API] ERREUR: Clé '{response_key}' non trouvée dans la réponse: {json_response}")
             return None
    except requests.exceptions.Timeout:
        print(f"  [TTS API] ERREUR: Timeout pour le segment: {text_chunk[:30]}...")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [TTS API] ERREUR de requête pour le segment audio: {e}")
        return None
    except KeyError: # Should be caught by api_url/response_key check, but as fallback
        print(f"  [TTS API] ERREUR: Configuration d'endpoint ou structure de réponse API invalide.")
        return None
    except Exception as e:
        print(f"  [TTS API] ERREUR inattendue pendant la récupération: {e}")
        return None

def generate_api_audio(text: str, output_file_path: str, voice_enum_val: Voice) -> bool:
    print(f"[TTS API] Génération audio pour voix {voice_enum_val.name} ({voice_enum_val.value}): {text[:50]}...")
    try:
        if not isinstance(voice_enum_val, Voice): raise TypeError(f"'voice_enum_val' doit être de type Voice, reçu {type(voice_enum_val)}")
        if not text or text.isspace(): raise ValueError("text ne doit pas être vide")
    except (TypeError, ValueError) as e:
        print(f"[TTS API] ERREUR: Arguments invalides - {e}")
        return False

    endpoint_data = _load_endpoints()
    if not endpoint_data:
        print("[TTS API] ERREUR: Aucun endpoint chargé. Impossible de générer l'audio API.")
        return False

    for endpoint in endpoint_data:
        print(f"[TTS API] Essai de l'endpoint: {endpoint.get('name', endpoint.get('url', 'N/A'))}")
        text_chunks: List[str] = _split_text(text)
        if not text_chunks:
            print("[TTS API] ERREUR: Le texte a résulté en segments vides après découpage.")
            continue # Try next endpoint

        audio_chunks_b64: List[Optional[str]] = [None] * len(text_chunks)
        threads: List[threading.Thread] = []
        results_lock = threading.Lock()
        results: Dict[int, Optional[str]] = {}

        def thread_target(index: int, chunk: str):
            audio_data = _fetch_audio_bytes_from_api(endpoint, chunk, voice_enum_val.value)
            with results_lock:
                results[index] = audio_data

        for i, chunk in enumerate(text_chunks):
            thread = threading.Thread(target=thread_target, args=(i, chunk))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join() # Wait for all chunk fetches for this endpoint

        for i in range(len(text_chunks)): # Populate in order
            audio_chunks_b64[i] = results.get(i)

        if all(chunk is not None for chunk in audio_chunks_b64):
            print("[TTS API] Tous les segments reçus avec succès. Concaténation et décodage...")
            try:
                full_audio_b64 = "".join([chunk for chunk in audio_chunks_b64 if chunk is not None])
                audio_bytes = base64.b64decode(full_audio_b64)
                with open(output_file_path, "wb") as file: # _save_audio_file inlined
                    file.write(audio_bytes)
                print(f"[TTS API] Audio sauvegardé avec succès dans {output_file_path}")
                return True
            except base64.binascii.Error as e:
                print(f"[TTS API] ERREUR de décodage base64: {e}")
            except IOError as e:
                print(f"[TTS API] ERREUR de sauvegarde du fichier audio {output_file_path}: {e}")
            except Exception as e:
                print(f"[TTS API] ERREUR inattendue pendant sauvegarde/décodage: {e}")
            return False # Error during processing this successful fetch
        else:
             missing_indices = [i for i, ch_data in enumerate(audio_chunks_b64) if ch_data is None]
             print(f"[TTS API] Échec de récupération de tous les segments audio pour l'endpoint {endpoint.get('name', endpoint.get('url', 'N/A'))}. Segments manquants indices: {missing_indices}")
    print("[TTS API] ERREUR: Échec de génération audio avec tous les endpoints disponibles.")
    return False

def generate_female_audio(text: str, output_file_path: str) -> bool:
    if not TTS_GTTS_OK:
        print("[TTS gTTS] ERREUR: gTTS n'est pas disponible. Voix féminine française désactivée.")
        return False
    print(f"[TTS gTTS] Génération audio (fr_female) pour: {text[:50]}...")
    try:
        if not text or text.isspace(): raise ValueError("text ne doit pas être vide")
        tts = gTTS(text=text, lang='fr', slow=False)
        tts.save(output_file_path)
        print(f"[TTS gTTS] Audio sauvegardé avec succès dans {output_file_path}")
        return True
    except ValueError as e:
        print(f"[TTS gTTS] ERREUR: Argument invalide - {e}")
        return False
    except Exception as e:
        print(f"[TTS gTTS] ERREUR de génération ou sauvegarde audio: {e}")
        return False

def speed_change(sound: AudioSegment, speed_factor: float = 1.0) -> AudioSegment:
    if not TTS_PYDUB_OK: return sound # Should not happen if called
    if speed_factor == 1.0:
        return sound
    print(f"  [TTS Playback] Application du facteur de vitesse: {speed_factor:.2f}")
    try:
        new_frame_rate = int(sound.frame_rate * speed_factor)
        if new_frame_rate <= 0:
            print(f"  [TTS Playback] ATTENTION: Taux d'échantillonnage calculé ({new_frame_rate}) invalide. Utilisation du taux original.")
            return sound
        return sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
    except Exception as e:
        print(f"  [TTS Playback] ERREUR pendant le changement de vitesse: {e}. Utilisation du son original.")
        return sound


def play_audio_file(file_path: str, speed_to_use: float):
    if not TTS_PYDUB_OK:
        print("[TTS Playback] Pydub non disponible. Lecture audio sautée.")
        return
    final_audio_to_play = None
    try:
        print(f"[TTS Playback] Chargement de {file_path}...")
        file_extension = os.path.splitext(file_path)[1].lower().strip('.')
        if not file_extension:
            file_extension = "mp3" # Assume mp3 if no extension
            print(f"[TTS Playback] ATTENTION: Aucune extension de fichier détectée, en supposant '{file_extension}'.")

        audio = AudioSegment.from_file(file_path, format=file_extension)
        audio_at_speed = speed_change(audio, speed_to_use)

        # Resample if speed changed and frame rate is not standard, to help playback compatibility
        # Common playback rates: 44100 Hz, 48000 Hz. Let's target 44100 for broad compatibility.
        target_frame_rate_for_playback = 44100
        if int(audio_at_speed.frame_rate) != target_frame_rate_for_playback and speed_to_use != 1.0:
            print(f"  [TTS Playback] Rééchantillonnage de {audio_at_speed.frame_rate} Hz à {target_frame_rate_for_playback} Hz pour compatibilité...")
            final_audio_to_play = audio_at_speed.set_frame_rate(target_frame_rate_for_playback)
        else:
            final_audio_to_play = audio_at_speed
        
        if final_audio_to_play is None: # Should not happen
             print(f"[TTS Playback] ERREUR: `final_audio_to_play` est None avant pydub_play.")
             return

        print(f"[TTS Playback] Lecture à {speed_to_use:.2f}x vitesse (Taux Échant.: {final_audio_to_play.frame_rate} Hz)...")
        pydub_play(final_audio_to_play)
        print(f"[TTS Playback] Lecture de {file_path} terminée.")

    except FileNotFoundError:
        print(f"[TTS Playback] ERREUR: Fichier non trouvé {file_path}")
    except Exception as e:
        current_rate_info = final_audio_to_play.frame_rate if final_audio_to_play else "N/A"
        if "Weird sample rates" in str(e):
             print(f"[TTS Playback] ERREUR de lecture {file_path}: {e}")
             print(f"[TTS Playback] Même après rééchantillonnage, le taux {current_rate_info} Hz peut ne pas être supporté.")
        elif "Cannot find ffprobe" in str(e) or "Cannot find ffmpeg" in str(e) or "[WinError 2]" in str(e):
             print(f"[TTS Playback] ERREUR de lecture {file_path}: {e}")
             print(f"[TTS Playback] ERREUR: Assurez-vous que 'ffmpeg'/'ffprobe' sont trouvés par pydub (PATH).")
        elif "[Errno 13] Permission denied" in str(e): # Common on Windows with temp files
             print(f"[TTS Playback] ERREUR de lecture {file_path}: {e}")
             print(f"[TTS Playback] Permission refusée, possiblement sur un fichier temporaire. Vérifiez antivirus/permissions.")
        else:
            print(f"[TTS Playback] ERREUR de lecture {file_path}: {e}")
            print("[TTS Playback] Assurez-vous que 'ffmpeg' est installé et dans le PATH.")
            print("[TTS Playback] Vérifiez aussi le format du fichier ou les permissions.")

tts_queue: queue.Queue = queue.Queue() # (text, voice_key, specific_speed)
stop_worker_event = threading.Event() # Renamed for clarity

def _validate_request_args(text: str, voice_key: str, speed: float):
    if not text or text.isspace():
        raise ValueError("[TTS Validation] ERREUR: Le texte ne peut pas être vide.")
    voice_key_lower = voice_key.lower()
    if voice_key_lower not in valid_voice_keys:
        raise ValueError(f"[TTS Validation] ERREUR: Clé de voix invalide '{voice_key}'. Valides: {', '.join(valid_voice_keys)}")
    if not isinstance(speed, (int, float)) or speed <= 0:
        raise ValueError("[TTS Validation] ERREUR: La vitesse doit être un nombre positif.")


def tts_worker():
    print("[TTS Worker] Thread Worker TTS démarré.")
    while not stop_worker_event.is_set():
        try:
            text, voice_key, specific_speed = tts_queue.get(timeout=1.0) # Get (text, voice_key, specific_speed)
            print(f"\n[TTS Worker] Traitement requête: VoiceKey='{voice_key}', Speed={specific_speed:.2f}x, Text='{text[:50]}...'")

            if not TTS_PYDUB_OK: # Double check, though add_speech_request should prevent queueing
                print("[TTS Worker] Pydub non disponible. Annulation de la requête.")
                tts_queue.task_done()
                continue

            success = False
            temp_file_path = None
            try:
                # Create a named temporary file that persists after close (delete=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
                    temp_file_path = tmp_audio_file.name
                print(f"[TTS Worker] Utilisation du fichier temporaire: {temp_file_path}")

                voice_key_lower = voice_key.lower()
                if voice_key_lower == 'fr_female':
                    success = generate_female_audio(text, temp_file_path)
                elif voice_key_lower in api_voice_keys:
                    target_voice_enum = api_voice_keys[voice_key_lower]
                    success = generate_api_audio(text, temp_file_path, target_voice_enum)
                else:
                    print(f"[TTS Worker] ERREUR: Clé de voix inconnue '{voice_key}'") # Should be caught by _validate

                if success and temp_file_path and os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                    play_audio_file(temp_file_path, specific_speed) # Pass specific speed
                elif not success:
                    print(f"[TTS Worker] Échec de génération audio pour la requête.")
                elif temp_file_path and os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) == 0:
                    print(f"[TTS Worker] ATTENTION: Fichier audio généré est vide: {temp_file_path}")


            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"[TTS Worker] Fichier temporaire nettoyé: {temp_file_path}")
                    except OSError as e:
                        print(f"[TTS Worker] ERREUR de suppression du fichier temporaire {temp_file_path}: {e}")
            tts_queue.task_done()
        except queue.Empty:
            continue # Normal, just loop and wait for new items or stop signal
        except Exception as e:
            print(f"[TTS Worker] ERREUR inattendue dans la boucle worker: {e}")
            import traceback
            traceback.print_exc()
            try: # Attempt to mark task done even on error to prevent blocking queue.join()
                tts_queue.task_done()
            except ValueError: # If task_done called too many times
                 pass
            time.sleep(1) # Brief pause before retrying loop
    print("[TTS Worker] Thread Worker TTS terminé.")

def add_speech_request(text: str, voice_key: str, speed: float = 1.0) -> bool:
    if not TTS_PYDUB_OK: # Do not queue if playback is impossible
        print(f"DEBUG (TTS add_speech_request): Pydub non disponible. Requête TTS pour '{text[:20]}...' ignorée.")
        return False
    try:
        _validate_request_args(text, voice_key, speed)
    except ValueError as e:
        print(e) # Print validation error
        return False # Indicate failure to queue

    print(f"[TTS Queue] Ajout requête: VoiceKey={voice_key}, Speed={speed:.2f}x, Text='{text[:50]}...'")
    tts_queue.put((text, voice_key.lower(), speed))
    return True # Indicate success in queuing

# --- Colors Class ---
class Colors:
    CV_BLUE = (255, 0, 0); CV_GREEN = (0, 255, 0); CV_RED = (0, 0, 255)
    CV_WHITE = (255, 255, 255); CV_YELLOW = (0, 255, 255); CV_LIGHT_GREY = (211, 211, 211)
    try:
        MP_HAND_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils') else None
        MP_CONNECTION_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2) if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils') else None
    except AttributeError as e_colors_mp:
        print(f"WARN: Exception defining MediaPipe drawing specs: {e_colors_mp}"); MP_HAND_DRAWING_SPEC = None; MP_CONNECTION_DRAWING_SPEC = None
    except Exception as e_colors:
        print(f"ERREUR: Exception during Colors class (MediaPipe part): {e_colors}"); MP_HAND_DRAWING_SPEC = None; MP_CONNECTION_DRAWING_SPEC = None

# --- VideoThread Class ---
class VideoThread(QThread):
    frame_ready = Signal(np.ndarray); prediction_ready = Signal(str); top_n_ready = Signal(list)
    models_loaded = Signal(bool); error_occurred = Signal(str); hands_detected_signal = Signal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        print("DEBUG (VideoThread.__init__): Initializing...")
        self._running = False
        try:
            self.MODEL_PATH = os.path.join(config.BASE_DIR, config.MODEL_DIR, config.ACTIVE_MODEL_FILENAME)
            self.VOCABULARY_PATH = config.VOCABULARY_PATH; self.FIXED_LENGTH = config.FIXED_LENGTH
            self.FEATURE_DIM = config.ACTIVE_FEATURE_DIM; self.CNN_MODEL_CHOICE = config.CNN_MODEL_CHOICE
            self.CNN_INPUT_SHAPE = config.CNN_INPUT_SHAPE; self.CAPTURE_SOURCE = config.CAPTURE_SOURCE
            self.FRAMES_TO_SKIP = config.FRAMES_TO_SKIP; self.PREDICTION_THRESHOLD = config.PREDICTION_THRESHOLD
            self.SMOOTHING_WINDOW_SIZE = config.CAPTURE_SMOOTHING_WINDOW_SIZE; self.TOP_N = config.CAPTURE_TOP_N
            self.MAX_FRAME_WIDTH = config.CAPTURE_MAX_FRAME_WIDTH
            self.MIN_HAND_DETECTION_CONFIDENCE = getattr(config, 'MIN_HAND_DETECTION_CONFIDENCE', 0.5)
            self.MIN_HAND_TRACKING_CONFIDENCE = getattr(config, 'MIN_HAND_TRACKING_CONFIDENCE', 0.5)
            self.MAX_HANDS = getattr(config, 'MAX_HANDS', 2); self.ui_TOP_N = config.CAPTURE_TOP_N
            print("DEBUG (VideoThread.__init__): Config loaded.")
        except AttributeError as e:
            error_msg = f"Erreur de configuration (config.py): Attribut manquant '{e.name}'"; print(f"ERREUR: {error_msg}"); raise RuntimeError(error_msg) from e
        self.cnn_feature_extractor_model = None; self.preprocess_function = None; self.cnn_target_size = None
        self.lstm_prediction_model = None; self.vocabulaire = None; self.index_to_word = None; self.cap = None
        self.mp_hands = None; self.hands_solution = None; self.mp_drawing = None
        self.drawing_spec_hand = Colors.MP_HAND_DRAWING_SPEC; self.drawing_spec_connection = Colors.MP_CONNECTION_DRAWING_SPEC
        self.last_hands_detected_status = False; print("DEBUG (VideoThread.__init__): Initialized variables.")
    def load_vocabulary(self):
        vocab = {}; print(f"DEBUG (VideoThread.load_vocabulary): Attempting to load from {self.VOCABULARY_PATH}")
        try:
            if not os.path.exists(self.VOCABULARY_PATH): raise FileNotFoundError(f"Fichier vocabulaire non trouvé: '{self.VOCABULARY_PATH}'")
            with open(self.VOCABULARY_PATH, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or ':' not in line:
                        if line: logging.warning(f"Ligne {line_num} format incorrect '{self.VOCABULARY_PATH}': '{line}' - Ignorée"); continue
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[0] and parts[1].isdigit(): vocab[parts[0].strip().lower()] = int(parts[1].strip())
                    else: logging.warning(f"Ligne {line_num} ignorée '{self.VOCABULARY_PATH}': '{line}' - Format après split."); print(f"DEBUG (VideoThread.load_vocabulary): Ligne {line_num} ignorée: '{line}'")
            if not vocab: error_msg = f"Vocabulaire chargé de '{self.VOCABULARY_PATH}' est vide/invalide."; logging.error(error_msg); self.error_occurred.emit(error_msg); print(f"DEBUG (VideoThread.load_vocabulary): Vocabulaire vide: {self.VOCABULARY_PATH}"); return None
            logging.info(f"Vocabulaire chargé ({len(vocab)} mots)."); print(f"DEBUG (VideoThread.load_vocabulary): Succès ({len(vocab)} mots)."); return vocab
        except FileNotFoundError as e: error_msg = str(e); logging.error(f"Erreur: {error_msg}"); self.error_occurred.emit(error_msg); print(f"DEBUG (VideoThread.load_vocabulary): Fichier non trouvé: {self.VOCABULARY_PATH}"); return None
        except Exception as e: error_msg = f"Erreur chargement vocabulaire '{self.VOCABULARY_PATH}': {e}"; logging.exception(error_msg); self.error_occurred.emit(error_msg); print(f"DEBUG (VideoThread.load_vocabulary): Erreur chargement: {e}"); return None
    def load_models_and_preprocessing(self):
        print("DEBUG (VideoThread.load_models_and_preprocessing): Chargement modèles..."); model_name = self.CNN_MODEL_CHOICE; input_shape = self.CNN_INPUT_SHAPE; self.cnn_target_size = input_shape[:2]
        logging.info(f"Chargement CNN: {model_name} avec shape entrée {input_shape}..."); print(f"DEBUG (VideoThread.load_models_and_preprocessing): Chargement CNN: {model_name}")
        try:
            if model_name == 'MobileNetV2': base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet'); self.preprocess_function = mobilenet_preprocess
            elif model_name == 'ResNet50': base = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet'); self.preprocess_function = resnet_preprocess
            elif model_name == 'EfficientNetB0': base = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet'); self.preprocess_function = efficientnet_preprocess
            else: raise ValueError(f"Modèle CNN non supporté: {model_name}")
            output = GlobalAveragePooling2D()(base.output); self.cnn_feature_extractor_model = Model(inputs=base.input, outputs=output, name=f"{model_name}_FeatureExtractor")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): Structure CNN {model_name} créée. Initialisation..."); dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32); _ = self.cnn_feature_extractor_model(dummy_input, training=False)
            logging.info(f"Modèle CNN {model_name} chargé et initialisé."); print(f"DEBUG (VideoThread.load_models_and_preprocessing): Modèle CNN {model_name} chargé et initialisé.")
        except Exception as e: error_msg = f"Erreur chargement modèle CNN '{model_name}': {e}"; logging.exception(error_msg); self.error_occurred.emit(error_msg); print(f"DEBUG (VideoThread.load_models_and_preprocessing): ERREUR CRITIQUE chargement CNN: {e}"); return False
        logging.info(f"Chargement modèle LSTM: {self.MODEL_PATH}..."); print(f"DEBUG (VideoThread.load_models_and_preprocessing): Chargement LSTM: {self.MODEL_PATH}")
        try:
            if not os.path.exists(self.MODEL_PATH): raise FileNotFoundError(f"Fichier modèle LSTM non trouvé: {self.MODEL_PATH}")
            self.lstm_prediction_model = tf.keras.models.load_model(self.MODEL_PATH); logging.info(f"Modèle LSTM chargé de {self.MODEL_PATH}"); print(f"DEBUG (VideoThread.load_models_and_preprocessing): LSTM chargé. Vérif shape...")
            expected_lstm_shape = self.lstm_prediction_model.input_shape; logging.info(f"Shape entrée LSTM attendue: {expected_lstm_shape}"); print(f"DEBUG (VideoThread.load_models_and_preprocessing): Shape entrée LSTM attendue: {expected_lstm_shape}")
            if len(expected_lstm_shape) != 3: raise ValueError(f"Shape entrée LSTM a un rang inattendu: {len(expected_lstm_shape)}")
            model_seq_len = expected_lstm_shape[1]; model_feat_dim = expected_lstm_shape[2]
            if model_seq_len is not None and model_seq_len != self.FIXED_LENGTH: logging.warning(f"Avertissement Incompatibilité Longueur Séquence LSTM! Modèle attend {model_seq_len}, config.FIXED_LENGTH est {self.FIXED_LENGTH}. Remplissage/troncature aura lieu.")
            if model_feat_dim is not None and model_feat_dim != self.FEATURE_DIM: raise ValueError(f"CRITIQUE Incompatibilité Dimension Features LSTM! Modèle attend {model_feat_dim}, config.ACTIVE_FEATURE_DIM est {self.FEATURE_DIM}.")
            dummy_lstm_input = tf.zeros((1, self.FIXED_LENGTH, self.FEATURE_DIM), dtype=tf.float32); _ = self.lstm_prediction_model(dummy_lstm_input, training=False); logging.info("Modèle LSTM initialisé."); print("DEBUG (VideoThread.load_models_and_preprocessing): LSTM initialisé.")
        except Exception as e: error_msg = f"Erreur chargement/initialisation LSTM '{self.MODEL_PATH}': {e}"; logging.exception(error_msg); self.error_occurred.emit(f"Erreur chargement LSTM: {e}"); print(f"DEBUG (VideoThread.load_models_and_preprocessing): Erreur LSTM: {e}"); return False
        print("DEBUG (VideoThread.load_models_and_preprocessing): Modèles chargés avec succès."); return True
    def extract_cnn_features_realtime(self, frame):
        if self.cnn_feature_extractor_model is None or self.preprocess_function is None or self.cnn_target_size is None: logging.error("Extracteur CNN non initialisé."); print("DEBUG (VideoThread.extract_cnn_features_realtime): Extracteur CNN non initialisé."); return None
        try:
            target_size_cv2 = (self.cnn_target_size[1], self.cnn_target_size[0]); img_resized_cv = cv2.resize(frame, target_size_cv2, interpolation=cv2.INTER_AREA)
            img_resized_tensor = tf.convert_to_tensor(img_resized_cv, dtype=tf.float32); img_batch_tensor = tf.expand_dims(img_resized_tensor, axis=0)
            img_preprocessed_tensor = self.preprocess_function(img_batch_tensor); features_tensor = self.cnn_feature_extractor_model(img_preprocessed_tensor, training=False)
            return features_tensor[0].numpy()
        except Exception as e: logging.warning(f"Erreur extraction features CNN (CV2Resize+TF): {e}", exc_info=False); print(f"DEBUG (VideoThread.extract_cnn_features_realtime): ERREUR extraction: {type(e).__name__}: {e}"); return None
    def run(self):
        self._running = True; logging.info("Thread traitement vidéo démarré."); print("DEBUG: VideoThread run() démarré"); print("DEBUG (VideoThread.run): Configuration TensorFlow/GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configurés pour croissance mémoire: {gpus}"); print(f"DEBUG (VideoThread.run): GPU(s) configurés: {gpus}")
            except RuntimeError as e: logging.error(f"Erreur config croissance mémoire GPU: {e}"); print(f"DEBUG (VideoThread.run): Erreur config GPU (croissance mémoire): {e}")
        else: logging.warning("Aucun GPU détecté par TensorFlow. Inférence sur CPU."); print("DEBUG (VideoThread.run): Aucun GPU détecté par TensorFlow.")
        print(f"DEBUG (VideoThread.run): Vérif GPU terminée.")
        print("DEBUG (VideoThread.run): Tentative chargement modèles..."); models_ok = self.load_models_and_preprocessing(); print(f"DEBUG (VideoThread.run): Modèles chargés OK: {models_ok}")
        if not models_ok: self.models_loaded.emit(False); self._running = False; logging.error("Échec chargement modèles. Arrêt thread vidéo."); print("DEBUG (VideoThread.run): Sortie run() cause échec chargement modèle"); return
        print("DEBUG (VideoThread.run): Tentative chargement vocabulaire..."); self.vocabulaire = self.load_vocabulary(); print(f"DEBUG (VideoThread.run): Vocabulaire chargé: {'OK' if self.vocabulaire else 'ÉCHEC'}")
        if not self.vocabulaire: self.models_loaded.emit(False); self._running = False; logging.error("Échec chargement vocabulaire. Arrêt thread vidéo."); print("DEBUG (VideoThread.run): Sortie run() cause échec chargement vocab"); return
        try:
            self.index_to_word = {i: word for word, i in self.vocabulaire.items()}
            if len(self.index_to_word) != len(self.vocabulaire): logging.warning("Indices dupliqués potentiels dans fichier vocabulaire.")
            logging.info(f"Vocabulaire inversé créé ({len(self.index_to_word)} entrées)."); print(f"DEBUG (VideoThread.run): Vocabulaire inversé créé ({len(self.index_to_word)} mots).")
        except Exception as e: error_msg = f"Erreur création map vocabulaire inversé: {e}"; logging.error(error_msg); self.error_occurred.emit(error_msg); self.models_loaded.emit(False); self._running = False; print(f"DEBUG (VideoThread.run): Sortie run() cause erreur vocab inversé"); return
        if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            print("DEBUG (VideoThread.run): Initialisation Mediapipe Hands..."); self.mp_hands = mp.solutions.hands; self.mp_drawing = mp.solutions.drawing_utils
            try:
                if self.drawing_spec_hand is None: self.drawing_spec_hand = self.mp_drawing.DrawingSpec(color=Colors.CV_GREEN, thickness=2, circle_radius=2)
                if self.drawing_spec_connection is None: self.drawing_spec_connection = self.mp_drawing.DrawingSpec(color=Colors.CV_RED, thickness=2)
                self.hands_solution = self.mp_hands.Hands(static_image_mode=False, max_num_hands=self.MAX_HANDS, min_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE, min_tracking_confidence=self.MIN_HAND_TRACKING_CONFIDENCE)
                print(f"DEBUG (VideoThread.run): Mediapipe Hands initialisé (max_mains={self.MAX_HANDS}, conf_det={self.MIN_HAND_DETECTION_CONFIDENCE:.2f}, conf_track={self.MIN_HAND_TRACKING_CONFIDENCE:.2f}).")
            except Exception as e_mp: print(f"ERREUR (VideoThread.run): Échec init Mediapipe Hands: {e_mp}"); self.error_occurred.emit(f"Erreur init Mediapipe: {e_mp}"); self.hands_solution = None
        else: print("ATTENTION (VideoThread.run): Module Mediapipe ou solution hands non trouvé/chargé. Optimisation détection main désactivée."); self.hands_solution = None; self.drawing_spec_hand = None; self.drawing_spec_connection = None
        self.models_loaded.emit(True); print("DEBUG (VideoThread.run): Émis models_loaded(True)")
        logging.info(f"Ouverture source capture caméra: {self.CAPTURE_SOURCE}"); print(f"DEBUG (VideoThread.run): Tentative ouverture source caméra: {self.CAPTURE_SOURCE} (type: {type(self.CAPTURE_SOURCE)})")
        self.cap = None; capture_backend = cv2.CAP_ANY
        if sys.platform == "win32": capture_backend = cv2.CAP_DSHOW; print("DEBUG (VideoThread.run): Utilisation backend cv2.CAP_DSHOW préféré sur Windows")
        try:
            source_to_open = int(self.CAPTURE_SOURCE) if str(self.CAPTURE_SOURCE).isdigit() else self.CAPTURE_SOURCE
            print(f"DEBUG (VideoThread.run): Appel cv2.VideoCapture({source_to_open}, {capture_backend})"); self.cap = cv2.VideoCapture(source_to_open, capture_backend); time.sleep(0.5)
            is_opened = self.cap.isOpened() if self.cap else False; print(f"DEBUG (VideoThread.run): Caméra ouverte après tentative initiale: {is_opened}")
            if not is_opened and sys.platform == "win32" and capture_backend == cv2.CAP_DSHOW:
                 print("DEBUG (VideoThread.run): CAP_DSHOW échoué, essai backend par défaut (CAP_ANY)...")
                 if self.cap: self.cap.release(); capture_backend = cv2.CAP_ANY; self.cap = cv2.VideoCapture(source_to_open, capture_backend); time.sleep(0.5)
                 is_opened = self.cap.isOpened() if self.cap else False; print(f"DEBUG (VideoThread.run): Caméra ouverte avec backend par défaut: {is_opened}")
            if not is_opened: raise IOError(f"Impossible d'ouvrir source caméra '{source_to_open}' avec backends testés.")
        except Exception as e_cap:
             logging.error(f"Erreur ouverture capture caméra {self.CAPTURE_SOURCE}: {e_cap}", exc_info=True)
             error_msg = f"Erreur ouverture webcam {self.CAPTURE_SOURCE}: {e_cap}"; self.error_occurred.emit(error_msg); self.models_loaded.emit(False); self._running = False
             print(f"DEBUG (VideoThread.run): Sortie run() car caméra non ouverte.");
             if self.cap: self.cap.release()
             if self.hands_solution: self.hands_solution.close()
             return
        logging.info("Webcam ouverte avec succès."); print("DEBUG (VideoThread.run): Webcam ouverte avec succès.")
        sequence_window = deque(maxlen=self.FIXED_LENGTH); prediction_display_buffer = deque(maxlen=self.SMOOTHING_WINDOW_SIZE)
        frame_processing_times = deque(maxlen=30); frame_count = 0; last_smoothed_word = "?"; print("DEBUG (VideoThread.run): Variables boucle temps-réel initialisées.")
        target_width = None; target_height = None; resize_needed = False
        try:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width <= 0 or frame_height <= 0: raise ValueError("Dimensions frame invalides depuis caméra")
            logging.info(f"Résolution native webcam: {frame_width}x{frame_height}"); print(f"DEBUG (VideoThread.run): Résolution native caméra: {frame_width}x{frame_height}")
            if self.MAX_FRAME_WIDTH and frame_width > self.MAX_FRAME_WIDTH:
                scale = self.MAX_FRAME_WIDTH / frame_width; target_width = self.MAX_FRAME_WIDTH; target_height = int(frame_height * scale)
                target_height = target_height if target_height % 2 == 0 else target_height + 1; resize_needed = True
                logging.info(f"Redimensionnement affichage activé: Largeur cible {target_width}px (hauteur ~{target_height}px)"); print(f"DEBUG (VideoThread.run): Affichage redimensionné à {target_width}x{target_height}")
            else: target_width = frame_width; target_height = frame_height; print("DEBUG (VideoThread.run): Pas de redimensionnement affichage basé sur MAX_FRAME_WIDTH.")
        except Exception as e_res:
            logging.warning(f"Impossible lire résolution caméra: {e_res}. Utilisation taille affichage secours."); print(f"DEBUG (VideoThread.run): Impossible obtenir résolution caméra: {e_res}")
            target_width = 640; target_height = 480; resize_needed = True; print(f"DEBUG (VideoThread.run): Recours à taille affichage {target_width}x{target_height}")
        print("DEBUG (VideoThread.run): Entrée boucle vidéo principale..."); loop_count = 0
        while self._running:
            loop_start_time = time.time(); loop_count += 1
            try: ret, frame = self.cap.read()
            except Exception as e_read: logging.error(f"Exception pendant cap.read() (itération {loop_count}): {e_read}", exc_info=True); self.error_occurred.emit(f"Erreur lecture webcam: {e_read}"); print(f"DEBUG: Rupture boucle... cap.read(): {e_read}"); break
            if not ret or frame is None:
                logging.error(f"Échec lecture frame (itération {loop_count}, ret={ret}, frame is None={frame is None}). Arrêt thread.")
                is_still_opened = self.cap.isOpened() if self.cap else False; error_msg = f"Impossible lire frame (tentative {loop_count}). Caméra ouverte: {is_still_opened}"
                if self._running: self.error_occurred.emit(error_msg); print(f"DEBUG: Rupture boucle... impossible lire frame. Caméra ouverte: {is_still_opened}"); break
            frame_count += 1; display_frame = None
            if resize_needed and target_width and target_height:
                try: display_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                except Exception as e_resize: logging.warning(f"Erreur redimensionnement frame affichage: {e_resize}"); display_frame = frame.copy()
            else: display_frame = frame.copy()
            hands_detected_this_frame = False
            if self.hands_solution:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False; results = self.hands_solution.process(image_rgb)
                    if results.multi_hand_landmarks:
                        hands_detected_this_frame = True
                        for hand_landmarks in results.multi_hand_landmarks:
                            if self.mp_drawing and self.drawing_spec_hand and self.drawing_spec_connection:
                                try: self.mp_drawing.draw_landmarks(display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.drawing_spec_hand, self.drawing_spec_connection)
                                except Exception as e_draw: logging.warning(f"Erreur dessin landmarks main: {e_draw}", exc_info=False); print(f"DEBUG Boucle {loop_count}: Avertissement - erreur dessin landmarks: {e_draw}")
                except Exception as e_hand_detect: logging.warning(f"Erreur pendant traitement main Mediapipe: {e_hand_detect}", exc_info=False); print(f"DEBUG Boucle {loop_count}: Avertissement - erreur détection main: {e_hand_detect}")
            if hands_detected_this_frame != self.last_hands_detected_status:
                if self._running: self.hands_detected_signal.emit(hands_detected_this_frame)
                self.last_hands_detected_status = hands_detected_this_frame; print(f"DEBUG Boucle {loop_count}: Statut détection mains changé: {hands_detected_this_frame}")
            should_run_inference = False; hand_check_passed = (not self.hands_solution) or hands_detected_this_frame; frame_interval_check_passed = (frame_count % (self.FRAMES_TO_SKIP + 1) == 0)
            if hand_check_passed and frame_interval_check_passed: should_run_inference = True
            if should_run_inference:
                inference_start_time = time.time(); cnn_features = self.extract_cnn_features_realtime(frame); processing_time_ms = (time.time() - inference_start_time) * 1000; frame_processing_times.append(processing_time_ms)
                if cnn_features is not None:
                    sequence_window.append(cnn_features); current_sequence_len = len(sequence_window)
                    if current_sequence_len > 0:
                        padded_sequence = None; current_sequence_np = np.array(sequence_window, dtype=np.float32)
                        if current_sequence_len < self.FIXED_LENGTH:
                            padding_size = self.FIXED_LENGTH - current_sequence_len
                            try: paddings = tf.constant([[padding_size, 0], [0, 0]], dtype=tf.int32); padded_sequence = tf.pad(current_sequence_np, paddings, "CONSTANT", constant_values=0.0).numpy()
                            except Exception as e_pad: print(f"DEBUG: Erreur tf.pad: {e_pad}. Recours à np.concatenate."); padding_array = np.zeros((padding_size, self.FEATURE_DIM), dtype=np.float32); padded_sequence = np.concatenate((padding_array, current_sequence_np), axis=0)
                        else: padded_sequence = current_sequence_np
                        if padded_sequence is not None and padded_sequence.shape == (self.FIXED_LENGTH, self.FEATURE_DIM):
                            reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                            try:
                                prediction_probs = self.lstm_prediction_model(reshaped_sequence, training=False).numpy()[0]
                                top_n_indices = np.argsort(prediction_probs)[-self.TOP_N:][::-1]; top_n_confidences = prediction_probs[top_n_indices]
                                top_n_words = [self.index_to_word.get(idx, f"UNK_{idx}") for idx in top_n_indices]; top_n_display_list = [f"{word} ({conf:.2f})" for word, conf in zip(top_n_words, top_n_confidences)]
                                if self._running: self.top_n_ready.emit(top_n_display_list)
                                top_pred_idx = top_n_indices[0]; top_pred_conf = top_n_confidences[0]
                                if top_pred_conf >= self.PREDICTION_THRESHOLD: prediction_display_buffer.append(top_pred_idx)
                            except Exception as e_pred: logging.exception(f"Erreur prédiction LSTM: {e_pred}"); print(f"DEBUG: Exception prédiction LSTM: {e_pred}"); self.top_n_ready.emit(["Erreur Prediction LSTM"])
                        else:
                            if padded_sequence is not None: print(f"DEBUG: Shape séquence incorrecte avant LSTM: {padded_sequence.shape}")
                            else: print("DEBUG: Séquence remplie est None."); self.top_n_ready.emit(["Erreur Shape Séquence"])
                else: print(f"DEBUG Boucle {loop_count}: Extraction Features CNN a retourné None."); self.top_n_ready.emit(["Erreur Extraction CNN"])
            else:
                if self.hands_solution and not hands_detected_this_frame:
                    if sequence_window: sequence_window.clear(); print(f"DEBUG Boucle {loop_count}: Mains disparues/non détectées, vidage fenêtre séquence.")
                    if prediction_display_buffer: prediction_display_buffer.clear(); print(f"DEBUG Boucle {loop_count}: Mains disparues/non détectées, vidage buffer prédiction.")
                    if self.last_hands_detected_status: # Changed from True to False
                         if self._running: self.top_n_ready.emit([""]) # Clear status bar
                         if last_smoothed_word != "?": last_smoothed_word = "?"; self.prediction_ready.emit(last_smoothed_word) # Emit "?" to clear current word
            current_smoothed_word = "?"
            if prediction_display_buffer:
                try:
                    word_counts = Counter(prediction_display_buffer); most_common_word = word_counts.most_common(1)
                    if most_common_word: smoothed_index = most_common_word[0][0]; current_smoothed_word = self.index_to_word.get(smoothed_index, "?")
                except Exception as e_smooth: logging.warning(f"Erreur lissage prédiction: {e_smooth}"); print(f"DEBUG Boucle {loop_count}: Exception lissage: {e_smooth}")
            if current_smoothed_word != last_smoothed_word:
                if self._running: self.prediction_ready.emit(current_smoothed_word)
                last_smoothed_word = current_smoothed_word
            try:
                 if frame_processing_times: avg_proc_time = np.mean(frame_processing_times); fps_proc_approx = 1000 / avg_proc_time if avg_proc_time > 0 else 0; cv2.putText(display_frame, f"Proc: {avg_proc_time:.1f}ms (~{fps_proc_approx:.1f} FPS)", (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CV_LIGHT_GREY, 1, cv2.LINE_AA)
                 loop_time_ms = (time.time() - loop_start_time) * 1000; fps_loop_approx = 1000 / loop_time_ms if loop_time_ms > 0 else 0; cv2.putText(display_frame, f"Loop: {loop_time_ms:.1f}ms (~{fps_loop_approx:.1f} FPS)", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CV_LIGHT_GREY, 1, cv2.LINE_AA)
                 if self.hands_solution: status_text = "Mains: Oui" if hands_detected_this_frame else "Mains: Non"; status_color = Colors.CV_GREEN if hands_detected_this_frame else Colors.CV_RED; cv2.putText(display_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)
            except Exception as e_display_debug: logging.warning(f"Erreur dessin info debug: {e_display_debug}", exc_info=False); print(f"DEBUG Boucle {loop_count}: Avertissement - erreur dessin info debug: {e_display_debug}")
            try:
                 if display_frame is not None and display_frame.size > 0:
                      if self._running: self.frame_ready.emit(display_frame)
            except Exception as e_emit: logging.warning(f"Erreur émission signal frame: {e_emit}", exc_info=False); print(f"DEBUG Boucle {loop_count}: Avertissement - erreur émission frame: {e_emit}")
        print(f"DEBUG: Sortie boucle vidéo principale après {loop_count} itérations."); logging.info("Boucle thread vidéo terminée ou arrêtée.")
        if self.cap and self.cap.isOpened():
            try: self.cap.release(); logging.info("Webcam relâchée."); print("DEBUG: Caméra relâchée.")
            except Exception as e_rel: logging.error(f"Exception relâchement caméra: {e_rel}")
        else: print("DEBUG: Caméra non ouverte ou déjà relâchée.")
        if self.hands_solution:
            try: self.hands_solution.close(); print("DEBUG: Solution Mediapipe Hands fermée.")
            except Exception as e_mp_close: print(f"DEBUG: Erreur fermeture Mediapipe: {e_mp_close}")
        try: print("DEBUG: Tentative nettoyage session Keras..."); tf.keras.backend.clear_session(); logging.info("Session Keras/TensorFlow nettoyée."); print("DEBUG: Session Keras nettoyée.")
        except Exception as e_clear: logging.warning(f"Erreur nettoyage session Keras: {e_clear}"); print(f"DEBUG: Erreur nettoyage session Keras: {e_clear}")
        logging.info("Thread vidéo terminé proprement."); print("DEBUG: VideoThread run() terminé")
    def stop(self): print("DEBUG: VideoThread stop() appelé"); self._running = False; logging.info("Arrêt demandé pour thread vidéo.")

# --- Ui_ParametersWindow Class ---
class Ui_ParametersWindow(QWidget):
    color_changed = Signal(QColor); bg_color_changed = Signal(QColor)
    def setupUi(self, ParametersWindow):
        ParametersWindow.setObjectName(u"ParametersWindow"); ParametersWindow.resize(400, 250); ParametersWindow.setWindowTitle("Paramètres d'Affichage")
        self.main_layout = QVBoxLayout(ParametersWindow); self.main_layout.setSpacing(15); self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.label = QLabel("Paramètres d'Affichage", ParametersWindow); self.label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.label.setStyleSheet(u"font-size: 16px; font-weight: bold; margin-bottom: 10px;"); self.main_layout.addWidget(self.label)
        color_group_box = QFrame(ParametersWindow); color_group_box.setFrameShape(QFrame.Shape.StyledPanel); color_layout = QGridLayout(color_group_box); color_layout.setVerticalSpacing(10); color_layout.setHorizontalSpacing(10)
        self.text_color_label = QLabel("Couleur du texte:", color_group_box); self.text_color_btn = QPushButton("Choisir", color_group_box); self.text_color_btn.clicked.connect(self.choose_text_color)
        self.text_color_preview = QLabel(color_group_box); self.text_color_preview.setFixedSize(25, 25); self.text_color_preview.setStyleSheet("background-color: white; border: 1px solid black; border-radius: 3px;"); self.text_color_preview.setToolTip("Aperçu couleur texte")
        self.bg_color_label = QLabel("Couleur de fond:", color_group_box); self.bg_color_btn = QPushButton("Choisir", color_group_box); self.bg_color_btn.clicked.connect(self.choose_bg_color)
        self.bg_color_preview = QLabel(color_group_box); self.bg_color_preview.setFixedSize(25, 25); self.bg_color_preview.setStyleSheet("background-color: rgb(10, 32, 77); border: 1px solid black; border-radius: 3px;"); self.bg_color_preview.setToolTip("Aperçu couleur fond")
        color_layout.addWidget(self.text_color_label, 0, 0); color_layout.addWidget(self.text_color_preview, 0, 1); color_layout.addWidget(self.text_color_btn, 0, 2)
        color_layout.addWidget(self.bg_color_label, 1, 0); color_layout.addWidget(self.bg_color_preview, 1, 1); color_layout.addWidget(self.bg_color_btn, 1, 2)
        color_layout.setColumnStretch(0, 1); color_layout.setColumnStretch(2, 0); self.main_layout.addWidget(color_group_box); self.main_layout.addStretch(1)
        self.buttons_layout = QHBoxLayout(); self.buttons_layout.addStretch(1); self.default_btn = QPushButton("Par défaut", ParametersWindow); self.default_btn.setToolTip("Réinitialiser les couleurs par défaut"); self.default_btn.clicked.connect(self.reset_defaults); self.buttons_layout.addWidget(self.default_btn)
        self.close_btn = QPushButton("Fermer", ParametersWindow); self.close_btn.clicked.connect(ParametersWindow.close); self.buttons_layout.addWidget(self.close_btn); self.main_layout.addLayout(self.buttons_layout); ParametersWindow.setLayout(self.main_layout)
    @Slot()
    def choose_text_color(self):
        parent = self.parentWidget() if self.parentWidget() else self; current = self.text_color_preview.palette().window().color(); color = QColorDialog.getColor(current, parent=parent, title="Choisir couleur du texte")
        if color.isValid(): self.text_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius: 3px;"); self.color_changed.emit(color); print(f"DEBUG (Ui_ParametersWindow): Couleur texte choisie: {color.name()}")
    @Slot()
    def choose_bg_color(self):
        parent = self.parentWidget() if self.parentWidget() else self; current = self.bg_color_preview.palette().window().color(); color = QColorDialog.getColor(current, parent=parent, title="Choisir couleur de fond")
        if color.isValid(): self.bg_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius: 3px;"); self.bg_color_changed.emit(color); print(f"DEBUG (Ui_ParametersWindow): Couleur fond choisie: {color.name()}")
    @Slot()
    def reset_defaults(self):
        default_text = QColor("white"); default_bg = QColor(10, 32, 77)
        self.text_color_preview.setStyleSheet(f"background-color: {default_text.name()}; border: 1px solid black; border-radius: 3px;"); self.bg_color_preview.setStyleSheet(f"background-color: {default_bg.name()}; border: 1px solid black; border-radius: 3px;")
        self.color_changed.emit(default_text); self.bg_color_changed.emit(default_bg); print("DEBUG (Ui_ParametersWindow): Couleurs réinitialisées par défaut.")

# --- Ui_MainWindow Class (Structure verticale) ---
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(u"MainWindow"); MainWindow.setWindowModality(Qt.WindowModality.NonModal); MainWindow.resize(800, 750); MainWindow.setWindowTitle("Traduction LSF en Temps Réel")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); MainWindow.setSizePolicy(sizePolicy)
        self.default_bg_color = QColor(10, 32, 77); self.default_text_color = QColor("white")
        self.centralwidget = QWidget(MainWindow); self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_3 = QGridLayout(self.centralwidget); self.gridLayout_3.setObjectName(u"gridLayout_3"); self.gridLayout_3.setContentsMargins(10, 10, 10, 10); self.gridLayout_3.setVerticalSpacing(15)
        self.setup_top_toolbar(self.centralwidget); self.gridLayout_3.addLayout(self.gridLayout_top_toolbar, 0, 0)
        self.setup_camera_view(self.centralwidget); self.gridLayout_3.addWidget(self.frame_camera, 1, 0)
        self.setup_text_area(self.centralwidget); self.gridLayout_3.addWidget(self.frame_text, 2, 0)
        self.setup_export_controls(self.centralwidget); self.gridLayout_3.addLayout(self.horizontalLayout_export, 3, 0, Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_3.setRowStretch(0, 0); self.gridLayout_3.setRowStretch(1, 2); self.gridLayout_3.setRowStretch(2, 1); self.gridLayout_3.setRowStretch(3, 0); self.gridLayout_3.setColumnStretch(0, 1)
        MainWindow.setCentralWidget(self.centralwidget); self.setup_menu_statusbar(MainWindow); self.retranslateUi(MainWindow)
    def setup_top_toolbar(self, parent):
        self.gridLayout_top_toolbar = QGridLayout(); self.gridLayout_top_toolbar.setObjectName(u"gridLayout_TopToolbar")
        self.boutonparametre = QPushButton(parent); self.boutonparametre.setObjectName(u"boutonparametre"); self.boutonparametre.setFixedSize(QSize(50, 50)); self.boutonparametre.setToolTip("Ouvrir les paramètres d'affichage"); self.boutonparametre.setText("⚙️"); self.boutonparametre.setFont(QFont("Segoe UI Emoji", 16))
        self.boutonparametre.setStyleSheet("QPushButton {border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white;} QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); }")
        self.gridLayout_top_toolbar.addWidget(self.boutonparametre, 0, 0, Qt.AlignmentFlag.AlignLeft)
        self.logo = QLabel(parent); self.logo.setObjectName(u"logo"); self.logo.setText("Traduction LSF"); self.logo.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: transparent;"); self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_top_toolbar.addWidget(self.logo, 0, 1); spacerItem = QSpacerItem(50, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum); self.gridLayout_top_toolbar.addItem(spacerItem, 0, 2); self.gridLayout_top_toolbar.setColumnStretch(1, 1)
    def setup_camera_view(self, parent):
        self.frame_camera = QFrame(parent); self.frame_camera.setObjectName(u"frame_camera"); sizePolicyCamFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.frame_camera.setSizePolicy(sizePolicyCamFrame); self.frame_camera.setMinimumHeight(300)
        self.frame_camera.setStyleSheet("QFrame#frame_camera { border: 1px solid gray; border-radius: 5px; background-color: black; }"); gridLayout_camera_inner = QGridLayout(self.frame_camera); gridLayout_camera_inner.setObjectName(u"gridLayout_camera_inner"); gridLayout_camera_inner.setContentsMargins(1, 1, 1, 1)
        self.camera_view = QLabel(self.frame_camera); self.camera_view.setObjectName(u"camera_view"); sizePolicyCamLabel = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored); self.camera_view.setSizePolicy(sizePolicyCamLabel)
        self.camera_view.setStyleSheet(u"QLabel#camera_view { background-color: transparent; border: none; color: grey; font-size: 16pt; }"); self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter); self.camera_view.setScaledContents(False); gridLayout_camera_inner.addWidget(self.camera_view, 0, 0, 1, 1)
    def setup_text_area(self, parent):
        self.frame_text = QFrame(parent); self.frame_text.setObjectName(u"frame_text"); sizePolicyTextFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.frame_text.setSizePolicy(sizePolicyTextFrame); self.frame_text.setMinimumHeight(100); self.frame_text.setMaximumHeight(250)
        self.frame_text.setStyleSheet("QFrame#frame_text { background-color: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px; padding: 5px; }"); layout_inside_frame = QVBoxLayout(self.frame_text); layout_inside_frame.setContentsMargins(5, 5, 5, 5); layout_inside_frame.setSpacing(5)
        self.label_predictions = QLabel("Prédictions:", self.frame_text); self.label_predictions.setObjectName(u"label_predictions"); font_pred = QFont(); font_pred.setPointSize(11); font_pred.setBold(True); self.label_predictions.setFont(font_pred)
        self.label_predictions.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop); self.label_predictions.setStyleSheet("background-color: transparent; color: white;"); layout_inside_frame.addWidget(self.label_predictions)
        self.textEdit = QTextEdit(self.frame_text); self.textEdit.setObjectName(u"textEdit"); self.textEdit.setStyleSheet(u"QTextEdit { font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: white; }"); self.textEdit.setReadOnly(True); layout_inside_frame.addWidget(self.textEdit, 1)
    def setup_export_controls(self, parent):
        self.horizontalLayout_export = QHBoxLayout(); self.horizontalLayout_export.setObjectName(u"horizontalLayout_export"); self.verticalLayout_export = QVBoxLayout(); self.verticalLayout_export.setObjectName(u"verticalLayout_export")
        self.exportation = QPushButton(parent); self.exportation.setObjectName(u"exportation"); self.exportation.setFixedSize(QSize(50, 50)); self.exportation.setText("💾"); self.exportation.setFont(QFont("Segoe UI Emoji", 16)); self.exportation.setToolTip("Exporter le texte (Non implémenté)"); self.exportation.setEnabled(False)
        self.exportation.setStyleSheet("QPushButton { border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white; border: 1px solid rgba(255, 255, 255, 0.3); } QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); } QPushButton:disabled { background-color: rgba(128, 128, 128, 0.2); color: gray; border-color: rgba(128, 128, 128, 0.4); }")
        self.verticalLayout_export.addWidget(self.exportation); self.horizontalLayout_export.addLayout(self.verticalLayout_export)
    def setup_menu_statusbar(self, MainWindow):
        self.statusbar = QStatusBar(MainWindow); self.statusbar.setObjectName(u"statusbar"); self.statusbar.setStyleSheet("QStatusBar { color: #DDDDDD; padding-left: 5px; background-color: transparent; }"); MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow); self.menubar.setObjectName(u"menubar"); MainWindow.setMenuBar(self.menubar) # Menu bar can be empty
    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        if hasattr(self, 'boutonparametre'): self.boutonparametre.setToolTip(_translate("MainWindow", u"Ouvrir les paramètres d'affichage", None))
        if hasattr(self, 'exportation'): self.exportation.setToolTip(_translate("MainWindow", u"Exporter le texte (Non implémenté)", None))
        if hasattr(self, 'logo') and self.logo.text() == "": self.logo.setText(_translate("MainWindow", u"Traduction LSF", None))
        if hasattr(self, 'camera_view'): self.camera_view.setText(_translate("MainWindow", u"Initialisation...", None))
        if hasattr(self, 'textEdit'): self.textEdit.setPlaceholderText(_translate("MainWindow", u"Les mots prédits apparaîtront ici...", None))
        if hasattr(self, 'label_predictions'): self.label_predictions.setText(_translate("MainWindow", u"Prédictions :", None))

# --- MainWindow Class (Application Logic) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow(); self.ui.setupUi(self); print("DEBUG (MainWindow.__init__): UI setup complete.")
        self.parameters_window = None; self.ui_parameters = None; self.ui.boutonparametre.clicked.connect(self.open_parameters); print("DEBUG (MainWindow.__init__): Parameters button connected.")
        self.current_text_color = self.ui.default_text_color; self.current_bg_color = self.ui.default_bg_color; self.apply_initial_styles(); print("DEBUG (MainWindow.__init__): Initial styles applied.")
        self.placeholder_timer = QTimer(self); self.placeholder_timer.timeout.connect(self.show_placeholder_frame); self.placeholder_frame_counter = 0; self.placeholder_active = True
        self.ui.camera_view.setText("Initialisation..."); self.ui.camera_view.setAlignment(Qt.AlignCenter); print("DEBUG (MainWindow.__init__): Placeholder timer setup.")
        self.video_thread = VideoThread(self); print("DEBUG (MainWindow.__init__): VideoThread instance created.")
        self.video_thread.frame_ready.connect(self.update_frame); self.video_thread.prediction_ready.connect(self.update_prediction)
        self.video_thread.top_n_ready.connect(self.update_top_n_status); self.video_thread.error_occurred.connect(self.handle_error)
        self.video_thread.models_loaded.connect(self.on_models_loaded); self.video_thread.finished.connect(self.on_thread_finished)
        self.video_thread.hands_detected_signal.connect(self.update_hand_detection_status); print("DEBUG (MainWindow.__init__): VideoThread signals connected.")
        
        # TTS specific initialization
        self.last_spoken_word = ""
        self.tts_thread = None
        if TTS_PYDUB_OK: # Only start TTS worker if pydub (for playback) is available
            print("DEBUG (MainWindow.__init__): Démarrage du thread worker TTS...")
            self.tts_thread = threading.Thread(target=tts_worker, daemon=True) # daemon=True allows main program to exit even if thread is running
            self.tts_thread.start()
        else:
            print("INFO (MainWindow.__init__): Thread worker TTS non démarré (Pydub indisponible).")

        self.ui.statusbar.showMessage("Initialisation: Chargement des modèles et de la caméra...")
        self.placeholder_timer.start(50); print("DEBUG (MainWindow.__init__): Placeholder timer démarré.")
        print("DEBUG (MainWindow.__init__): Démarrage VideoThread..."); self.video_thread.start()
    def apply_initial_styles(self):
         print("DEBUG (MainWindow.apply_initial_styles): Application styles initiaux."); bg_color_name = self.current_bg_color.name()
         style = f"QWidget#centralwidget {{ background-color: {bg_color_name}; }}"; self.ui.centralwidget.setStyleSheet(style); self.update_text_colors(self.current_text_color)
    @Slot()
    def show_placeholder_frame(self):
        if not self.placeholder_active: return
        try:
            label = self.ui.camera_view
            if not label or not label.isVisible() or label.width() <= 0 or label.height() <= 0: return
            w, h = label.width(), label.height(); pixmap = QPixmap(w, h); pixmap.fill(Qt.black); painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            center_x, center_y = w / 2.0, h / 2.0; max_radius = min(w, h) / 7.0; radius_variation = max_radius / 2.5
            pulse = (1 + np.sin(self.placeholder_frame_counter * 0.15)) / 2.0; current_radius = max_radius - (radius_variation * pulse)
            if current_radius > 0: painter.setBrush(QColor(50, 50, 50)); painter.setPen(Qt.NoPen); painter.drawEllipse(QPointF(center_x, center_y), current_radius, current_radius)
            font = QFont("Arial", 14); painter.setFont(font); painter.setPen(QColor(180, 180, 180)); text = "En attente de la caméra..."
            if self.video_thread and self.video_thread.isFinished():
                 current_status = self.ui.statusbar.currentMessage(); text = "Échec initialisation / Erreur" if "ERREUR" in current_status or "ÉCHEC" in current_status else "Caméra déconnectée"
            painter.drawText(pixmap.rect(), Qt.AlignCenter, text); painter.end(); label.setPixmap(pixmap); self.placeholder_frame_counter += 1
        except Exception as e: print(f"ERREUR (MainWindow.show_placeholder_frame): {e}"); self.placeholder_timer.stop(); self.placeholder_active = False; self.ui.camera_view.setText(f"Erreur Placeholder:\n{e}")
    @Slot(np.ndarray)
    def update_frame(self, cv_img):
        if self.placeholder_active: print("DEBUG (MainWindow.update_frame): Première frame réelle reçue, arrêt placeholder."); self.placeholder_timer.stop(); self.placeholder_active = False; self.ui.camera_view.clear(); self.ui.camera_view.setText("")
        if cv_img is None or cv_img.size == 0: print("DEBUG (MainWindow.update_frame): Frame vide/invalide reçue, ignorée."); return
        try:
            h, w, ch = cv_img.shape; bytes_per_line = ch * w; qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            if qt_image.isNull(): print("ERREUR DEBUG (MainWindow.update_frame): Création QImage échouée!"); self.ui.camera_view.setText("Erreur: QImage Nulle"); return
            qt_pixmap = QPixmap.fromImage(qt_image)
            if qt_pixmap.isNull(): print("ERREUR DEBUG (MainWindow.update_frame): Conversion QPixmap échouée!"); self.ui.camera_view.setText("Erreur: QPixmap Nulle"); return
            label, label_size = self.ui.camera_view, self.ui.camera_view.size()
            if label_size.isValid() and label_size.width() > 10 and label_size.height() > 10:
                scaled_pixmap = qt_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap if not scaled_pixmap.isNull() else qt_pixmap) # Fallback to unscaled if scaling fails
            else: label.setPixmap(qt_pixmap)
        except cv2.error as e_cv: logging.error(f"Erreur OpenCV dans update_frame: {e_cv}", exc_info=True); print(f"DEBUG (MainWindow.update_frame): Erreur OpenCV: {e_cv}"); self.ui.camera_view.setText(f"Erreur OpenCV:\n{e_cv}"); self._activate_placeholder()
        except Exception as e: logging.error(f"Erreur màj frame: {e}", exc_info=True); print(f"DEBUG (MainWindow.update_frame): Exception: {e}"); self.ui.camera_view.setText(f"Erreur affichage:\n{str(e)}"); self._activate_placeholder()
    def _activate_placeholder(self): # Helper to avoid code duplication
        if not self.placeholder_active: self.placeholder_active = True; self.placeholder_timer.start(50)
    @Slot(str)
    def update_prediction(self, text):
        text_to_display = text # Use a different variable for display logic
        word_to_speak = None

        if text_to_display and text_to_display != "?":
            current_content = self.ui.textEdit.toPlainText(); words = current_content.split()
            if not words or text_to_display.lower() != words[-1].lower():
                 max_words_history = 50; new_words = words + [text_to_display]
                 if len(new_words) > max_words_history: new_words = new_words[-max_words_history:]
                 self.ui.textEdit.setPlainText(" ".join(new_words)); self.ui.textEdit.moveCursor(QTextCursor.MoveOperation.End)
                 # Only set word_to_speak if it's a new word for the display
                 if text_to_display.lower() != self.last_spoken_word.lower():
                     word_to_speak = text_to_display
        
        if word_to_speak and TTS_PYDUB_OK: # Check if TTS is possible and there's a new word
            print(f"DEBUG (MainWindow.update_prediction): Ajout de '{word_to_speak}' à la file d'attente TTS (voix fr_male, vitesse 1.0).")
            if add_speech_request(word_to_speak, "fr_male", 1.0): # fr_male for API, fr_female for gTTS
                 self.last_spoken_word = word_to_speak # Update last spoken word *only if successfully queued*
            else:
                 print(f"DEBUG (MainWindow.update_prediction): Échec de l'ajout de '{word_to_speak}' à la file d'attente TTS.")
    @Slot(list)
    def update_top_n_status(self, top_n_list):
        filtered_list = [item for item in top_n_list if item and item.strip()]
        if filtered_list: self.ui.statusbar.showMessage(" | ".join(filtered_list[:self.video_thread.ui_TOP_N]))
        else:
            current_message = self.ui.statusbar.currentMessage()
            if not current_message or ("ERREUR" not in current_message and "ÉCHEC" not in current_message): self.ui.statusbar.showMessage("Prêt.", 3000)
    @Slot(bool)
    def update_hand_detection_status(self, detected): pass # Placeholder, can be used to change UI based on hand detection
    @Slot(str)
    def handle_error(self, message):
        logging.error(f"Erreur signalée par thread vidéo: {message}"); print(f"DEBUG (MainWindow.handle_error): Erreur reçue: {message}")
        critical_keywords = ["impossible d'ouvrir", "webcam", "fichier modèle", "vocabulaire", "shape lstm", "erreur chargement cnn", "erreur chargement lstm", "erreur ouverture webcam", "erreur lecture webcam", "mediapipe", "config gpu", "critical", "manquant", "not found", "empty", "failed to initialize", "attributeerror", "indexerror", "valueerror", "runtimeerror"]
        is_critical = any(keyword in message.lower() for keyword in critical_keywords)
        if is_critical:
             print(f"DEBUG (MainWindow.handle_error): Erreur critique identifiée: {message}"); self.ui.statusbar.showMessage(f"ERREUR CRITIQUE: {message}", 0); self.ui.camera_view.setText(f"ERREUR CRITIQUE:\n{message}\nVérifiez console/logs.")
             self._activate_placeholder(); QMessageBox.critical(self, "Erreur Critique", f"Erreur critique:\n\n{message}\n\nLa traduction risque de ne pas fonctionner.\nVérifiez configuration et logs.")
        else: self.ui.statusbar.showMessage(f"Erreur: {message}", 10000)
    @Slot(bool)
    def on_models_loaded(self, success):
        print(f"DEBUG (MainWindow.on_models_loaded): Signal models_loaded reçu succès={success}")
        if success: self.ui.statusbar.showMessage("Modèles chargés. Démarrage capture webcam...", 5000)
        else:
            final_fail_msg = "ÉCHEC INITIALISATION: Modèles/Vocab/Webcam. Vérifiez logs."; self.ui.statusbar.showMessage(final_fail_msg, 0); self.ui.camera_view.setText(f"ÉCHEC INITIALISATION:\nVérifiez console/logs.")
            print(f"DEBUG (MainWindow.on_models_loaded): Initialisation échouée (models_loaded=False reçu)"); self._activate_placeholder()
    @Slot()
    def on_thread_finished(self):
        logging.info("Thread traitement vidéo terminé."); print("DEBUG (MainWindow.on_thread_finished): Signal fin thread vidéo reçu.")
        current_status = self.ui.statusbar.currentMessage()
        if "ERREUR" not in current_status and "ÉCHEC" not in current_status: self.ui.statusbar.showMessage("Connexion caméra terminée.", 5000)
        self._activate_placeholder()
        self.ui.camera_view.setText("Caméra déconnectée" + (" / Erreur" if "ERREUR" in current_status or "ÉCHEC" in current_status else ""))
    def open_parameters(self):
        print("DEBUG (MainWindow.open_parameters): Ouverture fenêtre paramètres...")
        if self.parameters_window is None:
            self.parameters_window = QWidget(self, Qt.WindowType.Window); self.parameters_window.setWindowModality(Qt.WindowModality.NonModal); self.ui_parameters = Ui_ParametersWindow(); self.ui_parameters.setupUi(self.parameters_window)
            if self.ui_parameters:
                self.ui_parameters.text_color_preview.setStyleSheet(f"background-color: {self.current_text_color.name()}; border: 1px solid black; border-radius: 3px;"); self.ui_parameters.bg_color_preview.setStyleSheet(f"background-color: {self.current_bg_color.name()}; border: 1px solid black; border-radius: 3px;")
                self.ui_parameters.color_changed.connect(self.update_text_colors); self.ui_parameters.bg_color_changed.connect(self.update_bg_color); print("DEBUG (MainWindow.open_parameters): Fenêtre paramètres créée et signaux connectés.")
            else: print("ERREUR: Échec setup UI fenêtre paramètres."); self.parameters_window = None; return
        else: print("DEBUG (MainWindow.open_parameters): Fenêtre paramètres existe déjà.")
        self.parameters_window.show(); self.parameters_window.raise_(); self.parameters_window.activateWindow()
    @Slot(QColor)
    def update_text_colors(self, color):
        if not color.isValid(): return
        self.current_text_color = color; color_name = color.name(); print(f"DEBUG (MainWindow.update_text_colors): Application couleur texte: {color_name}")
        text_style = f"color: {color_name}; background-color: transparent;"; bold_text_style = f"color: {color_name}; background-color: transparent; font-weight: bold;"
        if hasattr(self.ui, 'label_predictions'): self.ui.label_predictions.setStyleSheet(bold_text_style)
        if hasattr(self.ui, 'logo'): self.ui.logo.setStyleSheet(f"font-size: 24px; {bold_text_style}")
        if hasattr(self.ui, 'statusbar'): self.ui.statusbar.setStyleSheet(f"QStatusBar {{ color: {color.lighter(110).name()}; padding-left: 5px; background-color: transparent; }}") # Lighter for status bar
        if hasattr(self.ui, 'textEdit'): self.ui.textEdit.setStyleSheet(f"QTextEdit {{ font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: {color_name}; }}")
    @Slot(QColor)
    def update_bg_color(self, color):
        if not color.isValid(): return
        self.current_bg_color = color; color_name = color.name(); print(f"DEBUG (MainWindow.update_bg_color): Application couleur fond: {color_name}")
        self.ui.centralwidget.setStyleSheet(f"QWidget#centralwidget {{ background-color: {color_name}; }}"); self.update_text_colors(self.current_text_color)
    def closeEvent(self, event):
        logging.info("Fermeture fenêtre principale demandée."); print("DEBUG (MainWindow.closeEvent): Événement fermeture déclenché.")
        if self.placeholder_timer.isActive(): print("DEBUG: Arrêt placeholder timer."); self.placeholder_timer.stop()
        if self.parameters_window and self.parameters_window.isVisible(): print("DEBUG: Fermeture fenêtre paramètres."); self.parameters_window.close()
        if self.video_thread and self.video_thread.isRunning():
            logging.info("Arrêt thread vidéo..."); print("DEBUG: Signal d'arrêt au thread vidéo...")
            self.video_thread.stop(); print("DEBUG: Attente thread vidéo...")
            if not self.video_thread.wait(3000): logging.warning("Timeout thread vidéo."); print("DEBUG: Timeout thread vidéo.")
            else: logging.info("Thread vidéo arrêté."); print("DEBUG: Thread vidéo arrêté.")
        else: logging.info("Thread vidéo non en cours."); print("DEBUG: Thread vidéo non en cours.")
        
        # Shutdown TTS worker
        if TTS_PYDUB_OK and self.tts_thread and self.tts_thread.is_alive():
            print("DEBUG (MainWindow.closeEvent): Signal d'arrêt au worker TTS...")
            stop_worker_event.set()
            print("DEBUG (MainWindow.closeEvent): Attente de la fin de la file d'attente TTS...")
            tts_queue.join() # Wait for all queued items to be processed
            print("DEBUG (MainWindow.closeEvent): Attente de la fin du thread TTS...")
            self.tts_thread.join(timeout=5) # Wait for the thread to actually finish
            if self.tts_thread.is_alive(): print("ATTENTION (MainWindow.closeEvent): Thread TTS non terminé à temps.")
            else: print("DEBUG (MainWindow.closeEvent): Thread TTS terminé.")

        print("DEBUG (MainWindow.closeEvent): Acceptation événement fermeture."); event.accept()

# --- Application Entry Point ---
if __name__ == "__main__":
     print("DEBUG: Démarrage application depuis mainwindow.py")

     # Setup for TTS config file (data/config.json)
     script_dir_main = os.path.dirname(os.path.abspath(__file__))
     data_dir_main = os.path.join(script_dir_main, 'data')
     config_file_tts_main = os.path.join(data_dir_main, 'config.json')

     if not os.path.exists(data_dir_main):
         try:
             os.makedirs(data_dir_main)
             print(f"DEBUG: Répertoire 'data' créé : {data_dir_main}")
         except OSError as e:
             print(f"ERREUR: lors de la création du répertoire 'data' {data_dir_main}: {e}")
     
     if not os.path.exists(config_file_tts_main):
         print(f"ATTENTION: config.json pour TTS non trouvé à {config_file_tts_main}. Création d'un fichier par défaut.")
         print("Veuillez éditer 'data/config.json' avec vos vrais endpoints API TikTok si vous utilisez l'API TTS.")
         dummy_config_tts_content = [
             {"name": "TikTok Default", "url": "https://tiktok-tts.weilnet.workers.dev/api/generation", "response": "data"},
             # {"name": "Example API", "url": "https://example.com/api/tts", "response": "audio_base64"}
         ]
         try:
             with open(config_file_tts_main, 'w', encoding='utf-8') as f: # Added encoding
                 json_dump(dummy_config_tts_content, f, indent=4)
             print(f"DEBUG: Fichier config.json TTS par défaut créé à {config_file_tts_main}")
         except Exception as e:
              print(f"ERREUR: lors de la création du fichier config.json TTS par défaut : {e}")
     else:
         print(f"DEBUG: Fichier config.json TTS existant trouvé à {config_file_tts_main}")

     app = QApplication(sys.argv)
     window = MainWindow()
     window.showMaximized() # Ou window.show()
     print("DEBUG: Entrée boucle événements Qt application...")
     exit_code = app.exec()
     print(f"DEBUG: Boucle événements application terminée avec code sortie {exit_code}.")
     sys.exit(exit_code)
