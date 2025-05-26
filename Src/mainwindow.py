# -*- coding: utf-8 -*-
# mainwindow.py

# --- PyQt Imports ---
import sys
print("DEBUG: Importing sys")
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale, # type: ignore
                            QMetaObject, QObject, QPoint, QPointF, QRect,
                            QSize, QTime, QUrl, Qt, Signal, QThread, Slot, QTimer, QStandardPaths) # AJOUTÉ QStandardPaths et QDateTime
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
                             QWidget, QMessageBox, QButtonGroup, QSlider,
                             QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog) # AJOUTÉ QFileDialog
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
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
print("DEBUG: Keras imports complete.")

import os
import logging
import time
from collections import deque, Counter

print("DEBUG: Standard library imports complete.")

# --- Import Config (pour la partie vidéo) ---
print("DEBUG: Importing config...")
try:
    import config
    print("DEBUG: config imported successfully.")
except ImportError:
    print("ERREUR: Impossible d'importer config.py (pour VideoThread). Assurez-vous qu'il existe.")
    config = None
except Exception as e:
    print(f"ERREUR: Problème lors de l'import de config.py: {e}")
    config = None

# --- Imports pour le Text-To-Speech (TTS) ---
print("DEBUG: Importing TTS modules...")
import requests
import base64
import re
import queue as py_queue
import threading
import tempfile
from json import load as json_load, dump as json_dump
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    print("DEBUG: gTTS imported successfully.")
except ImportError:
    print("AVERTISSEMENT: Le module 'gTTS' n'est pas installé. La voix féminine française TTS locale sera désactivée.")
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


# --- Définitions TTS ---
class Voice(Enum):
    FR_MALE_API = "fr_002"

VOICE_KEY_FR_FEMALE_GTTS = "fr_female_gtts"

API_VOICE_OPTIONS = {
    "Masculine (API TikTok)": Voice.FR_MALE_API.value,
}
ALL_VOICE_OPTIONS_MAP = {
    "Masculine (API TikTok)": Voice.FR_MALE_API.value,
}
if GTTS_AVAILABLE:
    ALL_VOICE_OPTIONS_MAP["Féminine (Locale gTTS)"] = VOICE_KEY_FR_FEMALE_GTTS

TTS_PYDUB_OK = PYDUB_AVAILABLE
TTS_GTTS_OK = GTTS_AVAILABLE

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
    character_limit: int = 300
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
            if current_chunk: merged_chunks.append(current_chunk)
            if len(separated_chunk.encode("utf-8")) <= character_limit:
                current_chunk = separated_chunk
            else:
                print(f"ATTENTION (TTS): Segment trop long: {separated_chunk[:50]}...")
                merged_chunks.append(separated_chunk); current_chunk = ""
    if current_chunk: merged_chunks.append(current_chunk)
    return [chunk for chunk in merged_chunks if chunk and not chunk.isspace()]

def _fetch_audio_bytes_from_api(endpoint: Dict[str, str], text_chunk: str, voice_id_value: str) -> Optional[str]:
    try:
        api_url, response_key = endpoint.get("url"), endpoint.get("response")
        if not api_url or not response_key: print(f"ERREUR (TTS API): Config endpoint invalide: {endpoint}"); return None
        response = requests.post(api_url, json={"text": text_chunk, "voice": voice_id_value}, timeout=15)
        response.raise_for_status(); json_response = response.json()
        if response_key in json_response: return json_response[response_key]
        else: print(f"  [TTS API] ERREUR: Clé '{response_key}' absente: {json_response}"); return None
    except requests.exceptions.Timeout: print(f"  [TTS API] ERREUR: Timeout: {text_chunk[:30]}..."); return None
    except requests.exceptions.RequestException as e: print(f"  [TTS API] ERREUR requête: {e}"); return None
    except Exception as e: print(f"  [TTS API] ERREUR inattendue fetch: {e}"); return None

def generate_api_audio(text: str, output_file_path: str, voice_id_value: str) -> bool:
    if not text or text.isspace(): print(f"[TTS API] ERREUR: Texte vide"); return False
    endpoint_data = _load_endpoints()
    if not endpoint_data: print("[TTS API] ERREUR: Pas d'endpoints."); return False

    for endpoint in endpoint_data:
        text_chunks: List[str] = _split_text(text)
        if not text_chunks: print("[TTS API] ERREUR: Segments vides."); continue

        audio_chunks_b64: List[Optional[str]] = [None] * len(text_chunks)
        threads: List[threading.Thread] = []; results_lock = threading.Lock(); results: Dict[int, Optional[str]] = {}
        def thread_target(index: int, chunk: str):
            audio_data = _fetch_audio_bytes_from_api(endpoint, chunk, voice_id_value)
            with results_lock: results[index] = audio_data
        for i, chunk in enumerate(text_chunks):
            thread = threading.Thread(target=thread_target, args=(i, chunk)); threads.append(thread); thread.start()
        for thread in threads: thread.join()
        for i in range(len(text_chunks)): audio_chunks_b64[i] = results.get(i)

        if all(chunk is not None for chunk in audio_chunks_b64):
            try:
                full_audio_b64 = "".join([chunk for chunk in audio_chunks_b64 if chunk is not None])
                audio_bytes = base64.b64decode(full_audio_b64)
                with open(output_file_path, "wb") as file: file.write(audio_bytes)
                return True
            except Exception as e: print(f"[TTS API] ERREUR sauvegarde/décodage: {e}")
            return False
    print("[TTS API] ERREUR: Échec génération audio avec tous endpoints."); return False

def generate_gtts_female_audio(text: str, output_file_path: str) -> bool:
    if not TTS_GTTS_OK: print("[TTS gTTS] ERREUR: gTTS non dispo."); return False
    try:
        if not text or text.isspace(): print(f"[TTS gTTS] ERREUR: Texte vide"); return False
        tts = gTTS(text=text, lang='fr', slow=False); tts.save(output_file_path)
        return True
    except Exception as e: print(f"[TTS gTTS] ERREUR génération/sauvegarde: {e}"); return False

def apply_volume_and_speed_change(sound: AudioSegment, speed_factor: float = 1.0, volume_db_change: float = 0.0) -> AudioSegment:
    if not TTS_PYDUB_OK: return sound
    if volume_db_change != 0.0:
        sound = sound + volume_db_change
    if speed_factor == 1.0: return sound
    try:
        new_frame_rate = int(sound.frame_rate * speed_factor)
        if new_frame_rate <= 0: print(f"  [TTS Playback] ATTENTION: Taux échant. invalide ({new_frame_rate})."); return sound
        return sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
    except Exception as e: print(f"  [TTS Playback] ERREUR changement vitesse: {e}."); return sound

def play_audio_file_processed(file_path: str, speed_to_use: float, volume_db_change: float):
    if not TTS_PYDUB_OK: print("[TTS Playback] Pydub non dispo. Lecture sautée."); return
    final_audio_to_play = None
    try:
        file_extension = os.path.splitext(file_path)[1].lower().strip('.') or "mp3"
        audio = AudioSegment.from_file(file_path, format=file_extension)
        
        processed_audio = apply_volume_and_speed_change(audio, speed_to_use, volume_db_change)

        target_frame_rate = 44100
        if int(processed_audio.frame_rate) != target_frame_rate and (speed_to_use != 1.0 or processed_audio.frame_rate < 22050):
            final_audio_to_play = processed_audio.set_frame_rate(target_frame_rate)
        else:
            final_audio_to_play = processed_audio
        
        if final_audio_to_play is None: print(f"[TTS Playback] ERREUR: audio final est None."); return
        pydub_play(final_audio_to_play)
    except FileNotFoundError: print(f"[TTS Playback] ERREUR: Fichier non trouvé {file_path}")
    except Exception as e:
        print(f"[TTS Playback] ERREUR lecture {file_path}: {e}")
        if "ffprobe" in str(e) or "ffmpeg" in str(e): print("[TTS Playback] ERREUR: ffmpeg/ffprobe non trouvé (PATH?).")

tts_request_queue: py_queue.Queue = py_queue.Queue()
stop_tts_worker_event = threading.Event()

def tts_worker_thread_func():
    print("[TTS Worker] Thread démarré.")
    while not stop_tts_worker_event.is_set():
        try:
            text, voice_key, speed, volume_db = tts_request_queue.get(timeout=1.0)
            print(f"\n[TTS Worker] Traitement: Voix='{voice_key}', Vit={speed:.2f}x, Vol={volume_db:.1f}dB, Txt='{text[:30]}...'")

            if not TTS_PYDUB_OK: tts_request_queue.task_done(); continue
            success = False; temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_f: temp_file_path = tmp_f.name
                if voice_key == VOICE_KEY_FR_FEMALE_GTTS:
                    success = generate_gtts_female_audio(text, temp_file_path)
                else:
                    success = generate_api_audio(text, temp_file_path, voice_key)
                
                if success and temp_file_path and os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                    play_audio_file_processed(temp_file_path, speed, volume_db)
                elif not success: print(f"[TTS Worker] Échec génération audio.")
                elif temp_file_path and os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) == 0:
                    print(f"[TTS Worker] ATTENTION: Fichier audio généré vide: {temp_file_path}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try: os.remove(temp_file_path)
                    except OSError as e_rem: print(f"[TTS Worker] ERREUR suppression temp: {e_rem}")
            tts_request_queue.task_done()
        except py_queue.Empty: continue
        except Exception as e:
            print(f"[TTS Worker] ERREUR inattendue: {e}"); import traceback; traceback.print_exc()
            try: tts_request_queue.task_done()
            except ValueError: pass
            time.sleep(1)
    print("[TTS Worker] Thread terminé.")

def add_tts_request_to_queue(text: str, voice_id_value_or_gtts_key: str, speed_factor: float, volume_db_change: float) -> bool:
    if not TTS_PYDUB_OK: return False
    if not text or text.isspace(): print("[TTS Queue] ERREUR: Texte vide."); return False
    tts_request_queue.put((text, voice_id_value_or_gtts_key, speed_factor, volume_db_change))
    return True

class Colors: # From your first script
    CV_BLUE = (255, 0, 0)
    CV_GREEN = (0, 255, 0)
    CV_RED = (0, 0, 255)
    CV_WHITE = (255, 255, 255)
    CV_YELLOW = (0, 255, 255)
    CV_LIGHT_GREY = (211, 211, 211)

    try:
        MP_HAND_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils') else None
        MP_CONNECTION_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2) if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils') else None
    except Exception as e_colors_mp:
        print(f"WARN: Exception MediaPipe drawing specs: {e_colors_mp}"); MP_HAND_DRAWING_SPEC = None; MP_CONNECTION_DRAWING_SPEC = None

class VideoThread(QThread):
    frame_ready = Signal(np.ndarray); prediction_ready = Signal(str); top_n_ready = Signal(list)
    models_loaded = Signal(bool); error_occurred = Signal(str); hands_detected_signal = Signal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        print("DEBUG (VideoThread.__init__): Initializing...")
        self._running = False
        if config is None:
            self.error_occurred.emit("Fichier config.py (pour VideoThread) manquant ou erroné.")
            print("ERREUR (VideoThread.__init__): config.py non chargé. VideoThread ne fonctionnera pas.")
            self.MODEL_PATH, self.VOCABULARY_PATH, self.FIXED_LENGTH, self.FEATURE_DIM, \
            self.CNN_MODEL_CHOICE, self.CNN_INPUT_SHAPE, self.CAPTURE_SOURCE, self.FRAMES_TO_SKIP, \
            self.PREDICTION_THRESHOLD, self.SMOOTHING_WINDOW_SIZE, self.TOP_N, \
            self.MAX_FRAME_WIDTH, self.MIN_HAND_DETECTION_CONFIDENCE, \
            self.MIN_HAND_TRACKING_CONFIDENCE, self.MAX_HANDS, self.ui_TOP_N = (None,) * 17
            return

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
            error_msg = f"Erreur config.py: Attribut manquant '{e.name if hasattr(e, 'name') else str(e)}'"; print(f"ERREUR: {error_msg}");
            self.error_occurred.emit(error_msg)
            return

        self.cnn_feature_extractor_model = None; self.preprocess_function = None; self.cnn_target_size = None
        self.lstm_prediction_model = None; self.vocabulaire = None; self.index_to_word = None; self.cap = None
        self.mp_hands = None; self.hands_solution = None; self.mp_drawing = None
        self.drawing_spec_hand = Colors.MP_HAND_DRAWING_SPEC; self.drawing_spec_connection = Colors.MP_CONNECTION_DRAWING_SPEC
        self.last_hands_detected_status = False; print("DEBUG (VideoThread.__init__): Initialized variables.")

    def load_vocabulary(self):
        vocab = {};
        try:
            if not os.path.exists(self.VOCABULARY_PATH): raise FileNotFoundError(f"Fichier vocab non trouvé: '{self.VOCABULARY_PATH}'")
            with open(self.VOCABULARY_PATH, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or ':' not in line:
                        if line: logging.warning(f"L{line_num} format incorrect '{self.VOCABULARY_PATH}': '{line}'"); continue
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[0] and parts[1].isdigit(): vocab[parts[0].strip().lower()] = int(parts[1].strip())
            if not vocab: error_msg = f"Vocab chargé de '{self.VOCABULARY_PATH}' est vide."; logging.error(error_msg); self.error_occurred.emit(error_msg); return None
            logging.info(f"Vocab chargé ({len(vocab)} mots)."); return vocab
        except FileNotFoundError as e: self.error_occurred.emit(str(e)); logging.error(f"Erreur: {e}"); return None
        except Exception as e: self.error_occurred.emit(f"Erreur chargement vocab: {e}"); logging.exception(""); return None

    def load_models_and_preprocessing(self):
        model_name = self.CNN_MODEL_CHOICE; input_shape = self.CNN_INPUT_SHAPE; self.cnn_target_size = input_shape[:2]
        logging.info(f"Chargement CNN: {model_name} shape {input_shape}...");
        try:
            if model_name == 'MobileNetV2': base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet'); self.preprocess_function = mobilenet_preprocess
            elif model_name == 'ResNet50': base = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet'); self.preprocess_function = resnet_preprocess
            elif model_name == 'EfficientNetB0': base = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet'); self.preprocess_function = efficientnet_preprocess
            else: raise ValueError(f"Modèle CNN non supporté: {model_name}")
            output = GlobalAveragePooling2D()(base.output); self.cnn_feature_extractor_model = Model(inputs=base.input, outputs=output, name=f"{model_name}_FeatureExtractor")
            dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32); _ = self.cnn_feature_extractor_model(dummy_input, training=False)
            logging.info(f"CNN {model_name} chargé et initialisé.");
        except Exception as e: self.error_occurred.emit(f"Erreur chargement CNN '{model_name}': {e}"); logging.exception(""); return False
        
        logging.info(f"Chargement LSTM: {self.MODEL_PATH}...");
        try:
            if not os.path.exists(self.MODEL_PATH): raise FileNotFoundError(f"Modèle LSTM non trouvé: {self.MODEL_PATH}")
            self.lstm_prediction_model = tf.keras.models.load_model(self.MODEL_PATH); logging.info(f"LSTM chargé de {self.MODEL_PATH}")
            expected_lstm_shape = self.lstm_prediction_model.input_shape; logging.info(f"Shape LSTM attendue: {expected_lstm_shape}")
            if len(expected_lstm_shape) != 3: raise ValueError(f"Shape LSTM rang inattendu: {len(expected_lstm_shape)}")
            model_seq_len, model_feat_dim = expected_lstm_shape[1], expected_lstm_shape[2]
            if model_seq_len is not None and model_seq_len != self.FIXED_LENGTH: logging.warning(f"Avert. Longueur Séquence LSTM! Modèle={model_seq_len}, Config={self.FIXED_LENGTH}.")
            if model_feat_dim is not None and model_feat_dim != self.FEATURE_DIM: raise ValueError(f"CRITIQUE Dim Features LSTM! Modèle={model_feat_dim}, Config={self.FEATURE_DIM}.")
            dummy_lstm_input = tf.zeros((1, self.FIXED_LENGTH, self.FEATURE_DIM), dtype=tf.float32); _ = self.lstm_prediction_model(dummy_lstm_input, training=False); logging.info("LSTM initialisé.")
        except Exception as e: self.error_occurred.emit(f"Erreur chargement/init LSTM: {e}"); logging.exception(""); return False
        return True

    def extract_cnn_features_realtime(self, frame):
        if not all([self.cnn_feature_extractor_model, self.preprocess_function, self.cnn_target_size]):
            logging.error("Extracteur CNN non initialisé (extract_cnn_features_realtime)."); return None
        try:
            target_size_cv2 = (self.cnn_target_size[1], self.cnn_target_size[0]); img_resized_cv = cv2.resize(frame, target_size_cv2, interpolation=cv2.INTER_AREA)
            img_resized_tensor = tf.convert_to_tensor(img_resized_cv, dtype=tf.float32); img_batch_tensor = tf.expand_dims(img_resized_tensor, axis=0)
            img_preprocessed_tensor = self.preprocess_function(img_batch_tensor); features_tensor = self.cnn_feature_extractor_model(img_preprocessed_tensor, training=False)
            return features_tensor[0].numpy()
        except Exception as e: logging.warning(f"Erreur extraction features CNN: {e}", exc_info=False); return None

    def run(self):
        if config is None or not hasattr(self, 'MODEL_PATH') or self.MODEL_PATH is None :
             print("ERREUR (VideoThread.run): VideoThread n'a pas pu s'initialiser correctement. Arrêt.")
             self.models_loaded.emit(False)
             self._running = False
             return

        self._running = True; logging.info("Thread traitement vidéo démarré.");
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configurés croissance mémoire: {gpus}")
            except RuntimeError as e: logging.error(f"Erreur config GPU: {e}")
        else: logging.warning("Aucun GPU détecté par TensorFlow.")

        if not self.load_models_and_preprocessing(): self.models_loaded.emit(False); self._running = False; return
        self.vocabulaire = self.load_vocabulary()
        if not self.vocabulaire: self.models_loaded.emit(False); self._running = False; return
        try:
            self.index_to_word = {i: word for word, i in self.vocabulaire.items()}
            if len(self.index_to_word) != len(self.vocabulaire): logging.warning("Indices dupliqués potentiels vocab.")
            logging.info(f"Vocab inversé créé ({len(self.index_to_word)} entrées).")
        except Exception as e: self.error_occurred.emit(f"Erreur création vocab inversé: {e}"); self.models_loaded.emit(False); self._running = False; return

        if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            self.mp_hands = mp.solutions.hands; self.mp_drawing = mp.solutions.drawing_utils
            try:
                if self.drawing_spec_hand is None: self.drawing_spec_hand = self.mp_drawing.DrawingSpec(color=Colors.CV_GREEN, thickness=2, circle_radius=2)
                if self.drawing_spec_connection is None: self.drawing_spec_connection = self.mp_drawing.DrawingSpec(color=Colors.CV_RED, thickness=2)
                self.hands_solution = self.mp_hands.Hands(static_image_mode=False, max_num_hands=self.MAX_HANDS, min_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE, min_tracking_confidence=self.MIN_HAND_TRACKING_CONFIDENCE)
                logging.info(f"Mediapipe Hands initialisé.")
            except Exception as e_mp: self.error_occurred.emit(f"Erreur init Mediapipe: {e_mp}"); self.hands_solution = None
        else: logging.warning("Mediapipe Hands non dispo."); self.hands_solution = None

        self.models_loaded.emit(True)
        
        logging.info(f"Ouverture caméra: {self.CAPTURE_SOURCE}"); self.cap = None; backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        try:
            src = int(self.CAPTURE_SOURCE) if str(self.CAPTURE_SOURCE).isdigit() else self.CAPTURE_SOURCE
            self.cap = cv2.VideoCapture(src, backend); time.sleep(0.5)
            is_opened = self.cap.isOpened() if self.cap else False
            if not is_opened and sys.platform == "win32" and backend == cv2.CAP_DSHOW:
                 if self.cap: self.cap.release(); self.cap = cv2.VideoCapture(src, cv2.CAP_ANY); time.sleep(0.5)
                 is_opened = self.cap.isOpened() if self.cap else False
            if not is_opened: raise IOError(f"Impossible d'ouvrir caméra '{src}'")
        except Exception as e_cap:
             self.error_occurred.emit(f"Erreur ouverture webcam {self.CAPTURE_SOURCE}: {e_cap}"); self.models_loaded.emit(False); self._running = False
             if self.cap: self.cap.release();
             if self.hands_solution: self.hands_solution.close();
             return
        logging.info("Webcam ouverte.");

        sequence_window = deque(maxlen=self.FIXED_LENGTH); prediction_buffer = deque(maxlen=self.SMOOTHING_WINDOW_SIZE)
        frame_proc_times = deque(maxlen=30); frame_count = 0; last_smoothed_word = "?"
        
        target_w, target_h, resize_disp = None, None, False
        try:
            cam_w, cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if cam_w <= 0 or cam_h <= 0: raise ValueError("Dims caméra invalides")
            logging.info(f"Résolution native: {cam_w}x{cam_h}")
            if self.MAX_FRAME_WIDTH and cam_w > self.MAX_FRAME_WIDTH:
                scale = self.MAX_FRAME_WIDTH / cam_w; target_w = self.MAX_FRAME_WIDTH; target_h = int(cam_h * scale)
                target_h = target_h + 1 if target_h % 2 != 0 else target_h; resize_disp = True
                logging.info(f"Redim. affichage: {target_w}x{target_h}")
            else: target_w, target_h = cam_w, cam_h
        except Exception as e_res: logging.warning(f"Pb résolution caméra: {e_res}. Fallback."); target_w, target_h, resize_disp = 640, 480, True

        logging.info("Entrée boucle vidéo principale...")
        while self._running:
            loop_start_t = time.time();
            try: ret, frame = self.cap.read()
            except Exception as e_read: self.error_occurred.emit(f"Erreur lecture webcam: {e_read}"); logging.error(f"Exception cap.read(): {e_read}"); break
            if not ret or frame is None: self.error_occurred.emit(f"Impossible lire frame. Caméra ouverte: {self.cap.isOpened() if self.cap else False}"); logging.error("Échec lecture frame."); break
            frame_count += 1
            
            display_frame = cv2.resize(frame, (target_w, target_h), cv2.INTER_LINEAR) if resize_disp else frame.copy()

            hands_detected = False
            if self.hands_solution:
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img_rgb.flags.writeable = False
                    results = self.hands_solution.process(img_rgb)
                    if results.multi_hand_landmarks:
                        hands_detected = True
                        for hand_lm in results.multi_hand_landmarks:
                            if self.mp_drawing and self.drawing_spec_hand and self.drawing_spec_connection:
                                self.mp_drawing.draw_landmarks(display_frame, hand_lm, self.mp_hands.HAND_CONNECTIONS, self.drawing_spec_hand, self.drawing_spec_connection)
                except Exception as e_hand: logging.warning(f"Erreur Mediapipe hand: {e_hand}", exc_info=False)
            if hands_detected != self.last_hands_detected_status:
                if self._running: self.hands_detected_signal.emit(hands_detected)
                self.last_hands_detected_status = hands_detected

            run_inference = ((not self.hands_solution) or hands_detected) and (frame_count % (self.FRAMES_TO_SKIP + 1) == 0)
            if run_inference:
                inf_start_t = time.time(); cnn_feat = self.extract_cnn_features_realtime(frame)
                frame_proc_times.append((time.time() - inf_start_t) * 1000)
                if cnn_feat is not None:
                    sequence_window.append(cnn_feat); current_seq_len = len(sequence_window)
                    if current_seq_len > 0:
                        current_seq_np = np.array(sequence_window, dtype=np.float32)
                        pad_seq = np.pad(current_seq_np, ((self.FIXED_LENGTH - current_seq_len, 0), (0,0)), 'constant', constant_values=0.0) if current_seq_len < self.FIXED_LENGTH else current_seq_np
                        
                        if pad_seq.shape == (self.FIXED_LENGTH, self.FEATURE_DIM):
                            reshaped_seq = np.expand_dims(pad_seq, axis=0)
                            try:
                                probs = self.lstm_prediction_model(reshaped_seq, training=False).numpy()[0]
                                top_n_idx = np.argsort(probs)[-self.TOP_N:][::-1]; top_n_conf = probs[top_n_idx]
                                top_n_w = [self.index_to_word.get(idx, f"UNK_{idx}") for idx in top_n_idx]
                                if self._running: self.top_n_ready.emit([f"{w} ({c:.2f})" for w, c in zip(top_n_w, top_n_conf)])
                                if top_n_conf[0] >= self.PREDICTION_THRESHOLD: prediction_buffer.append(top_n_idx[0])
                            except Exception as e_pred: logging.exception(f"Erreur prédiction LSTM: {e_pred}"); self.top_n_ready.emit(["Erreur LSTM"])
                
            elif self.hands_solution and not hands_detected:
                if sequence_window: sequence_window.clear()
                if prediction_buffer: prediction_buffer.clear()
                if self.last_hands_detected_status:
                     if self._running: self.top_n_ready.emit([""])
                     if last_smoothed_word != "?": last_smoothed_word = "?"; self.prediction_ready.emit(last_smoothed_word)

            smoothed_word = "?"
            if prediction_buffer:
                try:
                    counts = Counter(prediction_buffer); common = counts.most_common(1)
                    if common: smoothed_word = self.index_to_word.get(common[0][0], "?")
                except Exception as e_smooth: logging.warning(f"Erreur lissage: {e_smooth}")
            if smoothed_word != last_smoothed_word:
                if self._running: self.prediction_ready.emit(smoothed_word)
                last_smoothed_word = smoothed_word
            
            try:
                 if frame_proc_times: avg_proc_t = np.mean(frame_proc_times); fps_proc = 1000/avg_proc_t if avg_proc_t > 0 else 0; cv2.putText(display_frame, f"Proc: {avg_proc_t:.1f}ms (~{fps_proc:.1f}FPS)", (10,target_h-30), cv2.FONT_HERSHEY_SIMPLEX,0.5,Colors.CV_LIGHT_GREY,1,cv2.LINE_AA)
                 loop_t_ms=(time.time()-loop_start_t)*1000; fps_loop=1000/loop_t_ms if loop_t_ms>0 else 0; cv2.putText(display_frame,f"Loop: {loop_t_ms:.1f}ms (~{fps_loop:.1f}FPS)",(10,target_h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,Colors.CV_LIGHT_GREY,1,cv2.LINE_AA)
                 if self.hands_solution: cv2.putText(display_frame, f"Mains: {'Oui' if hands_detected else 'Non'}", (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.6, Colors.CV_GREEN if hands_detected else Colors.CV_RED,1,cv2.LINE_AA)
            except Exception: pass

            if self._running and display_frame is not None and display_frame.size > 0: self.frame_ready.emit(display_frame)
        
        logging.info("Boucle vidéo terminée.");
        if self.cap and self.cap.isOpened(): self.cap.release(); logging.info("Webcam relâchée.")
        if self.hands_solution: self.hands_solution.close()
        try: tf.keras.backend.clear_session(); logging.info("Session Keras nettoyée.")
        except Exception as e_clear: logging.warning(f"Erreur nettoyage Keras: {e_clear}")
        logging.info("Thread vidéo terminé proprement.")

    def stop(self): self._running = False; logging.info("Arrêt demandé pour thread vidéo.")

class Ui_ParametersWindow(QWidget):
    color_changed = Signal(QColor)
    bg_color_changed = Signal(QColor)
    tts_voice_changed = Signal(str)
    tts_speed_changed = Signal(float)
    tts_main_volume_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initial_text_color = QColor("white")
        self.initial_bg_color = QColor(10, 32, 77)
        self.initial_tts_voice_key = Voice.FR_MALE_API.value
        self.initial_tts_speed = 1.0
        self.initial_tts_volume = 100

    def setupUi(self, ParametersWindow):
        ParametersWindow.setObjectName(u"ParametersWindow")
        ParametersWindow.resize(450, 450)
        ParametersWindow.setWindowTitle("Paramètres de l'application")

        self.main_v_layout = QVBoxLayout(ParametersWindow)
        self.main_v_layout.setSpacing(15)
        self.main_v_layout.setContentsMargins(15, 15, 15, 15)
        
        self.title_label = QLabel("Paramètres de l'application", ParametersWindow)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(u"font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        self.main_v_layout.addWidget(self.title_label)

        display_colors_frame = QFrame(ParametersWindow)
        display_colors_frame.setFrameShape(QFrame.Shape.StyledPanel)
        display_colors_layout = QGridLayout(display_colors_frame)

        self.text_color_label = QLabel("Couleur du texte:", display_colors_frame)
        self.text_color_btn = QPushButton("Choisir", display_colors_frame)
        self.text_color_preview = QLabel(display_colors_frame)
        self.text_color_preview.setFixedSize(30, 30)
        self.text_color_preview.setStyleSheet(f"background-color: {self.initial_text_color.name()}; border: 1px solid black; border-radius:3px;")
        
        display_colors_layout.addWidget(self.text_color_label, 0, 0)
        display_colors_layout.addWidget(self.text_color_preview, 0, 1)
        display_colors_layout.addWidget(self.text_color_btn, 0, 2)

        self.bg_color_label = QLabel("Couleur de fond:", display_colors_frame)
        self.bg_color_btn = QPushButton("Choisir", display_colors_frame)
        self.bg_color_preview = QLabel(display_colors_frame)
        self.bg_color_preview.setFixedSize(30, 30)
        self.bg_color_preview.setStyleSheet(f"background-color: {self.initial_bg_color.name()}; border: 1px solid black; border-radius:3px;")

        display_colors_layout.addWidget(self.bg_color_label, 1, 0)
        display_colors_layout.addWidget(self.bg_color_preview, 1, 1)
        display_colors_layout.addWidget(self.bg_color_btn, 1, 2)
        display_colors_layout.setColumnStretch(0,1)

        self.main_v_layout.addWidget(display_colors_frame)

        tts_settings_frame = QFrame(ParametersWindow)
        tts_settings_frame.setFrameShape(QFrame.Shape.StyledPanel)
        tts_settings_layout = QGridLayout(tts_settings_frame)

        self.tts_voice_label = QLabel("Voix TTS:", tts_settings_frame)
        self.tts_voice_combo = QComboBox(tts_settings_frame)
        for display_name, voice_val_key in ALL_VOICE_OPTIONS_MAP.items():
            self.tts_voice_combo.addItem(display_name, voice_val_key)
        
        initial_voice_index = self.tts_voice_combo.findData(self.initial_tts_voice_key)
        if initial_voice_index != -1:
            self.tts_voice_combo.setCurrentIndex(initial_voice_index)
        
        tts_settings_layout.addWidget(self.tts_voice_label, 0, 0)
        tts_settings_layout.addWidget(self.tts_voice_combo, 0, 1, 1, 2)

        self.tts_speed_label = QLabel("Vitesse TTS:", tts_settings_frame)
        self.tts_speed_slider = QSlider(Qt.Orientation.Horizontal, tts_settings_frame)
        self.tts_speed_slider.setMinimum(5)
        self.tts_speed_slider.setMaximum(20)
        self.tts_speed_slider.setValue(int(self.initial_tts_speed * 10)) 
        self.tts_speed_value_label = QLabel(f"{self.initial_tts_speed:.1f}x", tts_settings_frame)
        
        tts_settings_layout.addWidget(self.tts_speed_label, 1, 0)
        tts_settings_layout.addWidget(self.tts_speed_slider, 1, 1)
        tts_settings_layout.addWidget(self.tts_speed_value_label, 1, 2)

        self.tts_volume_label = QLabel("Volume TTS:", tts_settings_frame)
        self.tts_volume_slider = QSlider(Qt.Orientation.Horizontal, tts_settings_frame)
        self.tts_volume_slider.setMinimum(0)
        self.tts_volume_slider.setMaximum(150)
        self.tts_volume_slider.setValue(self.initial_tts_volume)
        self.tts_volume_value_label = QLabel(f"{self.initial_tts_volume}%", tts_settings_frame)

        tts_settings_layout.addWidget(self.tts_volume_label, 2, 0)
        tts_settings_layout.addWidget(self.tts_volume_slider, 2, 1)
        tts_settings_layout.addWidget(self.tts_volume_value_label, 2, 2)
        tts_settings_layout.setColumnStretch(1,1)

        self.main_v_layout.addWidget(tts_settings_frame)
        self.main_v_layout.addStretch(1)

        self.buttons_frame = QFrame(ParametersWindow)
        self.buttons_layout = QHBoxLayout(self.buttons_frame)
        self.buttons_layout.addStretch(1)
        
        self.default_btn = QPushButton("Par défaut", self.buttons_frame)
        self.close_btn = QPushButton("Fermer", self.buttons_frame)
        
        self.buttons_layout.addWidget(self.default_btn)
        self.buttons_layout.addWidget(self.close_btn)
        self.main_v_layout.addWidget(self.buttons_frame)
        
        self.text_color_btn.clicked.connect(self.choose_text_color_slot)
        self.bg_color_btn.clicked.connect(self.choose_bg_color_slot)
        self.default_btn.clicked.connect(self.reset_defaults_slot)
        self.close_btn.clicked.connect(ParametersWindow.close)

        self.tts_voice_combo.currentIndexChanged.connect(self.on_tts_voice_changed_slot)
        self.tts_speed_slider.valueChanged.connect(self.on_tts_speed_changed_slot)
        self.tts_volume_slider.valueChanged.connect(self.on_tts_volume_changed_slot)

    def set_initial_values(self, text_color, bg_color, tts_voice_k, tts_spd, tts_vol_perc):
        self.initial_text_color = text_color
        self.initial_bg_color = bg_color
        self.initial_tts_voice_key = tts_voice_k
        self.initial_tts_speed = tts_spd
        self.initial_tts_volume = tts_vol_perc

        self.text_color_preview.setStyleSheet(f"background-color: {self.initial_text_color.name()}; border: 1px solid black; border-radius:3px;")
        self.bg_color_preview.setStyleSheet(f"background-color: {self.initial_bg_color.name()}; border: 1px solid black; border-radius:3px;")
        
        combo_idx = self.tts_voice_combo.findData(self.initial_tts_voice_key)
        if combo_idx != -1: self.tts_voice_combo.setCurrentIndex(combo_idx)
        
        self.tts_speed_slider.setValue(int(self.initial_tts_speed * 10))
        self.tts_speed_value_label.setText(f"{self.initial_tts_speed:.1f}x")
        
        self.tts_volume_slider.setValue(self.initial_tts_volume)
        self.tts_volume_value_label.setText(f"{self.initial_tts_volume}%")

    @Slot()
    def choose_text_color_slot(self):
        current_color = QColor(self.text_color_preview.styleSheet().split("background-color: ")[1].split(";")[0])
        color = QColorDialog.getColor(current_color, self, "Choisir couleur du texte")
        if color.isValid():
            self.text_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius:3px;")
            self.color_changed.emit(color)
    
    @Slot()
    def choose_bg_color_slot(self):
        current_color = QColor(self.bg_color_preview.styleSheet().split("background-color: ")[1].split(";")[0])
        color = QColorDialog.getColor(current_color, self, "Choisir couleur de fond")
        if color.isValid():
            self.bg_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius:3px;")
            self.bg_color_changed.emit(color)
    
    @Slot()
    def reset_defaults_slot(self):
        default_text_qcolor = QColor("white")
        default_bg_qcolor = QColor(10, 32, 77)
        default_tts_voice_k = Voice.FR_MALE_API.value
        default_tts_spd = 1.0
        default_tts_vol_perc = 100
        
        self.text_color_preview.setStyleSheet(f"background-color: {default_text_qcolor.name()}; border: 1px solid black; border-radius:3px;")
        self.bg_color_preview.setStyleSheet(f"background-color: {default_bg_qcolor.name()}; border: 1px solid black; border-radius:3px;")
        
        default_voice_idx = self.tts_voice_combo.findData(default_tts_voice_k)
        if default_voice_idx != -1: self.tts_voice_combo.setCurrentIndex(default_voice_idx)
        
        self.tts_speed_slider.setValue(int(default_tts_spd * 10))
        self.tts_speed_value_label.setText(f"{default_tts_spd:.1f}x")
        
        self.tts_volume_slider.setValue(default_tts_vol_perc)
        self.tts_volume_value_label.setText(f"{default_tts_vol_perc}%")

        self.color_changed.emit(default_text_qcolor)
        self.bg_color_changed.emit(default_bg_qcolor)
        self.tts_voice_changed.emit(default_tts_voice_k)
        self.tts_speed_changed.emit(default_tts_spd)
        self.tts_main_volume_changed.emit(default_tts_vol_perc)

    @Slot(int)
    def on_tts_voice_changed_slot(self, index):
        voice_key_val = self.tts_voice_combo.itemData(index)
        self.tts_voice_changed.emit(voice_key_val)

    @Slot(int)
    def on_tts_speed_changed_slot(self, value):
        speed = value / 10.0
        self.tts_speed_value_label.setText(f"{speed:.1f}x")
        self.tts_speed_changed.emit(speed)

    @Slot(int)
    def on_tts_volume_changed_slot(self, value):
        self.tts_volume_value_label.setText(f"{value}%")
        self.tts_main_volume_changed.emit(value)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(800, 750) 
        MainWindow.setWindowTitle("Traduction LSF en Temps Réel") 

        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        MainWindow.setSizePolicy(sizePolicy)
        
        self.default_bg_qcolor = QColor(10, 32, 77)
        self.default_text_qcolor = QColor("white")
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget") 
        self.centralwidget.setStyleSheet(f"background-color: {self.default_bg_qcolor.name()};")
        
        self.gridLayout_3 = QGridLayout(self.centralwidget) 
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(10,10,10,10)
        self.gridLayout_3.setVerticalSpacing(15)

        self.setup_top_toolbar(self.centralwidget)
        self.gridLayout_3.addLayout(self.gridLayout_top_toolbar, 0, 0, 1, 1)

        self.setup_camera_view(self.centralwidget)
        self.gridLayout_3.addWidget(self.frame_camera, 1, 0, 1, 1)

        self.setup_text_area_with_audio_toggles(self.centralwidget)
        self.gridLayout_3.addWidget(self.frame_text_with_audio, 2, 0, 1, 1)

        self.setup_export_controls(self.centralwidget) # MODIFIÉ: Appel de la fonction
        self.gridLayout_3.addLayout(self.horizontalLayout_export, 3, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        
        self.gridLayout_3.setRowStretch(0, 0) 
        self.gridLayout_3.setRowStretch(1, 2) 
        self.gridLayout_3.setRowStretch(2, 1) 
        self.gridLayout_3.setRowStretch(3, 0) 
        self.gridLayout_3.setColumnStretch(0, 1) 

        MainWindow.setCentralWidget(self.centralwidget)
        self.setup_menu_statusbar(MainWindow)
        self.retranslateUi(MainWindow)
    
    def setup_top_toolbar(self, parent_widget): 
        self.gridLayout_top_toolbar = QGridLayout()
        self.gridLayout_top_toolbar.setObjectName(u"gridLayout_TopToolbar")
        
        self.boutonparametre = QPushButton(parent_widget)
        self.boutonparametre.setObjectName(u"boutonparametre")
        self.boutonparametre.setFixedSize(QSize(50, 50))
        self.boutonparametre.setToolTip("Ouvrir les paramètres d'affichage")
        self.boutonparametre.setText("⚙️"); self.boutonparametre.setFont(QFont("Segoe UI Emoji", 16))
        self.boutonparametre.setStyleSheet("QPushButton {border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white;} QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); }")
        self.gridLayout_top_toolbar.addWidget(self.boutonparametre, 0, 0, Qt.AlignmentFlag.AlignLeft)

        self.logo = QLabel(parent_widget)
        self.logo.setObjectName(u"logo")
        self.logo.setText("Traduction LSF")
        self.logo.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: transparent;")
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_top_toolbar.addWidget(self.logo, 0, 1)
        
        spacerItem = QSpacerItem(50, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        self.gridLayout_top_toolbar.addItem(spacerItem, 0, 2)
        self.gridLayout_top_toolbar.setColumnStretch(1, 1)
    
    def setup_camera_view(self, parent_widget): 
        self.frame_camera = QFrame(parent_widget)
        self.frame_camera.setObjectName(u"frame_camera")
        sizePolicyCamFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_camera.setSizePolicy(sizePolicyCamFrame)
        self.frame_camera.setMinimumHeight(300)
        self.frame_camera.setStyleSheet("QFrame#frame_camera { border: 1px solid gray; border-radius: 5px; background-color: black; }")
        
        gridLayout_camera_inner = QGridLayout(self.frame_camera)
        gridLayout_camera_inner.setObjectName(u"gridLayout_camera_inner")
        gridLayout_camera_inner.setContentsMargins(1, 1, 1, 1)

        self.camera_view = QLabel(self.frame_camera)
        self.camera_view.setObjectName(u"camera_view")
        sizePolicyCamLabel = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.camera_view.setSizePolicy(sizePolicyCamLabel)
        self.camera_view.setStyleSheet(u"QLabel#camera_view { background-color: transparent; border: none; color: grey; font-size: 16pt; }")
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setScaledContents(False)
        gridLayout_camera_inner.addWidget(self.camera_view, 0, 0, 1, 1)

    def setup_text_area_with_audio_toggles(self, parent_widget): 
        self.frame_text_with_audio = QFrame(parent_widget) 
        self.frame_text_with_audio.setObjectName(u"frame_text_audio") 
        self.frame_text_with_audio.setStyleSheet("QFrame#frame_text_audio { background-color: transparent; }")

        hbox_text_audio = QHBoxLayout(self.frame_text_with_audio)
        hbox_text_audio.setContentsMargins(0,0,0,0) 
        hbox_text_audio.setSpacing(10)

        self.frame_text_content = QFrame(self.frame_text_with_audio) 
        self.frame_text_content.setObjectName(u"frame_text_content_styling") 
        sizePolicyTextFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_text_content.setSizePolicy(sizePolicyTextFrame)
        self.frame_text_content.setMinimumHeight(100)
        self.frame_text_content.setMaximumHeight(250)
        self.frame_text_content.setStyleSheet("QFrame#frame_text_content_styling { background-color: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px; padding: 5px; }")
        
        layout_inside_text_frame = QVBoxLayout(self.frame_text_content)
        layout_inside_text_frame.setContentsMargins(5, 5, 5, 5)
        layout_inside_text_frame.setSpacing(5)
        
        self.label_predictions = QLabel("Prédictions:", self.frame_text_content)
        self.label_predictions.setObjectName(u"label_predictions")
        font_pred = QFont(); font_pred.setPointSize(11); font_pred.setBold(True)
        self.label_predictions.setFont(font_pred)
        self.label_predictions.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.label_predictions.setStyleSheet("background-color: transparent; color: white;") 
        layout_inside_text_frame.addWidget(self.label_predictions)

        self.textEdit = QTextEdit(self.frame_text_content)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setStyleSheet(u"QTextEdit { font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: white; }") 
        self.textEdit.setReadOnly(True)
        self.textEdit.setPlaceholderText("Les mots prédits apparaîtront ici...")
        layout_inside_text_frame.addWidget(self.textEdit, 1)
        
        hbox_text_audio.addWidget(self.frame_text_content, 1) 

        audio_v_layout = QVBoxLayout()
        audio_v_layout.setSpacing(10)
        audio_v_layout.setAlignment(Qt.AlignmentFlag.AlignCenter) 

        self.tts_toggle_group = QButtonGroup(self.frame_text_with_audio)
        tts_button_style = "QPushButton {min-width: 50px; max-width: 50px; min-height: 50px; max-height: 50px; border-radius: 25px; background-color: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: white; font: 16pt 'Segoe UI Emoji';} QPushButton:hover{background-color: rgba(255,255,255,0.2);} QPushButton:pressed{background-color: rgba(255,255,255,0.3);} QPushButton:checked{background-color: rgba(0,200,0,0.3); border-color: lightgreen;}"
        
        self.activesynthese = QPushButton(self.frame_text_with_audio)
        self.activesynthese.setCheckable(True); self.activesynthese.setChecked(True)
        self.activesynthese.setText("🔊"); self.activesynthese.setToolTip("Activer TTS")
        self.activesynthese.setStyleSheet(tts_button_style)
        self.tts_toggle_group.addButton(self.activesynthese, 1)
        audio_v_layout.addWidget(self.activesynthese)

        self.desactivesynthese = QPushButton(self.frame_text_with_audio)
        self.desactivesynthese.setCheckable(True)
        self.desactivesynthese.setText("🔇"); self.desactivesynthese.setToolTip("Désactiver TTS")
        self.desactivesynthese.setStyleSheet(tts_button_style.replace("rgba(0,200,0,0.3); border-color: lightgreen;", "rgba(200,0,0,0.3); border-color: lightcoral;"))
        self.tts_toggle_group.addButton(self.desactivesynthese, 0)
        audio_v_layout.addWidget(self.desactivesynthese)

        audio_v_layout.addStretch(1) 
        hbox_text_audio.addLayout(audio_v_layout, 0) 


    def setup_export_controls(self, parent_widget): 
        self.horizontalLayout_export = QHBoxLayout()
        self.horizontalLayout_export.setObjectName(u"horizontalLayout_export")
                
        self.exportation = QPushButton(parent_widget)
        self.exportation.setObjectName(u"exportation")
        self.exportation.setFixedSize(QSize(50, 50))
        self.exportation.setText("💾"); self.exportation.setFont(QFont("Segoe UI Emoji", 16))
        # MODIFIÉ: Tooltip et état activé
        self.exportation.setToolTip("Exporter le texte traduit vers un fichier .txt")
        self.exportation.setEnabled(True) 
        self.exportation.setStyleSheet("QPushButton { border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white; border: 1px solid rgba(255, 255, 255, 0.3); } QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); } QPushButton:disabled { background-color: rgba(128, 128, 128, 0.2); color: gray; border-color: rgba(128, 128, 128, 0.4); }")
        self.horizontalLayout_export.addWidget(self.exportation)

    def setup_menu_statusbar(self, MainWindow_instance): 
        self.statusbar = QStatusBar(MainWindow_instance)
        self.statusbar.setObjectName(u"statusbar")
        self.statusbar.setStyleSheet("QStatusBar { color: #DDDDDD; padding-left: 5px; background-color: transparent; }")
        MainWindow_instance.setStatusBar(self.statusbar)
        
        self.menubar = QMenuBar(MainWindow_instance)
        self.menubar.setObjectName(u"menubar")
        MainWindow_instance.setMenuBar(self.menubar)
    
    def retranslateUi(self, MainWindow_instance): 
        _translate = QCoreApplication.translate
        MainWindow_instance.setWindowTitle(_translate("MainWindow", u"Traduction LSF en Temps Réel", None))
        if hasattr(self, 'camera_view'):
             self.camera_view.setText(_translate("MainWindow", u"Initialisation Caméra...", None))
        if hasattr(self, 'boutonparametre'):
            self.boutonparametre.setToolTip(_translate("MainWindow", u"Ouvrir les paramètres d'affichage", None))
        # MODIFIÉ: Tooltip du bouton d'exportation
        if hasattr(self, 'exportation'):
            self.exportation.setToolTip(_translate("MainWindow", u"Exporter le texte traduit vers un fichier .txt", None))
        if hasattr(self, 'textEdit'): 
            self.textEdit.setPlaceholderText(_translate("MainWindow", u"Les mots prédits apparaîtront ici...", None))
        if hasattr(self, 'label_predictions'): 
             self.label_predictions.setText(_translate("MainWindow", u"Prédictions :", None))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        print("DEBUG (MainWindow.__init__): UI setup complete.")

        self.parameters_window = None
        self.ui.boutonparametre.clicked.connect(self.open_parameters_window)

        self.current_text_color = self.ui.default_text_qcolor 
        self.current_bg_color = self.ui.default_bg_qcolor
        self.apply_current_styles()
        print("DEBUG (MainWindow.__init__): Initial styles applied.")

        self.placeholder_timer = QTimer(self); self.placeholder_timer.timeout.connect(self.show_placeholder_camera_frame)
        self.placeholder_frame_counter = 0; self.placeholder_active = True
        if hasattr(self.ui, 'camera_view'):
            self.ui.camera_view.setText("Initialisation..."); self.ui.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.video_thread = VideoThread(self)
        print("DEBUG (MainWindow.__init__): VideoThread instance created.")
        self.video_thread.frame_ready.connect(self.update_camera_frame)
        self.video_thread.prediction_ready.connect(self.handle_new_prediction)
        self.video_thread.top_n_ready.connect(self.update_top_n_predictions_status)
        self.video_thread.error_occurred.connect(self.show_video_thread_error)
        self.video_thread.models_loaded.connect(self.on_video_models_loaded)
        self.video_thread.finished.connect(self.on_video_thread_finished)
        print("DEBUG (MainWindow.__init__): VideoThread signals connected.")

        self.tts_enabled = True
        self.current_tts_voice_key = Voice.FR_MALE_API.value
        self.current_tts_speed = 1.0
        self.current_tts_volume_percent = 100
        self.last_spoken_word = ""
        self.tts_worker_thread = None
        self.current_tts_volume_db = self.map_volume_percent_to_db(self.current_tts_volume_percent)

        self.ui.tts_toggle_group.idClicked.connect(self.toggle_tts_activation)
        self.ui.activesynthese.setChecked(self.tts_enabled)

        # AJOUTÉ: Connexion du bouton d'exportation
        if hasattr(self.ui, 'exportation'):
            self.ui.exportation.clicked.connect(self.export_text_to_file)

        if TTS_PYDUB_OK:
            print("DEBUG (MainWindow.__init__): Démarrage du thread worker TTS...")
            self.tts_worker_thread = threading.Thread(target=tts_worker_thread_func, daemon=True)
            self.tts_worker_thread.start()
        else:
            self.ui.statusbar.showMessage("ATTENTION: Lecture TTS désactivée (Pydub manquant).", 5000)
            self.ui.activesynthese.setEnabled(False)
            self.ui.desactivesynthese.setEnabled(False)

        self.ui.statusbar.showMessage("Initialisation: Chargement des modèles et de la caméra...")
        self.placeholder_timer.start(50)
        print("DEBUG (MainWindow.__init__): Démarrage VideoThread...")
        self.video_thread.start()

    # AJOUTÉ: Méthode pour exporter le texte
    @Slot()
    def export_text_to_file(self):
        """Exporte le contenu du QTextEdit vers un fichier texte."""
        text_to_save = self.ui.textEdit.toPlainText()
        if not text_to_save.strip(): # Vérifie si le texte est vide ou ne contient que des espaces
            QMessageBox.information(self, "Exportation", "Il n'y a aucun texte à exporter.")
            return

        # Proposer un nom de fichier par défaut basé sur la date et l'heure actuelles
        now = QDateTime.currentDateTime()
        suggested_filename = f"traduction_lsf_{now.toString('yyyy-MM-dd_HH-mm-ss')}.txt"
        
        # Essayer d'obtenir le dossier "Documents" de l'utilisateur, sinon le dossier personnel
        default_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        if not default_dir: # Fallback si DocumentsLocation n'est pas trouvé
            default_dir = os.path.expanduser("~") 
        
        default_path = os.path.join(default_dir, suggested_filename)

        fileName, _ = QFileDialog.getSaveFileName(self,
                                                "Sauvegarder le texte traduit",
                                                default_path,
                                                "Fichiers texte (*.txt);;Tous les fichiers (*)")

        if fileName: # Si l'utilisateur a sélectionné un fichier (n'a pas annulé)
            try:
                with open(fileName, 'w', encoding='utf-8') as f:
                    f.write(text_to_save)
                self.ui.statusbar.showMessage(f"Texte exporté avec succès vers : {os.path.basename(fileName)}", 5000)
                print(f"DEBUG: Texte exporté vers {fileName}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur d'exportation", f"Impossible de sauvegarder le fichier :\n{e}")
                self.ui.statusbar.showMessage(f"Erreur d'exportation : {e}", 5000)
                print(f"ERREUR: Exportation vers {fileName} échouée: {e}")


    def map_volume_percent_to_db(self, percent):
        if percent == 0: return -60.0
        if percent <= 100:
            return (percent - 100.0) * 0.6
        else:
            return (percent - 100.0) * (6.0 / 50.0)

    @Slot(int)
    def toggle_tts_activation(self, button_id):
        self.tts_enabled = (button_id == 1)
        status_msg = "Synthèse vocale activée" if self.tts_enabled else "Synthèse vocale désactivée"
        self.ui.statusbar.showMessage(status_msg, 3000)
        print(f"DEBUG: TTS enabled: {self.tts_enabled}")
        if self.tts_enabled:
            self.ui.activesynthese.setChecked(True)
        else:
            self.ui.desactivesynthese.setChecked(True)

    def apply_current_styles(self):
         bg_name = self.current_bg_color.name()
         txt_name = self.current_text_color.name()
         
         self.ui.centralwidget.setStyleSheet(f"""
            QWidget#centralwidget {{ background-color: {bg_name}; }}
            QFrame {{ background-color: transparent; }}
            QLabel, QTextEdit {{ color: {txt_name}; background-color: transparent;}}
         """) 
         
         self.ui.boutonparametre.setStyleSheet(f"QPushButton {{border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: {txt_name};}} QPushButton:hover {{ background-color: rgba(255, 255, 255, 0.2); }} QPushButton:pressed {{ background-color: rgba(255, 255, 255, 0.3); }}")
         self.ui.logo.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {txt_name}; background-color: transparent;")
         
         if hasattr(self.ui, 'frame_camera'): 
            self.ui.frame_camera.setStyleSheet("QFrame#frame_camera { border: 1px solid gray; border-radius: 5px; background-color: black; }")
         if hasattr(self.ui, 'camera_view'):
            self.ui.camera_view.setStyleSheet(f"QLabel#camera_view {{ background-color: transparent; border: none; color: grey; font-size: 16pt; }}") 
         
         if hasattr(self.ui, 'frame_text_with_audio'): 
             self.ui.frame_text_with_audio.setStyleSheet("background-color: transparent;")
             if hasattr(self.ui, 'frame_text_content'): 
                self.ui.frame_text_content.setStyleSheet("QFrame#frame_text_content_styling { background-color: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px; padding: 5px; }")
         
         if hasattr(self.ui, 'label_predictions'): 
             self.ui.label_predictions.setStyleSheet(f"background-color: transparent; color: {txt_name}; font-weight: bold; font-size: 11pt;")
         if hasattr(self.ui, 'textEdit'):
            self.ui.textEdit.setStyleSheet(f"QTextEdit {{ font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: {txt_name}; }}")

         tts_button_font_size = "16pt" 
         tts_button_base_style = f"min-width: 50px; max-width: 50px; min-height: 50px; max-height: 50px; border-radius: 25px; background-color: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); color: {txt_name}; font: {tts_button_font_size} 'Segoe UI Emoji';"
         tts_button_hover_style = "background-color: rgba(255,255,255,0.2);"
         tts_button_pressed_style = "background-color: rgba(255,255,255,0.3);"
         tts_button_checked_on = "background-color: rgba(0,200,0,0.3); border-color: lightgreen;"
         tts_button_checked_off = "background-color: rgba(200,0,0,0.3); border-color: lightcoral;"

         if hasattr(self.ui, 'activesynthese'):
            self.ui.activesynthese.setStyleSheet(f"QPushButton{{{tts_button_base_style}}} QPushButton:hover{{{tts_button_hover_style}}} QPushButton:pressed{{{tts_button_pressed_style}}} QPushButton:checked{{{tts_button_checked_on}}}")
         if hasattr(self.ui, 'desactivesynthese'):
            self.ui.desactivesynthese.setStyleSheet(f"QPushButton{{{tts_button_base_style}}} QPushButton:hover{{{tts_button_hover_style}}} QPushButton:pressed{{{tts_button_pressed_style}}} QPushButton:checked{{{tts_button_checked_off}}}")
         
         if hasattr(self.ui, 'exportation'):
            self.ui.exportation.setStyleSheet(f"QPushButton {{ border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: {txt_name}; border: 1px solid rgba(255, 255, 255, 0.3); }} QPushButton:hover {{ background-color: rgba(255, 255, 255, 0.2); }} QPushButton:pressed {{ background-color: rgba(255, 255, 255, 0.3); }} QPushButton:disabled {{ background-color: rgba(128, 128, 128, 0.2); color: gray; border-color: rgba(128, 128, 128, 0.4); }}")
         
         if hasattr(self.ui, 'statusbar'):
            self.ui.statusbar.setStyleSheet(f"QStatusBar {{ color: {self.current_text_color.lighter(130).name()}; padding-left: 5px; background-color: transparent;}}")


    @Slot()
    def show_placeholder_camera_frame(self):
        if not self.placeholder_active: return
        label = self.ui.camera_view if hasattr(self.ui, 'camera_view') else None
        if not label or not label.isVisible() or label.width() <= 10 or label.height() <= 10 : return
        try:
            w, h = label.width(), label.height(); pixmap = QPixmap(w, h); pixmap.fill(Qt.GlobalColor.black)
            painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            cx, cy = w / 2.0, h / 2.0; max_r = min(w, h) / 8.0; r_var = max_r / 3.0
            pulse = (1 + np.sin(self.placeholder_frame_counter * 0.1)) / 2.0; cur_r = max_r - (r_var * pulse)
            if cur_r > 0: painter.setBrush(QColor(40,40,40)); painter.setPen(Qt.PenStyle.NoPen); painter.drawEllipse(QPointF(cx,cy), cur_r, cur_r)
            
            font = QFont("Arial", 12 if w > 200 else 10); painter.setFont(font); painter.setPen(QColor(150,150,150))
            text_to_draw = "En attente de la caméra..."
            if self.video_thread and self.video_thread.isFinished():
                 status = self.ui.statusbar.currentMessage()
                 text_to_draw = "Échec initialisation" if "ERREUR" in status or "ÉCHEC" in status else "Caméra déconnectée"
            
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text_to_draw); painter.end()
            label.setPixmap(pixmap); self.placeholder_frame_counter += 1
        except Exception as e:
             print(f"ERREUR (show_placeholder_camera_frame): {e}")
             if self.placeholder_timer.isActive(): self.placeholder_timer.stop()
             if label: label.setText(f"Erreur Placeholder:\n{e}")

    @Slot(np.ndarray)
    def update_camera_frame(self, cv_img):
        if self.placeholder_active:
            print("DEBUG (update_camera_frame): Première frame reçue, arrêt placeholder.")
            if self.placeholder_timer.isActive(): self.placeholder_timer.stop()
            self.placeholder_active = False;
            if hasattr(self.ui, 'camera_view'): self.ui.camera_view.clear()
        
        if cv_img is None or cv_img.size == 0: return
        try:
            h, w, ch = cv_img.shape; bytes_per_line = ch * w
            qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            if qt_image.isNull(): print("ERREUR: Création QImage échouée!"); return
            
            if hasattr(self.ui, 'camera_view'):
                label = self.ui.camera_view
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
        except Exception as e:
            logging.error(f"Erreur màj frame: {e}", exc_info=True)
            if hasattr(self.ui, 'camera_view'): self.ui.camera_view.setText(f"Erreur affichage frame:\n{str(e)}")
            self._reactivate_placeholder_if_needed()

    def _reactivate_placeholder_if_needed(self):
        if not self.placeholder_active:
            self.placeholder_active = True
            self.placeholder_timer.start(50)

    @Slot(str)
    def handle_new_prediction(self, predicted_text):
        word_to_speak = None
        if predicted_text and predicted_text != "?":
            current_content = self.ui.textEdit.toPlainText(); words = current_content.split()
            if not words or predicted_text.lower() != words[-1].lower():
                 max_hist = 50; new_words = (words + [predicted_text])[-max_hist:]
                 self.ui.textEdit.setPlainText(" ".join(new_words))
                 self.ui.textEdit.moveCursor(QTextCursor.MoveOperation.End)
                 if predicted_text.lower() != self.last_spoken_word.lower():
                     word_to_speak = predicted_text

        if self.tts_enabled and word_to_speak and TTS_PYDUB_OK:
            if add_tts_request_to_queue(word_to_speak, self.current_tts_voice_key, self.current_tts_speed, self.current_tts_volume_db):
                self.last_spoken_word = word_to_speak

    @Slot(list)
    def update_top_n_predictions_status(self, top_n_list):
        filtered = [item for item in top_n_list if item and item.strip()]
        if filtered: self.ui.statusbar.showMessage(" | ".join(filtered[:3]))
        else:
            if "ERREUR" not in self.ui.statusbar.currentMessage() and "ÉCHEC" not in self.ui.statusbar.currentMessage():
                 self.ui.statusbar.showMessage("Prêt.", 2000)

    @Slot(str)
    def show_video_thread_error(self, message):
        logging.error(f"Erreur VideoThread: {message}")
        critical_keys = ["impossible d'ouvrir", "webcam", "modèle", "vocab", "shape", "config.py", "manquant"]
        is_crit = any(key in message.lower() for key in critical_keys)
        if is_crit:
             self.ui.statusbar.showMessage(f"ERREUR CRITIQUE: {message}", 0)
             if hasattr(self.ui, 'camera_view'): self.ui.camera_view.setText(f"ERREUR CRITIQUE:\n{message}\nConsultez les logs.")
             self._reactivate_placeholder_if_needed()
             QMessageBox.critical(self, "Erreur Critique VideoThread", f"Une erreur critique est survenue:\n\n{message}\n\nL'application pourrait ne pas fonctionner.")
        else: self.ui.statusbar.showMessage(f"Erreur Vidéo: {message}", 10000)

    @Slot(bool)
    def on_video_models_loaded(self, success):
        if success: self.ui.statusbar.showMessage("Modèles vidéo chargés. Démarrage webcam...", 3000)
        else:
            msg = "ÉCHEC INITIALISATION VIDÉO. Vérifiez logs."
            self.ui.statusbar.showMessage(msg, 0)
            if hasattr(self.ui, 'camera_view'): self.ui.camera_view.setText(msg)
            self._reactivate_placeholder_if_needed()

    @Slot()
    def on_video_thread_finished(self):
        logging.info("Thread vidéo terminé.")
        if "ERREUR" not in self.ui.statusbar.currentMessage() and "ÉCHEC" not in self.ui.statusbar.currentMessage():
            self.ui.statusbar.showMessage("Connexion caméra terminée.", 3000)
        self._reactivate_placeholder_if_needed()
        if hasattr(self.ui, 'camera_view'): self.ui.camera_view.setText("Caméra déconnectée")

    def open_parameters_window(self):
        if self.parameters_window is None:
            self.parameters_window = Ui_ParametersWindow()
            self.parameters_window.setupUi(self.parameters_window)
            self.parameters_window.color_changed.connect(self.update_text_color_setting)
            self.parameters_window.bg_color_changed.connect(self.update_bg_color_setting)
            self.parameters_window.tts_voice_changed.connect(self.update_tts_voice_setting)
            self.parameters_window.tts_speed_changed.connect(self.update_tts_speed_setting)
            self.parameters_window.tts_main_volume_changed.connect(self.update_tts_volume_setting)
        
        self.parameters_window.set_initial_values(
            self.current_text_color, self.current_bg_color,
            self.current_tts_voice_key, self.current_tts_speed, self.current_tts_volume_percent
        )
        self.parameters_window.show()
        self.parameters_window.raise_()
        self.parameters_window.activateWindow()

    @Slot(QColor)
    def update_text_color_setting(self, color):
        if color.isValid():
            self.current_text_color = color
            self.apply_current_styles()
    
    @Slot(QColor)
    def update_bg_color_setting(self, color):
        if color.isValid():
            self.current_bg_color = color
            self.apply_current_styles()

    @Slot(str)
    def update_tts_voice_setting(self, voice_key_val):
        self.current_tts_voice_key = voice_key_val
        display_name = "Inconnue"
        for name, key_val in ALL_VOICE_OPTIONS_MAP.items():
            if key_val == voice_key_val:
                display_name = name
                break
        self.ui.statusbar.showMessage(f"Voix TTS changée: {display_name}", 3000)
        print(f"DEBUG: Voix TTS sélectionnée: {voice_key_val} ({display_name})")

    @Slot(float)
    def update_tts_speed_setting(self, speed):
        self.current_tts_speed = speed
        self.ui.statusbar.showMessage(f"Vitesse TTS changée: {speed:.1f}x", 3000)
        print(f"DEBUG: Vitesse TTS sélectionnée: {speed:.1f}x")

    @Slot(int)
    def update_tts_volume_setting(self, volume_percent):
        self.current_tts_volume_percent = volume_percent
        self.current_tts_volume_db = self.map_volume_percent_to_db(volume_percent)
        self.ui.statusbar.showMessage(f"Volume principal TTS: {volume_percent}%", 3000)
        print(f"DEBUG: Volume TTS sélectionné: {volume_percent}% (-> {self.current_tts_volume_db:.1f} dB)")

    def closeEvent(self, event):
        logging.info("Fermeture fenêtre principale demandée.");
        if self.placeholder_timer.isActive(): self.placeholder_timer.stop()
        if self.parameters_window and self.parameters_window.isVisible(): self.parameters_window.close()
        
        if self.video_thread and self.video_thread.isRunning():
            logging.info("Arrêt thread vidéo..."); self.video_thread.stop()
            if not self.video_thread.wait(3000): logging.warning("Timeout thread vidéo.")
            else: logging.info("Thread vidéo arrêté.")
        
        if TTS_PYDUB_OK and self.tts_worker_thread and self.tts_worker_thread.is_alive():
            logging.info("Arrêt worker TTS..."); stop_tts_worker_event.set()
            tts_request_queue.join()
            self.tts_worker_thread.join(timeout=5)
            if self.tts_worker_thread.is_alive(): logging.warning("Timeout thread TTS.")
            else: logging.info("Thread TTS arrêté.")
        event.accept()

if __name__ == "__main__":
    print("DEBUG: Démarrage application principale...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    config_file_tts = os.path.join(data_dir, 'config.json')
    if not os.path.exists(data_dir):
        try: os.makedirs(data_dir); print(f"DEBUG: Répertoire 'data' créé: {data_dir}")
        except OSError as e: print(f"ERREUR création répertoire 'data': {e}")
    if not os.path.exists(config_file_tts):
        print(f"ATTENTION: config.json TTS non trouvé ({config_file_tts}). Création par défaut.")
        dummy_tts_config = [{"name": "TikTok Default", "url": "https://tiktok-tts.weilnet.workers.dev/api/generation", "response": "data"}]
        try:
            with open(config_file_tts, 'w', encoding='utf-8') as f: json_dump(dummy_tts_config, f, indent=4)
            print(f"DEBUG: config.json TTS par défaut créé.")
        except Exception as e: print(f"ERREUR création config.json TTS: {e}")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print("DEBUG: Entrée boucle événements Qt...")
    exit_code = app.exec()
    print(f"DEBUG: Boucle événements Qt terminée avec code {exit_code}.")
    sys.exit(exit_code)
