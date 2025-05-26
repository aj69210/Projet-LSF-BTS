# Python standard modules
import os
import requests
import base64
import re
import queue
import threading
import time
import tempfile
from json import load, dump
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys

# Downloaded modules
from gtts import gTTS
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    PYDUB_AVAILABLE = True
except ImportError:
    # ... (Message d'erreur inchangé) ...
    PYDUB_AVAILABLE = False

# ... (Définition de Voice Enum inchangée) ...
try:
    class Voice(Enum):
        FR_MALE = "fr_002"
        EN_MALE = "en_us_006"
        EN_FEMALE = "en_us_001"
except ImportError:
    class Voice(Enum):
        FR_MALE = "fr_002"
        EN_MALE = "en_us_006"
        EN_FEMALE = "en_us_001"

api_voice_keys = {v.name.lower(): v for v in Voice}
valid_voice_keys = list(api_voice_keys.keys()) + ["fr_female"]


# ... (_load_endpoints, _split_text, _fetch_audio_bytes_from_api, generate_api_audio, _save_audio_file, _validate_args inchangés) ...
def _load_endpoints() -> List[Dict[str, str]]:
    script_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(script_dir, 'data', 'config.json')
    if not os.path.exists(json_file_path):
        json_file_path_alt = os.path.join(script_dir, '../data', 'config.json')
        if os.path.exists(json_file_path_alt):
             json_file_path = json_file_path_alt
        else:
             raise FileNotFoundError(f"Cannot find config.json in {os.path.join(script_dir, 'data')} or {os.path.join(script_dir, '../data')}")
    try:
        with open(json_file_path, 'r') as file:
            return load(file)
    except Exception as e:
        print(f"Error loading endpoints from {json_file_path}: {e}")
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
            if current_chunk:
                 merged_chunks.append(current_chunk)
            if len(separated_chunk.encode("utf-8")) <= character_limit:
                current_chunk = separated_chunk
            else:
                print(f"Warning: Segment too long even after splitting: {separated_chunk[:50]}...")
                merged_chunks.append(separated_chunk)
                current_chunk = ""
    if current_chunk:
        merged_chunks.append(current_chunk)
    return [chunk for chunk in merged_chunks if chunk and not chunk.isspace()]

def _fetch_audio_bytes_from_api(
    endpoint: Dict[str, str],
    text_chunk: str,
    voice_id: str
) -> Optional[str]:
    try:
        print(f"  [API TTS] Sending chunk to {endpoint.get('url', 'N/A')} (Voice: {voice_id}): {text_chunk[:30]}...")
        response = requests.post(endpoint["url"], json={"text": text_chunk, "voice": voice_id}, timeout=15)
        response.raise_for_status()
        json_response = response.json()
        if endpoint["response"] in json_response:
             print(f"  [API TTS] Received chunk data.")
             return json_response[endpoint["response"]]
        else:
             print(f"  [API TTS] Error: Key '{endpoint['response']}' not found in response: {json_response}")
             return None
    except requests.exceptions.Timeout:
        print(f"  [API TTS] Error: Request timed out for chunk: {text_chunk[:30]}...")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [API TTS] Error fetching audio chunk: {e}")
        return None
    except KeyError:
        print(f"  [API TTS] Error: Invalid endpoint configuration or API response structure.")
        return None
    except Exception as e:
        print(f"  [API TTS] An unexpected error occurred during fetch: {e}")
        return None

def generate_api_audio(text: str, output_file_path: str, voice: Voice) -> bool:
    print(f"[API TTS] Generating audio for voice {voice.name} ({voice.value}): {text[:50]}...")
    try:
        _validate_args(text, voice)
    except (TypeError, ValueError) as e:
        print(f"[API TTS] Error: Invalid arguments - {e}")
        return False
    endpoint_data = _load_endpoints()
    if not endpoint_data:
        print("[API TTS] Error: No endpoints loaded. Cannot generate audio.")
        return False
    for endpoint in endpoint_data:
        print(f"[API TTS] Trying endpoint: {endpoint.get('url', 'N/A')}")
        text_chunks: List[str] = _split_text(text)
        if not text_chunks:
            print("[API TTS] Error: Text resulted in empty chunks after splitting.")
            continue
        audio_chunks_b64: List[Optional[str]] = [None] * len(text_chunks)
        threads: List[threading.Thread] = []
        results: Dict[int, Optional[str]] = {}
        def thread_target(index: int, chunk: str):
            results[index] = _fetch_audio_bytes_from_api(endpoint, chunk, voice.value)
        for i, chunk in enumerate(text_chunks):
            thread = threading.Thread(target=thread_target, args=(i, chunk))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        for i in range(len(text_chunks)):
            audio_chunks_b64[i] = results.get(i)
        if all(chunk is not None for chunk in audio_chunks_b64):
            print("[API TTS] All chunks received successfully. Concatenating and decoding...")
            try:
                full_audio_b64 = "".join([chunk for chunk in audio_chunks_b64 if chunk is not None])
                audio_bytes = base64.b64decode(full_audio_b64)
                _save_audio_file(output_file_path, audio_bytes)
                print(f"[API TTS] Audio successfully saved to {output_file_path}")
                return True
            except base64.binascii.Error as e:
                print(f"[API TTS] Error decoding base64 string: {e}")
            except IOError as e:
                print(f"[API TTS] Error saving audio file {output_file_path}: {e}")
                return False
            except Exception as e:
                print(f"[API TTS] Unexpected error during saving/decoding: {e}")
        else:
             print(f"[API TTS] Failed to fetch all audio chunks for endpoint {endpoint.get('url', 'N/A')}. Missing chunks: {[i for i, ch in enumerate(audio_chunks_b64) if ch is None]}")
    print("[API TTS] Error: Failed to generate audio using all available endpoints.")
    return False

def _save_audio_file(output_file_path: str, audio_bytes: bytes):
    try:
        with open(output_file_path, "wb") as file:
            file.write(audio_bytes)
        print(f"  [Util] Audio data written to {output_file_path}")
    except IOError as e:
        print(f"  [Util] Error writing audio file {output_file_path}: {e}")
        raise

def _validate_args(text: str, voice: Voice): # Pour generate_api_audio
    if not isinstance(voice, Voice):
        raise TypeError(f"'voice' must be of type Voice, got {type(voice)}")
    if not text or text.isspace():
        raise ValueError("text must not be empty or whitespace")

def _validate_request_args(text: str, voice_key: str, speed: float): # Pour les fonctions d'ajout à la queue
    if not text or text.isspace():
        raise ValueError("[API Validation] Error: Text cannot be empty.")
    voice_key = voice_key.lower()
    if voice_key not in valid_voice_keys:
        raise ValueError(f"[API Validation] Error: Invalid voice_key '{voice_key}'. Valid keys are: {', '.join(valid_voice_keys)}")
    if not isinstance(speed, (int, float)) or speed <= 0:
        raise ValueError("[API Validation] Error: Speed must be a positive number.")


# File d'attente pour les requêtes TTS : (text, voice_key, specific_speed)
tts_queue: queue.Queue = queue.Queue()
stop_worker = threading.Event()

# Vitesse de lecture globale (utilisée par l'interface interactive comme défaut)
default_playback_speed: float = 1.0 # Renommée pour clarté

def generate_female_audio(text: str, output_file_path: str) -> bool:
    # ... (Inchangé) ...
    print(f"[FR Female TTS - gTTS] Generating audio for: {text[:50]}...")
    try:
        tts = gTTS(text=text, lang='fr', slow=False)
        tts.save(output_file_path)
        print(f"[FR Female TTS - gTTS] Audio successfully saved to {output_file_path}")
        return True
    except Exception as e:
        print(f"[FR Female TTS - gTTS] Error generating or saving audio: {e}")
        return False

# --- Fonctions de lecture audio ---
# speed_change est inchangé
def speed_change(sound: AudioSegment, speed_factor: float = 1.0) -> AudioSegment:
    if speed_factor == 1.0:
        return sound
    print(f"  [Playback] Applying speed factor: {speed_factor:.2f}")
    new_frame_rate = int(sound.frame_rate * speed_factor)
    if new_frame_rate <= 0:
        print(f"  [Playback] Warning: Calculated frame rate ({new_frame_rate}) is invalid. Using original rate.")
        return sound
    return sound._spawn(sound.raw_data, overrides={
        "frame_rate": new_frame_rate
    })

# Étape 3: Modifier play_audio_file
def play_audio_file(file_path: str, speed_to_use: float): # Prend la vitesse en argument
    """Plays an audio file using pydub, applying the SPECIFIED playback speed."""
    # Ne plus utiliser 'global default_playback_speed' ici

    if not PYDUB_AVAILABLE:
        print("[Playback] Pydub not available. Skipping playback.")
        return

    final_audio_to_play = None

    try:
        print(f"[Playback] Loading {file_path}...")
        file_extension = os.path.splitext(file_path)[1].lower().strip('.')
        if not file_extension:
            file_extension = "mp3"
            print(f"[Playback] Warning: No file extension detected, assuming '{file_extension}'.")

        audio = AudioSegment.from_file(file_path, format=file_extension)

        # Appliquer le changement de vitesse avec speed_to_use
        audio_at_speed = speed_change(audio, speed_to_use)

        target_frame_rate = audio.frame_rate if speed_to_use == 1.0 else 44100

        if int(audio_at_speed.frame_rate) != target_frame_rate and speed_to_use != 1.0:
            print(f"  [Playback] Resampling from {audio_at_speed.frame_rate} Hz to {target_frame_rate} Hz for compatibility...")
            final_audio_to_play = audio_at_speed.set_frame_rate(target_frame_rate)
        else:
             final_audio_to_play = audio_at_speed

        print(f"[Playback] Playing at {speed_to_use:.2f}x speed (Sample Rate: {final_audio_to_play.frame_rate} Hz)...")
        pydub_play(final_audio_to_play)
        print(f"[Playback] Finished playing {file_path}.")

    # ... (Gestion des erreurs de play_audio_file inchangée) ...
    except FileNotFoundError:
        print(f"[Playback] Error: File not found at {file_path}")
    except Exception as e:
        current_rate = final_audio_to_play.frame_rate if final_audio_to_play else "N/A"
        if "Weird sample rates" in str(e):
             print(f"[Playback] Error playing sound file {file_path}: {e}")
             print(f"[Playback] Even after resampling, the sample rate {current_rate} Hz might not be supported by the backend.")
        elif "Cannot find ffprobe" in str(e) or "Cannot find ffmpeg" in str(e) or "[WinError 2]" in str(e):
             print(f"[Playback] Error playing sound file {file_path}: {e}")
             print(f"[Playback] Error: Ensure 'ffmpeg'/'ffprobe' executables are found by pydub.")
             print(f"[Playback] Check your system PATH or explicit configuration if used.")
        elif "[Errno 13] Permission denied" in str(e):
             print(f"[Playback] Error playing sound file {file_path}: {e}")
             print(f"[Playback] Permission denied, possibly when creating/accessing a temporary WAV file for playback.")
             print(f"[Playback] Check antivirus or folder permissions for the Temp directory.")
        else:
            print(f"[Playback] Error playing sound file {file_path}: {e}")
            print("[Playback] Ensure 'ffmpeg' is installed and in your system's PATH.")
            print("[Playback] Also check if the audio file format is supported or if there are permission issues.")

# Étape 2: Modifier tts_worker
def tts_worker():
    """Worker thread that processes TTS requests from the queue."""
    print("[Worker] TTS Worker thread started.")
    while not stop_worker.is_set():
        try:
            # Récupère (texte, voice_key, specific_speed)
            text, voice_key, specific_speed = tts_queue.get(timeout=1.0)
            print(f"\n[Worker] Processing request: VoiceKey='{voice_key}', Speed={specific_speed:.2f}x, Text='{text[:50]}...'")

            success = False
            temp_file_path = None

            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
                    temp_file_path = tmp_audio_file.name
                    print(f"[Worker] Using temporary file: {temp_file_path}")

                if voice_key == 'fr_female':
                    success = generate_female_audio(text, temp_file_path)
                elif voice_key in api_voice_keys:
                    target_voice_enum = api_voice_keys[voice_key]
                    success = generate_api_audio(text, temp_file_path, target_voice_enum)
                else:
                    print(f"[Worker] Error: Unknown voice key '{voice_key}'")

                if success and temp_file_path and os.path.exists(temp_file_path):
                    # Passe la vitesse spécifique à play_audio_file
                    play_audio_file(temp_file_path, specific_speed)
                elif not success:
                    print(f"[Worker] Failed to generate audio for request.")

            finally:
                # ... (Nettoyage du fichier temp inchangé) ...
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"[Worker] Cleaned up temporary file: {temp_file_path}")
                    except OSError as e:
                        print(f"[Worker] Error removing temporary file {temp_file_path}: {e}")
            tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            # ... (Gestion d'erreur du worker inchangée) ...
            print(f"[Worker] Unexpected error in worker loop: {e}")
            import traceback
            traceback.print_exc()
            try:
                tts_queue.task_done()
            except ValueError:
                 pass
            time.sleep(1)
    print("[Worker] TTS Worker thread finished.")


# Étape 4: Créer la nouvelle fonction add_speech_request
def add_speech_request(text: str, voice_key: str, speed: float = 1.0):
    """
    Adds a text-to-speech request to the queue with a SPECIFIC playback speed.
    This function is intended for programmatic use.

    Args:
        text (str): The text to synthesize.
        voice_key (str): The key for the desired voice (e.g., "fr_male", "en_female").
        speed (float, optional): The desired playback speed for this specific request. Defaults to 1.0.
    """
    try:
        _validate_request_args(text, voice_key, speed)
    except ValueError as e:
        print(e) # Affiche l'erreur de validation
        return False # Indique l'échec

    print(f"[API - Programmatic] Queuing request: VoiceKey={voice_key}, Speed={speed:.2f}x, Text='{text[:50]}...'")
    tts_queue.put((text, voice_key.lower(), speed))
    return True # Indique le succès de la mise en file d'attente

if __name__ == "__main__":
    # --- Configuration initiale et vérifications (inchangées) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    config_file = os.path.join(data_dir, 'config.json')
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"Répertoire créé : {data_dir}")
        except OSError as e:
            print(f"Erreur lors de la création du répertoire {data_dir}: {e}")
            sys.exit(1)
    if not os.path.exists(config_file):
        print(f"Attention : config.json non trouvé à {config_file}. Création d'un fichier par défaut.")
        print("Veuillez éditer 'data/config.json' avec vos vrais endpoints API.")
        dummy_config = [
            {"url": "https://tiktok-tts.weilnet.workers.dev/api/generation", "response": "data"},
            {"url": "https://example.com/api/tts", "response": "audio_base64"}
        ]
        try:
            with open(config_file, 'w') as f:
                dump(dummy_config, f, indent=4)
        except Exception as e:
             print(f"Erreur lors de la création du fichier config.json par défaut : {e}")
             sys.exit(1)

    print("-" * 30)
    print("IMPORTANT : Vérifiez les identifiants des voix dans l'Enum 'Voice' du code.")
    print(f"Actuellement : FR_MALE='{Voice.FR_MALE.value}', EN_MALE='{Voice.EN_MALE.value}', EN_FEMALE='{Voice.EN_FEMALE.value}'")
    print("Modifiez ces valeurs pour correspondre aux identifiants réels de votre API !")
    print("-" * 30)

    # Plus besoin de demander la vitesse PAR DÉFAUT à l'utilisateur ici
    # default_playback_speed = get_playback_speed_from_user()

    # --- Démarrage du thread worker TTS ---
    if not PYDUB_AVAILABLE:
         print("\nATTENTION : pydub non disponible, la lecture audio sera désactivée.")
    print("[Main] Démarrage du worker TTS...")
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # --- Exemple d'utilisation programmatique ---
    # Vous mettriez ici vos appels à add_speech_request
    print("[Main] Ajout de requêtes TTS programmatiques...")
    add_speech_request("Bonjour le monde, ceci est un test en français.", "fr_male", 1.0)
    add_speech_request("Hello world, this is a test in English at faster speed.", "en_female", 1.3)
    add_speech_request("Un dernier test plus lent pour la route.", "fr_female", 0.8)
    add_speech_request("This is an English male voice.", "en_male", 1.1)

    # Attendre que toutes les tâches dans la file d'attente soient traitées
    print("[Main] Toutes les requêtes ont été ajoutées à la file d'attente.")
    print("[Main] Attente de la fin du traitement de toutes les tâches...")
    tts_queue.join() # Bloque jusqu'à ce que tous les items de la queue aient été récupérés et traités (task_done() appelé)

    # Une fois toutes les tâches traitées, on peut arrêter le worker
    print("\n[Main] Toutes les tâches TTS ont été traitées.")
    print("[Main] Arrêt du worker TTS...")
    stop_worker.set() # Signale au worker de s'arrêter après sa tâche en cours (ou immédiatement s'il attend)

    # Optionnel: attendre que le thread worker se termine réellement
    # Si daemon=True, ce n'est pas strictement nécessaire pour que le programme principal quitte,
    # mais c'est plus propre si vous voulez être sûr que son message "TTS Worker thread finished." s'affiche.
    # tts_thread.join(timeout=5) # Attendre jusqu'à 5 secondes que le thread se termine

    print("[Main] Programme terminé.")
