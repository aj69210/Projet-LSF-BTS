import os
import numpy as np

# --- Configuration ---
DATA_DIR = "data/data_tmp"
AUG_FACTOR = 1  # Nombre de copies générées par fichier existant (ici 1)

# --- Fonctions d'augmentation ---
def add_noise(sequence, noise_level=0.01):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_sequence(sequence, scale_range=(0.95, 1.05)):
    factor = np.random.uniform(*scale_range)
    return sequence * factor

def augment_sequence(sequence):
    sequence = add_noise(sequence, noise_level=0.01)
    sequence = scale_sequence(sequence, scale_range=(0.98, 1.02))
    return sequence

# --- Traitement des fichiers ---
def augment_dataset(data_dir):
    total_augmented = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                sequence = np.load(file_path)

                if sequence.ndim != 2 or sequence.shape[1] != 258:
                    print(f"❌ Ignoré (forme non conforme): {file}")
                    continue

                for i in range(AUG_FACTOR):
                    augmented_sequence = augment_sequence(sequence)
                    new_file_name = file.replace(".npy", f"_aug2.{i+1}.npy")
                    new_file_path = os.path.join(root, new_file_name)
                    np.save(new_file_path, augmented_sequence)
                    total_augmented += 1

    print(f"\n✅ Augmentation terminée. {total_augmented} nouveaux fichiers créés.")

# --- Lancement ---
if __name__ == "__main__":
    augment_dataset(DATA_DIR)
