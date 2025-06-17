import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model # Functional API
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Attention # Functional API
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
# from copy import deepcopy # Not strictly needed if loading best model from checkpoint

# Suppress specific TensorFlow warnings if desired (optional)
# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR level

# --- Configuration ---
DATA_DIR = "data/data_tmp"
SAVE_DIR = "data/models"
SEQUENCE_LENGTH = 100
LSTM_UNITS = 256
EPOCHS_PER_SESSION = 500 # High value, EarlyStopping will determine actual epochs
NUM_SESSIONS = 5
BATCH_SIZE = 64
TEST_SIZE = 0.15
NUM_FEATURES = 258 # Ensure this matches data generation
PATIENCE = 25
MIN_SAMPLES_PER_CLASS = 3
INITIAL_LEARNING_RATE = 0.001 # Initial LR for Adam optimizer

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Data Loading function ---
def load_data(data_dir):
    sequences, labels = [], []
    ignored_files = {'wrong_shape': [], 'empty': [], 'load_error': [], 'no_label': []}
    print(f"Loading data from: {DATA_DIR}")
    expected_feature_dim = NUM_FEATURES

    for class_label in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_label)
        if os.path.isdir(class_path):
            print(f"Processing class: {class_label}")
            for file_name in os.listdir(class_path):
                if file_name.endswith('.npy'):
                    sequence_path = os.path.join(class_path, file_name)
                    try:
                        sequence = np.load(sequence_path)

                        if sequence.size == 0:
                            print(f"  Ignoring empty file: {sequence_path}")
                            ignored_files['empty'].append(sequence_path)
                            continue

                        if sequence.ndim != 2 or sequence.shape[1] != expected_feature_dim:
                            print(f"  Ignoring file with wrong shape: {sequence_path}. Got {sequence.shape}, expected (*, {expected_feature_dim})")
                            ignored_files['wrong_shape'].append(sequence_path)
                            continue

                        # Append raw sequence, padding/normalization happens later
                        sequences.append(sequence.astype(np.float32))
                        labels.append(class_label)

                    except Exception as e:
                        print(f"  Error loading file {sequence_path}: {str(e)}")
                        ignored_files['load_error'].append(sequence_path)
        # else: Handle files directly in DATA_DIR if necessary (currently ignored)
            # Example: Check if class_path is a file and ends with .npy

    print(f"Finished loading. Found {len(sequences)} sequences.")
    return sequences, labels, ignored_files

# --- Load and Prepare Data ---
sequences_list, labels, ignored_files = load_data(DATA_DIR)

# Log ignored files
print("\n--- Ignored Files Report ---")
for reason, files in ignored_files.items():
    print(f"{reason}: {len(files)} files")
    # Optional: Show file names if needed
    # if files: print(f"  Examples: {files[:min(3, len(files))]}")

# Filter classes based on minimum samples
label_counts = Counter(labels)
filtered_sequences_list, filtered_labels = [], []
original_class_count = len(label_counts)

print("\n--- Class Filtering ---")
for seq, label in zip(sequences_list, labels):
    if label_counts[label] >= MIN_SAMPLES_PER_CLASS:
        filtered_sequences_list.append(seq)
        filtered_labels.append(label)
    else:
        # print(f"Filtering out class '{label}' (count: {label_counts[label]} < {MIN_SAMPLES_PER_CLASS})")
        pass # Avoid excessive printing

print(f"Original classes: {original_class_count}")
print(f"Classes after filtering (>= {MIN_SAMPLES_PER_CLASS} samples): {len(set(filtered_labels))}")

if not filtered_sequences_list:
     raise ValueError("No sequences left after filtering. Check MIN_SAMPLES_PER_CLASS or your data.")

# Pad sequences *after* filtering
print(f"\nPadding sequences to length {SEQUENCE_LENGTH}...")
X_padded = pad_sequences(filtered_sequences_list, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post', dtype='float32')
print(f"Shape after padding: {X_padded.shape}") # Should be (num_samples, SEQUENCE_LENGTH, NUM_FEATURES)

# Normalize *after* padding using global mean/std
print("Calculating normalization parameters (mean, std)...")
# Calculate mean/std across all samples and time steps for each feature
mean = np.mean(X_padded, axis=(0, 1), keepdims=True) # Shape: (1, 1, NUM_FEATURES)
std = np.std(X_padded, axis=(0, 1), keepdims=True)   # Shape: (1, 1, NUM_FEATURES)
print("Applying normalization...")
X = (X_padded - mean) / (std + 1e-8) # Add epsilon to prevent division by zero
print("Normalization applied.")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(filtered_labels) # Integer labels (0, 1, 2...)
y = tf.keras.utils.to_categorical(encoded_labels) # One-hot encode for CategoricalCrossentropy
class_names = label_encoder.classes_

# Save label encoder
encoder_path = os.path.join(SAVE_DIR, 'label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved to: {encoder_path}")

# Stratified split using integer labels for stratification
print(f"\nSplitting data (Test size: {TEST_SIZE})...")
X_train, X_test, y_train, y_test, encoded_labels_train, encoded_labels_test = train_test_split(
    X, y, encoded_labels, # Split X, one-hot y, and integer labels together
    test_size=TEST_SIZE,
    random_state=42,
    stratify=encoded_labels, # Use integer labels for stratification
    shuffle=True
)

print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- Enhanced Model Architecture (Using Functional API) ---
def create_model(input_shape, num_classes):
    print("Creating model architecture...")
    inputs = Input(shape=input_shape)

    # Block 1
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True,
                         kernel_regularizer=l2(0.001),
                         recurrent_regularizer=l2(0.001)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Attention Layer
    attention_out = Attention()([x, x]) # Self-attention

    # Block 2
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, # Still need sequences for next LSTM
                         kernel_regularizer=l2(0.001),
                         recurrent_regularizer=l2(0.001)))(attention_out) # Input from Attention
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Block 3
    x = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=False, # False before Dense layers
                         kernel_regularizer=l2(0.001)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Dense Layers
    x = Dense(LSTM_UNITS, activation='swish', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(LSTM_UNITS // 2, activation='swish', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create Model
    model = Model(inputs=inputs, outputs=outputs)

    # --- Optimizer: Use float learning rate for ReduceLROnPlateau compatibility ---
    # REMOVED: lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(...)
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE) # Use initial float LR
    print(f"Optimizer: Adam with initial LR = {INITIAL_LEARNING_RATE}")

    # Compile Model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    print("Model compiled.")
    return model

# --- Enhanced Training Process ---
best_accuracy = 0.0
best_model_path_final = os.path.join(SAVE_DIR, 'sign_language_recognition_best.h5') # Final best model path
best_history = None
best_session = 0
loaded_best_model_for_eval = None # To store the best model object loaded at the end

# Callbacks
print("\nSetting up callbacks...")
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=PATIENCE,
    verbose=1,
    restore_best_weights=True, # Restores model weights from the epoch with the best val_accuracy
    mode='max'
)

# ReduceLROnPlateau works with optimizers initialized with a float learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,             # new_lr = lr * factor
    patience=PATIENCE // 2, # Number of epochs with no improvement after which learning rate will be reduced.
    verbose=1,
    min_lr=1e-6,            # Lower bound on the learning rate.
    mode='max'
)

# Checkpoint to save the best model *of the current session* temporarily
checkpoint_filepath_temp = os.path.join(SAVE_DIR, 'best_session_model_temp.h5')
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath_temp,
    monitor='val_accuracy',
    save_best_only=True, # Save only when val_accuracy improves
    mode='max',
    verbose=0, # Less verbose checkpoint saving
    save_weights_only=False # Save the full model (needed for loading later if not using restore_best_weights)
)
print(f"Callbacks: EarlyStopping (patience={PATIENCE}), ReduceLROnPlateau (patience={PATIENCE//2}), ModelCheckpoint (temp path: {checkpoint_filepath_temp})")

all_histories = []

print(f"\n--- Starting Training ({NUM_SESSIONS} sessions max) ---")
for session in range(NUM_SESSIONS):
    print(f"\n=== Training Session {session+1}/{NUM_SESSIONS} ===")

    # Create a *new* instance of the model for each session to reset weights
    model = create_model((SEQUENCE_LENGTH, NUM_FEATURES), y_train.shape[1])

    # Print model summary only once
    if session == 0:
        model.summary()

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS_PER_SESSION,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[early_stop, reduce_lr, model_checkpoint], # Add checkpoint callback
        verbose=1
    )
    all_histories.append(history)

    # Note: Because restore_best_weights=True in EarlyStopping,
    # the 'model' object here already holds the weights from the epoch
    # with the best validation accuracy within this session.
    # No need to manually load from checkpoint_filepath_temp unless restore_best_weights is False.

    # Evaluate the best state of the model from *this session*
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Session {session+1} Completed. Best Validation Accuracy in this session: {max(history.history['val_accuracy']):.4f}")
    print(f"Session {session+1} Test Accuracy (at best val epoch): {accuracy:.4f}, Loss: {loss:.4f}")

    # Check if this session produced a better model than all previous sessions
    session_best_val_acc = max(history.history['val_accuracy']) # Use max val_accuracy achieved in the session history
    if session_best_val_acc > best_accuracy: # Compare based on validation accuracy
        best_accuracy = session_best_val_acc # Update best *validation* accuracy
        # Save the current model state (which is the best found *so far* across all sessions)
        model.save(best_model_path_final)
        best_history = history # Store history of the best session
        best_session = session + 1
        print(f"🎉 New overall best model found in session {best_session}!")
        print(f"   Best Validation Accuracy so far: {best_accuracy:.4f}")
        print(f"   Model saved to: {best_model_path_final}")
    else:
         print(f"Session {session+1} did not improve overall best validation accuracy ({best_accuracy:.4f})")

    # Optional: Clean up the temporary checkpoint file if desired
    # if os.path.exists(checkpoint_filepath_temp):
    #     os.remove(checkpoint_filepath_temp)

print("\n--- Training Finished ---")

# --- Save Final Best Model ---
if best_session > 0:
    print(f"\n✅ Best overall model was found in session {best_session} with Validation Accuracy: {best_accuracy:.4f}")
    print(f"   Final best model saved at: {best_model_path_final}")
    # Load the absolute best model saved for final evaluation
    try:
        print("Loading the best saved model for final evaluation...")
        loaded_best_model_for_eval = tf.keras.models.load_model(best_model_path_final)
    except Exception as e:
        print(f"Error loading the final best model: {e}")
        loaded_best_model_for_eval = None # Fallback or handle error
else:
    print("\n⚠️ No improvement found across training sessions, or training didn't run.")
    print("   No final best model was saved.")
    # Optionally save the last model state if needed:
    # model.save(os.path.join(SAVE_DIR, 'sign_language_recognition_last.h5'))


# --- Save Mean and Std for Normalization ---
# These were calculated after padding and before splitting
print("\n--- Saving Normalization Parameters ---")
try:
    mean_path = os.path.join(SAVE_DIR, 'normalization_mean.npy')
    std_path = os.path.join(SAVE_DIR, 'normalization_std.npy')
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"Normalization mean saved to: {mean_path} (shape: {mean.shape})")
    print(f"Normalization std saved to: {std_path} (shape: {std.shape})")
except NameError:
    print("\n⚠️ Warning: Could not find 'mean' and 'std' variables to save. Ensure they were calculated correctly.")
except Exception as e:
    print(f"\n⚠️ Error saving normalization parameters: {e}")


# --- Enhanced Evaluation (using the loaded best model) ---
def evaluate_model(model, X_test_data, y_test_data, class_names_list):
    if model is None:
        print("Error: No model available for evaluation.")
        return None
    print("\n--- Evaluating Model on Test Set ---")
    # Get predictions
    y_pred_proba = model.predict(X_test_data)
    y_pred_indices = np.argmax(y_pred_proba, axis=1)
    y_true_indices = np.argmax(y_test_data, axis=1) # Convert one-hot y_test back to indices

    # Classification report
    print("\nClassification Report:")
    # Ensure target_names correspond correctly to y_true/y_pred indices
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names_list, zero_division=0, digits=4)
    print(report)

    # Confidence analysis
    confidences = np.max(y_pred_proba, axis=1)
    print("\nPrediction Confidence Analysis:")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Minimum confidence: {np.min(confidences):.4f}")
    print(f"Confidence std dev: {np.std(confidences):.4f}")

    # Per-class performance
    results = []
    for i, class_name in enumerate(class_names_list): # Iterate using label encoder's classes
        class_mask = (y_true_indices == i)
        total = np.sum(class_mask)

        if total == 0:
            # print(f"Warning: No test samples for class '{class_name}' (index {i}).")
            continue # Skip classes with no samples in the test set

        correct = np.sum((y_true_indices == i) & (y_pred_indices == i))
        accuracy_perc = (correct / total) * 100
        class_confidences = confidences[class_mask]
        avg_conf = np.mean(class_confidences) if len(class_confidences) > 0 else 0
        min_conf = np.min(class_confidences) if len(class_confidences) > 0 else 0

        results.append({
            "Sign": class_name,
            "Correct": int(correct),
            "Total": int(total),
            "Accuracy (%)": round(accuracy_perc, 2),
            "Avg Confidence": round(avg_conf, 4),
            "Min Confidence": round(min_conf, 4)
        })

    if not results:
        print("Error: No per-class results generated (maybe no test samples?).")
        return None

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy (%)", ascending=False)

    print("\nDetailed Class Performance:")
    print(results_df.to_string(index=False))

    # Save detailed performance
    perf_path = os.path.join(SAVE_DIR, "detailed_performance.csv")
    results_df.to_csv(perf_path, index=False)
    print(f"Detailed performance saved to: {perf_path}")
    return results_df

# Run evaluation on the best model found (if loaded successfully)
if loaded_best_model_for_eval:
    performance_df = evaluate_model(loaded_best_model_for_eval, X_test, y_test, class_names)
else:
     print("\nSkipping evaluation as the best model could not be loaded.")


# --- Enhanced Learning Curves (Plot history of the best session) ---
def plot_enhanced_history(history_obj, session_num):
    if history_obj is None or not hasattr(history_obj, 'history'):
        print("No valid history object to plot.")
        return

    history_dict = history_obj.history
    acc = history_dict.get('accuracy')
    val_acc = history_dict.get('val_accuracy')
    loss = history_dict.get('loss')
    val_loss = history_dict.get('val_loss')

    if not all([acc, val_acc, loss, val_loss]):
        print("History object missing required keys (accuracy, val_accuracy, loss, val_loss). Available keys:", history_dict.keys())
        return

    epochs_range = range(len(acc)) # Determine epochs from history length

    plt.figure(figsize=(16, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy', marker='.', linestyle='-')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='.', linestyle='-')
    plt.title(f'Best Session ({session_num}) - Accuracy', pad=20)
    plt.xlabel('Epochs', labelpad=10)
    plt.ylabel('Accuracy', labelpad=10)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(bottom=max(0, min(acc[-1], val_acc[-1]) - 0.1)) # Adjust y-axis start

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss', marker='.', linestyle='-')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='.', linestyle='-')
    plt.title(f'Best Session ({session_num}) - Loss', pad=20)
    plt.xlabel('Epochs', labelpad=10)
    plt.ylabel('Loss', labelpad=10)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.ylim(top=min(5, max(loss[0], val_loss[0]) + 0.5)) # Adjust y-axis end

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f'enhanced_training_history_best_session_{session_num}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training history plot for best session ({session_num}) to: {plot_path}")

# Plot history of the best session if available
if best_history:
    plot_enhanced_history(best_history, best_session)
else:
    print("Skipping history plot as no best session history was recorded.")


# --- Enhanced Prediction Analysis (using the loaded best model and full dataset X) ---
def enhanced_prediction_analysis(model, X_data, true_labels_encoded, class_names_list, threshold=0.7):
    if model is None:
        print("Error: No model available for prediction analysis.")
        return None
    print("\n--- Running Enhanced Prediction Analysis on Full Dataset ---")
    probabilities = model.predict(X_data)
    detailed_predictions = []
    confidence_distribution = []

    # Ensure true_labels_encoded are indeed integer indices
    if true_labels_encoded.ndim > 1:
         print("Warning: Received multi-dimensional true labels, attempting argmax...")
         true_indices = np.argmax(true_labels_encoded, axis=1)
    else:
         true_indices = true_labels_encoded # Assume already encoded indices

    # Ensure class_names map correctly to indices
    num_classes_model = probabilities.shape[1]
    num_classes_names = len(class_names_list)
    if num_classes_names != num_classes_model:
         print(f"Warning: Mismatch between number of class names ({num_classes_names}) and model output dimension ({num_classes_model})")
         use_indices_as_names = True
    else:
         use_indices_as_names = False


    for i, (proba, true_idx) in enumerate(zip(probabilities, true_indices)):
        pred_index = np.argmax(proba)
        confidence = proba[pred_index]
        confidence_distribution.append(confidence)

        sorted_indices = np.argsort(proba)[::-1]

        # Get true and predicted labels carefully
        try:
            true_label_str = class_names_list[true_idx] if not use_indices_as_names else str(true_idx)
            pred_label_str = class_names_list[pred_index] if not use_indices_as_names else str(pred_index)
            top_choices = [(class_names_list[idx] if not use_indices_as_names else str(idx), float(proba[idx])) for idx in sorted_indices[:3]]
        except IndexError:
            print(f"Error: Index out of bounds accessing class_names. True index: {true_idx}, Pred index: {pred_index}, Num classes: {num_classes_names}")
            true_label_str = f"ErrorIdx_{true_idx}"
            pred_label_str = f"ErrorIdx_{pred_index}"
            top_choices = [("Error", 0.0)]


        if confidence < threshold:
            status = "LOW_CONFIDENCE"
        elif pred_label_str == true_label_str:
            status = "CORRECT"
        else:
            status = "INCORRECT"

        detailed_predictions.append({
            "index": i,
            "true_label": true_label_str,
            "predicted_label": pred_label_str,
            "confidence": float(confidence),
            "status": status,
            "top_choices": top_choices
        })

    # Save detailed predictions
    pred_df = pd.DataFrame(detailed_predictions)
    pred_analysis_path = os.path.join(SAVE_DIR, "enhanced_predictions_analysis.csv")
    pred_df.to_csv(pred_analysis_path, index=False)
    print(f"Enhanced prediction analysis saved to: {pred_analysis_path}")

    # Confidence distribution analysis plot
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_distribution, bins=30, edgecolor='black')
    plt.title('Prediction Confidence Distribution (Full Dataset)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Samples')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    conf_plot_path = os.path.join(SAVE_DIR, 'confidence_distribution.png')
    plt.savefig(conf_plot_path)
    plt.close()
    print(f"Saved confidence distribution plot to: {conf_plot_path}")

    return pred_df

# Run prediction analysis on the full dataset (X) using the best model
if loaded_best_model_for_eval:
    # Use encoded_labels (single integers) for true labels here, corresponding to the full dataset X
    prediction_df = enhanced_prediction_analysis(loaded_best_model_for_eval, X, encoded_labels, class_names, threshold=0.7)
else:
    print("\nSkipping prediction analysis as the best model could not be loaded.")


# --- Final Report ---
print("\n=== FINAL OVERALL REPORT ===")
if best_session > 0 and loaded_best_model_for_eval:
    final_loss, final_accuracy = loaded_best_model_for_eval.evaluate(X_test, y_test, verbose=0)
    print(f"Best Model (from session {best_session}):")
    print(f"  - Final Test Accuracy: {final_accuracy:.4f}")
    print(f"  - Final Test Loss:     {final_loss:.4f}")
    print(f"  - Best Validation Accuracy during training: {best_accuracy:.4f}")
    print(f"  - Model saved to:      {best_model_path_final}")
    print(f"  - Label Encoder saved: {encoder_path}")
    print(f"  - Normalization params saved (mean/std .npy files)")
else:
    print("Training did not complete successfully or no improvement was found.")
print(f"\nConfiguration used:")
print(f"  - Classes trained:     {len(class_names)}")
print(f"  - Training samples:    {len(X_train)}")
print(f"  - Testing samples:     {len(X_test)}")
print(f"  - Sequence Length:     {SEQUENCE_LENGTH}")
print(f"  - Features per frame:  {NUM_FEATURES}")
print(f"  - LSTM Units:          {LSTM_UNITS}")
print(f"  - Batch Size:          {BATCH_SIZE}")
print(f"  - Initial LR:          {INITIAL_LEARNING_RATE}")

# Save overall class distribution
class_dist = pd.Series(filtered_labels).value_counts().reset_index()
class_dist.columns = ['Class', 'Count']
dist_path = os.path.join(SAVE_DIR, 'class_distribution.csv')
class_dist.to_csv(dist_path, index=False)
print(f"\nSaved final class distribution to: {dist_path}")

print("\n✅ Enhanced training and evaluation script finished!")

