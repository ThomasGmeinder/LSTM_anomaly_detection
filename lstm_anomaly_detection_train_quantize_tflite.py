# %% [markdown]
# # LSTM trained for Anomaly Detection in Time-Series Data
# This example creates an LSTM in Tensorflow for the purpose of detecting anomalies in time series data.
# The model is trained with sinusoidal data plus noise (normal case). Random spikes are added to generate anomalies.
# The trained model is then saved, quantized to int8 and converted to .tflite format.

# Finally, inference is run with the original fp32 model and the quantized .tflite model to determine the difference in accuracy.


# %%
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
import os
import matplotlib.pyplot as plt
import random

# %% [markdown]
# ## Constants
# %%
SEED=42 
random.seed(SEED) 

UNROLL = True  # Use dynamic unrolling for LSTM
    
FEATURES = 1
SEQ_LEN = 100
TRAINING_BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 1
NUM_SAMPLES = 1000  # Increased for better generalization
HIDDEN_UNITS = 32   # Reduced to prevent overfitting

model_file = f"lstm_model_SL{SEQ_LEN}{'_unrolled' if UNROLL else ''}_fp32.keras"
tflite_file = f"lstm_model_SL{SEQ_LEN}{'_unrolled' if UNROLL else ''}_int8_quantized.tflite"

# suppress warnings
import logging
import time
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = +WARNING, 3 = ERROR only

# %% [markdown]
# ## Prepare training data
# ### Generate synthetic time-series data for anomaly detection
# %% 
X = np.zeros((NUM_SAMPLES, SEQ_LEN, FEATURES), dtype=np.float32)
y = np.zeros((NUM_SAMPLES, 1), dtype=np.float32)

for i in range(NUM_SAMPLES):

    # Normal: smooth sinusoidal pattern with noise
    t = np.linspace(0, 8 * np.pi, SEQ_LEN)
    series = np.sin(t) + 0.1 * np.random.normal(0, 0.2, size=SEQ_LEN)
    label = 0.0
    
    if i % 3 == 0:
        # inject spikes anomaly for 30 percent of samples
        # random nbumber of spikes between 1 and 5
        num_spikes = np.random.randint(1, 6)
        spike_indices = np.random.choice(SEQ_LEN, size=num_spikes, replace=False)
        # random spike magnitude between 1 and 5
        series[spike_indices] += np.random.uniform(1, 6, size=num_spikes)
        label = 1.0

    X[i, :, 0] = series
    y[i, 0] = label

# %% [markdown]
# ### Normalize input data to improve training stability
# %% 
X = (X - np.mean(X)) / np.std(X)

# %% [markdown]
# ### Split data into train/val/test
# 70% train, 15% validation, 15% test
# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)


# %% [markdown]
# ## Load or create/train the model
# %%
print("\n## Loading or creating the LSTM model...")
if os.path.exists(model_file):
    print(f"Loading existing model from {model_file}")
    model = tf.keras.models.load_model(model_file)
else:
    # Define the model with regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, FEATURES)),
        tf.keras.layers.LSTM(HIDDEN_UNITS, unroll=UNROLL, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(f"No existing model {model_file} found. A new model will be created.")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=TRAINING_BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )
    # save trained model
    print("Training complete. Saving the model as ", model_file)
    model.save(model_file)

    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()   
    
# %% [markdown]
# ## Evaluate on test data
# Create a inference model from the training model to guarantee batch size of 1 and no dropouts
# This is also important for the quanizatin and tflite conversion going forward
# %%
print("\n## Creating inference model from the trained model")
inference_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, FEATURES), batch_size=1),
    tf.keras.layers.LSTM(HIDDEN_UNITS, unroll=UNROLL),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

inference_model.set_weights(model.get_weights())

inference_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

print("\n## Evaluating the model on test data")
loss, accuracy = inference_model.evaluate(X_test, y_test, batch_size=INFERENCE_BATCH_SIZE)
print(f"FP32 Test Accuracy from evaluate: {accuracy:.8f}, Loss: {loss:.8f}")

fp32_preds = inference_model.predict(X_test, batch_size=INFERENCE_BATCH_SIZE)

# double check accuracy
fp32_preds_rounded = (fp32_preds > 0.5).astype(np.float32)
fp32_accuracy = np.mean(fp32_preds_rounded.flatten() == y_test.flatten())
print(f"FP32 Model Test Accuracy from predict: {fp32_accuracy:.8f}")

# %% [markdown]
# ## Visualise some predictions
# %%
fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
fig.suptitle("Sample Predictions from FP32 Model")

for i, ax in enumerate(axes):
    ax.plot(X_test[i].squeeze(), label='Sensor Signal')
    ax.set_title(f"Anomaly Label: {int(y_test[i][0])}, Predicted: {fp32_preds[i][0]:.2f}")
    ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Quantize and convert to the TFLite format

# %%
print("\n## Quantize and convert to the TFLite format")
# Quantization representative dataset
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1]]

# Convert to quantized TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

if not UNROLL:
    converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
    converter._experimental_lower_tensor_list_ops = False

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Save quantized model
tflite_model = converter.convert()
pathlib.Path(tflite_file).write_bytes(tflite_model)
print(f"INT8-quantized model saved as {tflite_file}")

# %% [markdown]
# ## Evaluate TFLite model
# %%
print("\n## Evaluating the INT8 TFLite model")
interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_tflite_model(interpreter, X):

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    preds = []
    times = []
    for i in range(len(X)):
        x = X[i:i+1]
        x_int8 = np.round(x / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], x_int8)
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()

        times.append(end_time - start_time)

        output = interpreter.get_tensor(output_details[0]['index'])
        output = (output.astype(np.float32) - output_zero_point) * output_scale
        preds.append(output[0])

    mean_time = sum(times) / len(times)
    print(f"Mean tflite inference time: {mean_time:.6f} seconds")
    return np.array(preds)

tflite_preds = run_tflite_model(interpreter, X_test)
tflite_preds = (tflite_preds > 0.5).astype(np.float32)
int8_accuracy = np.mean(tflite_preds.flatten() == y_test.flatten())
print(f"INT8 Quantized Model Test Accuracy: {int8_accuracy:.8f}")

# %% [markdown]
# ## Prepare test data for inference runs on embedded target

# %%
# Compute X_test_int8 and y_test_int8 using the same quantization parameters
input_scale, input_zero_point = input_details[0]['quantization']
X_test_int8 = np.round(X_test / input_scale + input_zero_point).astype(np.int8)
y_test_int8 = y_test.astype(np.int8)  # y is already binary float32 values (0, 1), so we can convert it directly to int8

# Output X_test_int8 and y_test_int8 to txt files for C++ use
np.savetxt("X_test_int8.txt", X_test_int8.reshape(-1, SEQ_LEN * FEATURES), fmt='%d')
np.savetxt("y_test_int8.txt", y_test_int8, fmt='%d')


# %% [markdown]
# ## Compare accuracy of FP32 and INT8 models
# %%
print("\n## Comparing accuracy of FP32 and INT8 models")
print(f"FP32 Test Accuracy: {accuracy:.8f}")
print(f"INT8 Quantized Model Test Accuracy: {int8_accuracy:.8f}")
# accuracy difference should be minimal, around 1-2% loss in accuracy
print(f"Difference in accuracy: {accuracy - int8_accuracy:.8f}") 
# print hyperparameters
print(f"Hyperparameters: UNROLL={UNROLL}, FEATURES={FEATURES}, SEQ_LEN={SEQ_LEN}, "
      f"TRAINING_BATCH_SIZE={TRAINING_BATCH_SIZE}, INFERENCE_BATCH_SIZE={INFERENCE_BATCH_SIZE}, "
      f"NUM_SAMPLES={NUM_SAMPLES}, HIDDEN_UNITS={HIDDEN_UNITS}")
