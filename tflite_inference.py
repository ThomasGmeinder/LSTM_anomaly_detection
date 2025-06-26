# %% [markdown]
# # Run Inference to Evaluate the TFLite model

# Import necessary libraries
# %%
import tensorflow as tf
import numpy as np
import time

# input tflite_file with argparse
import argparse
parser = argparse.ArgumentParser(description='Run inference on a TFLite model.')
parser.add_argument('--tflite_file', type=str, required=True, help='Path to the TFLite model file.')
# input X_test_int8.txt and y_test_int8.txt with argparse
parser.add_argument('--X_test_file', type=str, default='X_test_int8.txt', help='Path to the test input data file (default: X_test_int8.txt)')
parser.add_argument('--y_test_file', type=str, default='y_test_int8.txt', help='Path to the test labels data file (default: y_test_int8.txt)')
args = parser.parse_args()

# Load the test dataset from X_test_int8.txt and y_test_int8.txt
print(f"## Loading the test dataset from {args.X_test_file} and {args.y_test_file}")
X_test_int8 = np.loadtxt(args.X_test_file, delimiter=' ').astype(np.int8)
y_test_int8 = np.loadtxt(args.y_test_file, delimiter=' ').astype(np.int8)
X_test_int8 = X_test_int8.reshape((X_test_int8.shape[0], X_test_int8.shape[1], 1))  # Reshape to (samples, timesteps, features)_int8 

print("\n## Evaluating the INT8 TFLite model")
interpreter = tf.lite.Interpreter(model_path=args.tflite_file)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_tflite_model(interpreter, X_test_int8):
    preds = []
    times = []
    for i in range(len(X_test_int8)):
        x_int8 = X_test_int8[i:i+1]

        interpreter.set_tensor(input_details[0]['index'], x_int8)
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()

        times.append(end_time - start_time)

        output = interpreter.get_tensor(output_details[0]['index'])
        preds.append(output[0])

    mean_time = sum(times) / len(times)
    print(f"Mean tflite inference time: {mean_time:.6f} seconds")
    return np.array(preds)

tflite_outputs_int8 = run_tflite_model(interpreter, X_test_int8)
# Convert predictions to binary (0 or 1) and calculate accuracy
output_scale, output_zero_point = output_details[0]['quantization']
tflite_preds = (tflite_outputs_int8.astype(np.float32) - output_zero_point) * output_scale  # Dequantize the predictions
tflite_preds = (tflite_preds > 0.5).astype(np.int8)  # Convert predictions to binary (0 or 1)

int8_accuracy = np.mean(tflite_preds.flatten() == y_test_int8.flatten())
print(f"Test Accuracy for INT8 Quantized Model {args.tflite_file}: {int8_accuracy:.8f}")