import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, skew, kurtosis
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.set_seed(42)

# Load and preprocess data
data = pd.read_csv('data.csv')
values = data['Indexvalue'].values.reshape(-1, 1)
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

# Construct sequences
seq_len = 24
sequences = np.array([values_scaled[i:i+seq_len] for i in range(len(values_scaled) - seq_len + 1)])
X = sequences
n_sequences = X.shape[0]

# Define metric computation function
def compute_metrics(real_seqs, synthetic_seqs, lag=10):
    real_flat = real_seqs.reshape(-1)
    synthetic_flat = synthetic_seqs.reshape(-1)
    ks_stat, _ = ks_2samp(real_flat, synthetic_flat)
    skew_diff = abs(skew(real_flat) - skew(synthetic_flat))
    kurt_diff = abs(kurtosis(real_flat) - kurtosis(synthetic_flat))
    acf_real = np.mean([acf(seq[:, 0], nlags=lag) for seq in real_seqs], axis=0)
    acf_synthetic = np.mean([acf(seq[:, 0], nlags=lag) for seq in synthetic_seqs], axis=0)
    acf_mse = np.mean((acf_real - acf_synthetic) ** 2)
    return ks_stat, skew_diff, kurt_diff, acf_mse

# Generate AR(2) baseline
model = AutoReg(values_scaled[:, 0], lags=2)
model_fit = model.fit()
synthetic_ar_sequences = []
for _ in range(n_sequences):
    start_idx = np.random.randint(0, len(values_scaled) - seq_len)
    initial = values_scaled[start_idx:start_idx + 2, 0]
    synthetic_seq = list(initial)
    for t in range(2, seq_len):
        pred = model_fit.params[0] + model_fit.params[1] * synthetic_seq[-1] + model_fit.params[2] * synthetic_seq[-2]
        noise = np.random.normal(0, np.sqrt(model_fit.sigma2))
        next_val = np.clip(pred + noise, 0, 1)  # MinMax scaling 
        synthetic_seq.append(next_val)
    synthetic_ar_sequences.append(synthetic_seq)
synthetic_ar_sequences = np.array(synthetic_ar_sequences).reshape(-1, seq_len, 1)
ar_metrics = compute_metrics(X, synthetic_ar_sequences)

# Define TCN block
def tcn_block(x, filters=32, kernel_size=3, dilation_rate=1):
    for i in range(3):
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="causal",
                                  dilation_rate=dilation_rate ** i, activation="relu")(x)
    return x

# Build Generator
def build_generator():
    inputs = tf.keras.layers.Input(shape=(seq_len, 1))
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = tcn_block(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24, return_sequences=True))(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs, outputs)

# Build Discriminator
def build_discriminator():
    inputs = tf.keras.layers.Input(shape=(seq_len, 1))
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = tcn_block(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24, return_sequences=True))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

# Initialize models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers and loss
g_optimizer = tf.keras.optimizers.Adam(0.0005)
d_optimizer = tf.keras.optimizers.Adam(0.0005)
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Training step
@tf.function
def train_step(real_data):
    batch_size = tf.shape(real_data)[0]
    noise = tf.random.normal([batch_size, seq_len, 1])
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_data = generator(noise, training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)
        
        d_loss_real = binary_cross_entropy(tf.ones_like(real_output), real_output)
        d_loss_fake = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake
        g_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)
    
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    
    return d_loss, g_loss

# Training loop
batch_size = 128
iterations = 1000
dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(1000).batch(batch_size)

g_losses = []
d_losses = []

for iteration in range(iterations):
    for batch in dataset:
        d_loss, g_loss = train_step(batch)
    g_losses.append(g_loss.numpy())
    d_losses.append(d_loss.numpy())
    if (iteration + 1) % 200 == 0:
        print(f"Iteration {iteration + 1}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

# Generate synthetic data
noise = tf.random.normal([n_sequences, seq_len, 1])
synthetic_data = generator(noise, training=False).numpy()

# Compute  timeGAN metrics
proposed_gan_metrics = compute_metrics(X, synthetic_data)

# Display results
print("\nModel Comparison")
print("Model\t\tKS Statistic\tSkewness Diff\tKurtosis Diff\tACF MSE")
print(f"Proposed GAN\t{proposed_gan_metrics[0]:.4f}\t\t{proposed_gan_metrics[1]:.4f}\t\t"
      f"{proposed_gan_metrics[2]:.4f}\t\t{proposed_gan_metrics[3]:.4f}")
print(f"AR(2)\t\t{ar_metrics[0]:.4f}\t\t{ar_metrics[1]:.4f}\t\t"
      f"{ar_metrics[2]:.4f}\t\t{ar_metrics[3]:.4f}")

# loss curves
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss', color='purple')
plt.plot(d_losses, label='Discriminator Loss', color='green')
plt.title('Training Loss Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.close()