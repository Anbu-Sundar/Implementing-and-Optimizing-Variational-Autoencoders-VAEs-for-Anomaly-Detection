import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.datasets import fetch_kddcup99 
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# -----------------------------
# 1. Load KDD Cup 99 dataset
# -----------------------------
print("Loading dataset...")
ds = tfds.load('kddcup99', split='train', as_supervised=True, data_dir="./tfds_data")

data, labels = [], []
for x, y in tfds.as_numpy(ds):
    data.append(x.astype(np.float32))
    labels.append(y)

X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)

# Normalize features
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

input_dim = X.shape[1]
latent_dim = 16

# -----------------------------
# 2. Define Sampling Layer (VAE)
# -----------------------------
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# -----------------------------
# 3. Build Encoder & Decoder
# -----------------------------
# Encoder
inputs = tf.keras.Input(shape=(input_dim,))
h = tf.keras.layers.Dense(64, activation="relu")(inputs)
z_mean = tf.keras.layers.Dense(latent_dim)(h)
z_log_var = tf.keras.layers.Dense(latent_dim)(h)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
h_dec = tf.keras.layers.Dense(64, activation="relu")(latent_inputs)
outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid")(h_dec)
decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")

# VAE model
z_mean, z_log_var, z = encoder(inputs)
recon = decoder(z)
vae = tf.keras.Model(inputs, recon, name="vae")

# -----------------------------
# 4. VAE Loss
# -----------------------------
recon_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, recon))
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae.add_loss(recon_loss + kl_loss)
vae.compile(optimizer="adam")

# -----------------------------
# 5. Train VAE
# -----------------------------
print("Training VAE...")
vae.fit(X, X, epochs=10, batch_size=128)

# -----------------------------
# 6. Baseline Autoencoder (AE)
# -----------------------------
ae_inputs = tf.keras.Input(shape=(input_dim,))
h = tf.keras.layers.Dense(64, activation="relu")(ae_inputs)
ae_outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid")(h)
ae = tf.keras.Model(ae_inputs, ae_outputs, name="ae")
ae.compile(optimizer="adam", loss="mse")

print("Training AE...")
ae.fit(X, X, epochs=10, batch_size=128)

# -----------------------------
# 7. Evaluate Anomaly Detection
# -----------------------------
print("Evaluating anomalies...")

# VAE errors
vae_recon = vae.predict(X)
vae_errors = np.mean(np.square(X - vae_recon), axis=1)
