# Implementing-and-Optimizing-Variational-Autoencoders-VAEs-for-Anomaly-Detection
Implementing VAE - KL Divergence - AUC-ROC, Precision Recall curves for Anomaly Detection
## 1. Objective

The objective of this project is to detect anomalies in high-dimensional data using a Variational Autoencoder (VAE).

The model is trained only on normal data and anomalies are identified based on deviations in reconstruction behavior.

## 2. Dataset Design

A synthetic dataset with 30 features was generated using a latent variable model.

- Low-dimensional latent variables were sampled from a standard normal distribution

- A linear transformation mapped latent variables to feature space

- Small Gaussian noise was added to simulate realistic measurements

This process introduces natural correlations between features, closely matching real-world data behavior.

### Anomaly Injection

Three structured anomaly types were introduced:

1. **Mean Shift**

  A subset of features was shifted by a constant value, simulating systematic drift.

2. **Correlation Break**

  One feature was decoupled from latent structure and replaced with independent random values.

3. **Variance Explosion**

  Large noise was injected into a specific feature, simulating instability or sensor faults.

Only normal samples were used for training.

The test set contained both normal and anomalous samples with labels used strictly for evaluation.

3. Model Architecture
The Variational Autoencoder was implemented using a fully connected (MLP-based) architecture.

Encoder
The encoder consists of:

Input layer of size 30 (feature dimension)
Fully connected layer with 128 units and ReLU activation
Fully connected layer with 64 units and ReLU activation
Two parallel linear layers producing:
Mean vector (μ)
Log-variance vector (log σ²)
ReLU activation functions were chosen to introduce non-linearity while maintaining stable gradients during training.

Latent Space
The latent space dimensionality was treated as a tunable hyperparameter. Experiments were conducted with latent dimensions of 2, 4, 8, and 16 to analyze representational capacity and anomaly separation performance.

Decoder
The decoder mirrors the encoder structure:

Fully connected layer with 64 units and ReLU activation
Fully connected layer with 128 units and ReLU activation
Output layer of size 30 to reconstruct the input features
No activation function was applied at the output layer, as the data is continuous and reconstruction error is computed using Mean Squared Error (MSE).

## 4. Training Strategy

The model was trained using a combined loss:

- Mean Squared Error (reconstruction loss)

- KL divergence (regularization)

KL annealing was applied to gradually increase the influence of KL divergence during training, improving stability and preventing posterior collapse.

## 5. Anomaly Scoring Method

Anomaly scores were computed using per-sample reconstruction error.

Higher reconstruction error indicates greater deviation from learned normal patterns and a higher likelihood of being anomalous.

A statistical threshold was defined using the mean and standard deviation of reconstruction errors.

## 6. Evaluation Metrics

Model performance was evaluated using:

- **ROC-AUC** to measure ranking quality across thresholds

- **PR-AUC** to account for class imbalance

- **Precision** and **Recall** at a fixed threshold

A conservative threshold resulted in high precision with moderate recall, suitable for scenarios where false alarms are costly.

## 7. Latent Dimension Experiments

Experiments were conducted using multiple latent dimensions: 2, 4, 8, and 16.

### Observations:

- Smaller latent dimensions underfit the data and reduced anomaly separation

- Increasing latent dimension improved ROC-AUC and PR-AUC

- Performance gains diminished beyond a moderate latent size

This demonstrates a trade-off between representational capacity and regularization.

KL Weight Schedule Analysis
A linear KL annealing schedule was used, gradually increasing the KL weight from 0 to 1 during the early training epochs. This strategy allowed the model to first focus on accurate reconstruction before enforcing latent space regularization. Empirically, KL annealing resulted in stable convergence and improved anomaly separation compared to using a fixed KL weight from the start.

## 8. Conclusion

The VAE successfully learned the distribution of normal data and detected multiple structured anomaly types.

Results highlight the importance of principled dataset design, latent space capacity, and proper evaluation metrics for effective anomaly detection.
