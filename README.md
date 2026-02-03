# Implementing-and-Optimizing-Variational-Autoencoders-VAEs-for-Anomaly-Detection
Implementing VAE - KL Divergence - AUC-ROC, Precision Recall curves for Anomaly Detection
1. Project Overview
-------------------
This project implements a Variational Autoencoder (VAE) for unsupervised anomaly
detection using the KDD Cup 99 dataset. The VAE is trained to learn the latent space
distribution of normal network traffic data. Anomalies are detected based on
reconstruction error, assuming anomalous samples reconstruct poorly compared to
normal samples.

The implementation is done from scratch using the framework and evaluated
using standard anomaly detection metrics such as AUC-ROC and Precision-Recall AUC.


2. Dataset
----------
Dataset: KDD Cup 1999 (10% subset, SA configuration)

- Normal samples are used for training the VAE.
- Both normal and anomalous samples are used for testing.
- Features are standardized using StandardScaler.

Source:
The dataset is loaded directly using sklearn.datasets.fetch_kddcup99.


3. Methodology
--------------
- A Variational Autoencoder (VAE) is implemented with:
  - Encoder network
    - Latent space with reparameterization trick
      - Decoder network

      - Loss function:
        - Reconstruction loss (Mean Squared Error)
          - KL-Divergence loss

          Total Loss = Reconstruction Loss + KL Divergence Loss

          - The model is trained only on normal data to capture the normal data distribution.
          - During inference, reconstruction error is used as the anomaly score.


          4. Model Architecture
          ---------------------
          Encoder:
          - Input Layer → 128 → 64
          - Latent variables: Mean (μ) and Log Variance (log σ²)
          - Latent Dimension:
          - 16

          Decoder:
          - Latent → 64 → 128 → Output Layer


5. Evaluation Metrics
---------------------
The model performance is evaluated using:
- AUC-ROC (Area Under ROC Curve)
- AUC-PR (Area Under Precision-Recall Curve)

A Precision-Recall curve is plotted to visualize anomaly detection performance.


6. Software Requirements
------------------------
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

All dependencies are installed using pip.


7. How to Run the Project
------------------------

Step 1: Create virtual environment
----------------------------------
Windows:
    python -m venv venv
    venv\Scripts\activate

Mac/Linux:
    python3 -m venv venv
    source venv/bin/activate


Step 2: Install dependencies
----------------------------
    pip install torch torchvision torchaudio
    pip install numpy pandas scikit-learn matplotlib


Step 3: Run the program
-----------------------
    python vae_kdd99.py


8. Output
---------
- Training loss printed for each epoch
- AUC-ROC and AUC-PR scores displayed in the terminal
- Precision-Recall curve plotted using Matplotlib


9. Conclusion
-------------
The Variational Autoencoder successfully learns the latent representation of normal
network traffic and effectively detects anomalies based on reconstruction error.
The results demonstrate that VAEs are suitable for unsupervised anomaly detection
in high-dimensional datasets such as KDD Cup 99.


10. Future Improvements
-----------------------
- Compare performance with a standard Autoencoder baseline
- Hyperparameter tuning (latent dimension, learning rate)
- β-VAE implementation
- Latent-space based anomaly scoring
- Threshold-based classification and confusion matrix analysis
