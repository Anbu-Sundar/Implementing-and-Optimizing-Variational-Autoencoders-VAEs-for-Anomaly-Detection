import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_synthetic_anomaly_data(
    n_normal=10000,
    n_anomaly=1000,
    latent_dim=3,
    feature_dim=30,
    noise_std=0.1,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    # 1. Normal data creation (latent variable model)
    Z_normal = rng.normal(0, 1, size=(n_normal, latent_dim))
    W = rng.uniform(-1, 1, size=(latent_dim, feature_dim))
    noise_normal = rng.normal(0, noise_std, size=(n_normal, feature_dim))
    X_normal = Z_normal @ W + noise_normal

    # 2. Anomaly base creation
    Z_anom = rng.normal(0, 1, size=(n_anomaly, latent_dim))
    noise_anom = rng.normal(0, noise_std, size=(n_anomaly, feature_dim))
    X_anom = Z_anom @ W + noise_anom

    # 3. Inject different anomaly types
    idx1 = slice(0, int(0.3 * n_anomaly))
    X_anom[idx1, 0:5] += 5.0

    idx2 = slice(int(0.3 * n_anomaly), int(0.65 * n_anomaly))
    X_anom[idx2, 10] = rng.uniform(-10, 10, size=(idx2.stop - idx2.start))

    idx3 = slice(int(0.65 * n_anomaly), n_anomaly)
    X_anom[idx3, 20] += rng.normal(0, 10, size=(idx3.stop - idx3.start))

    # 4. Train / test split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_normal)

    X_test = scaler.transform(
        np.vstack([X_normal, X_anom])
    )

    y_test = np.hstack([
        np.zeros(n_normal, dtype=int),
        np.ones(n_anomaly, dtype=int)
    ])

    return X_train, X_test, y_test
