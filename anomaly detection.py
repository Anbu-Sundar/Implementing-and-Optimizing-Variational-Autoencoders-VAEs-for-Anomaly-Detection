"""
Variational Autoencoder (VAE) for Anomaly Detection
Implementation with Beta-VAE and ELBO loss function
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import json


class VAEEncoder(nn.Module):
    """Encoder network that maps input to latent distribution parameters"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(VAEEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Separate layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network that reconstructs input from latent representation"""
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super(VAEDecoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build decoder layers (reverse of encoder)
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class BetaVAE(nn.Module):
    """
    Beta-Variational Autoencoder for anomaly detection
    
    The Beta-VAE extends the standard VAE by introducing a hyperparameter beta
    that weights the KL divergence term in the ELBO loss function.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(BetaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, input_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (returns mean)"""
        mu, _ = self.encoder(x)
        return mu


def elbo_loss(x: torch.Tensor, x_recon: torch.Tensor, 
              mu: torch.Tensor, logvar: torch.Tensor, 
              beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the Evidence Lower Bound (ELBO) loss
    
    ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
    
    Where:
    - Reconstruction term: E[log p(x|z)] approximated by MSE
    - KL divergence term: KL(q(z|x) || p(z)) for Gaussian distributions
    
    Args:
        x: Original input
        x_recon: Reconstructed input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE parameter)
    
    Returns:
        total_loss, reconstruction_loss, kl_divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Derived from KL(N(mu, sigma) || N(0, 1))
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total ELBO loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


class StandardAutoencoder(nn.Module):
    """Standard Autoencoder for baseline comparison"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(StandardAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def train_vae(model: BetaVAE, train_loader: DataLoader, 
              epochs: int, beta: float, learning_rate: float = 1e-3,
              device: str = 'cpu') -> Dict[str, List[float]]:
    """Train the Beta-VAE model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_div': []
    }
    
    model.train()
    for epoch in range(epochs):
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_div_epoch = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar = model(data)
            
            # Calculate ELBO loss
            loss, recon_loss, kl_div = elbo_loss(data, x_recon, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_div_epoch += kl_div.item()
        
        # Average losses
        n_batches = len(train_loader)
        history['total_loss'].append(total_loss_epoch / n_batches)
        history['recon_loss'].append(recon_loss_epoch / n_batches)
        history['kl_div'].append(kl_div_epoch / n_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Total Loss: {history['total_loss'][-1]:.4f}, "
                  f"Recon Loss: {history['recon_loss'][-1]:.4f}, "
                  f"KL Div: {history['kl_div'][-1]:.4f}")
    
    return history


def train_baseline(model: StandardAutoencoder, train_loader: DataLoader,
                   epochs: int, learning_rate: float = 1e-3,
                   device: str = 'cpu') -> Dict[str, List[float]]:
    """Train the baseline standard autoencoder"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {'loss': []}
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            x_recon = model(data)
            loss = criterion(x_recon, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        history['loss'].append(total_loss / len(train_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {history['loss'][-1]:.4f}")
    
    return history


def compute_anomaly_scores(model: nn.Module, data: torch.Tensor, 
                          is_vae: bool = True, device: str = 'cpu') -> np.ndarray:
    """
    Compute anomaly scores using reconstruction error and KL divergence
    
    For VAE: score = reconstruction_error + KL_divergence
    For AE: score = reconstruction_error
    """
    model = model.to(device)
    model.eval()
    
    scores = []
    
    with torch.no_grad():
        data = data.to(device)
        
        if is_vae:
            x_recon, mu, logvar = model(data)
            
            # Reconstruction error (per sample)
            recon_error = torch.sum((data - x_recon) ** 2, dim=1)
            
            # KL divergence (per sample)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Combined anomaly score
            score = recon_error + kl_div
        else:
            x_recon = model(data)
            score = torch.sum((data - x_recon) ** 2, dim=1)
        
        scores = score.cpu().numpy()
    
    return scores


def evaluate_anomaly_detection(normal_scores: np.ndarray, 
                               anomaly_scores: np.ndarray) -> Dict[str, float]:
    """Evaluate anomaly detection performance using AUC-ROC"""
    
    # Combine scores and create labels
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(len(normal_scores)), 
                            np.ones(len(anomaly_scores))])
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(labels, all_scores)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, all_scores)
    
    return {
        'auc_roc': auc_roc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def generate_synthetic_dataset(n_samples: int = 1000, n_features: int = 20,
                               anomaly_ratio: float = 0.1, 
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic high-dimensional dataset with complex structure
    
    Normal data: Generated from a mixture of Gaussians with correlations
    Anomalies: Generated with different distributions and injected noise
    """
    np.random.seed(random_state)
    
    # Calculate sizes
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal
    
    # Generate normal data with complex structure
    # Create correlation structure
    A = np.random.randn(n_features, n_features // 2)
    cov_matrix = A @ A.T + np.eye(n_features) * 0.1
    
    # Mixture of two Gaussians
    split = n_normal // 2
    normal_data_1 = np.random.multivariate_normal(
        mean=np.zeros(n_features), 
        cov=cov_matrix, 
        size=split
    )
    normal_data_2 = np.random.multivariate_normal(
        mean=np.ones(n_features) * 0.5, 
        cov=cov_matrix * 0.8, 
        size=n_normal - split
    )
    normal_data = np.vstack([normal_data_1, normal_data_2])
    
    # Add non-linear transformations
    normal_data = normal_data + 0.1 * np.sin(normal_data)
    
    # Generate anomalies with different characteristics
    anomaly_data = []
    
    # Type 1: Extreme values
    anomaly_1 = normal_data[:n_anomaly // 3] + np.random.randn(n_anomaly // 3, n_features) * 3
    
    # Type 2: Different distribution (uniform)
    anomaly_2 = np.random.uniform(-3, 3, size=(n_anomaly // 3, n_features))
    
    # Type 3: Clustered anomalies
    anomaly_3 = np.random.multivariate_normal(
        mean=np.ones(n_features) * 2, 
        cov=np.eye(n_features) * 0.5, 
        size=n_anomaly - 2 * (n_anomaly // 3)
    )
    
    anomaly_data = np.vstack([anomaly_1, anomaly_2, anomaly_3])
    
    # Split normal data into train and test
    train_size = int(n_normal * 0.8)
    train_data = normal_data[:train_size]
    test_normal = normal_data[train_size:]
    
    return train_data, test_normal, anomaly_data, cov_matrix


def plot_training_curves(vae_history: Dict, ae_history: Dict, save_path: str = None):
    """Plot training curves for VAE and baseline AE"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # VAE losses
    axes[0].plot(vae_history['total_loss'], label='Total Loss', linewidth=2)
    axes[0].plot(vae_history['recon_loss'], label='Reconstruction Loss', linewidth=2)
    axes[0].plot(vae_history['kl_div'], label='KL Divergence', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Beta-VAE Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Comparison
    axes[1].plot(vae_history['total_loss'], label='VAE Total Loss', linewidth=2)
    axes[1].plot(ae_history['loss'], label='Baseline AE Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('VAE vs Baseline AE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(vae_results: Dict, ae_results: Dict, save_path: str = None):
    """Plot ROC curves for VAE and baseline AE"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(vae_results['fpr'], vae_results['tpr'], 
            label=f"Beta-VAE (AUC = {vae_results['auc_roc']:.4f})", 
            linewidth=2)
    ax.plot(ae_results['fpr'], ae_results['tpr'], 
            label=f"Baseline AE (AUC = {ae_results['auc_roc']:.4f})", 
            linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Anomaly Detection Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_space(model: BetaVAE, normal_data: torch.Tensor, 
                     anomaly_data: torch.Tensor, device: str = 'cpu',
                     save_path: str = None):
    """Visualize the latent space (first 2 dimensions)"""
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        normal_data = normal_data.to(device)
        anomaly_data = anomaly_data.to(device)
        
        normal_latent = model.encode(normal_data).cpu().numpy()
        anomaly_latent = model.encode(anomaly_data).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(normal_latent[:, 0], normal_latent[:, 1], 
              alpha=0.6, s=30, label='Normal', c='blue')
    ax.scatter(anomaly_latent[:, 0], anomaly_latent[:, 1], 
              alpha=0.6, s=30, label='Anomaly', c='red')
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Latent Space Visualization (First 2 Dimensions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def hyperparameter_tuning(train_data: np.ndarray, test_normal: np.ndarray,
                         test_anomaly: np.ndarray, beta_values: List[float],
                         epochs: int = 50, device: str = 'cpu') -> Dict:
    """
    Perform hyperparameter tuning for beta parameter
    """
    
    results = {}
    
    for beta in beta_values:
        print(f"\nTraining with beta = {beta}")
        
        # Prepare data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_normal_scaled = scaler.transform(test_normal)
        test_anomaly_scaled = scaler.transform(test_anomaly)
        
        train_tensor = torch.FloatTensor(train_scaled)
        test_normal_tensor = torch.FloatTensor(test_normal_scaled)
        test_anomaly_tensor = torch.FloatTensor(test_anomaly_scaled)
        
        train_loader = DataLoader(TensorDataset(train_tensor), 
                                 batch_size=32, shuffle=True)
        
        # Train model
        model = BetaVAE(
            input_dim=train_data.shape[1],
            hidden_dims=[128, 64],
            latent_dim=10
        )
        
        history = train_vae(model, train_loader, epochs, beta, device=device)
        
        # Evaluate
        normal_scores = compute_anomaly_scores(model, test_normal_tensor, 
                                              is_vae=True, device=device)
        anomaly_scores = compute_anomaly_scores(model, test_anomaly_tensor, 
                                               is_vae=True, device=device)
        
        eval_results = evaluate_anomaly_detection(normal_scores, anomaly_scores)
        
        results[beta] = {
            'auc_roc': eval_results['auc_roc'],
            'history': history
        }
        
        print(f"Beta = {beta}, AUC-ROC = {eval_results['auc_roc']:.4f}")
    
    return results


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("Variational Autoencoder (VAE) for Anomaly Detection")
    print("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    train_data, test_normal, test_anomaly, cov_matrix = generate_synthetic_dataset(
        n_samples=2000,
        n_features=20,
        anomaly_ratio=0.1,
        random_state=42
    )
    
    print(f"   Training samples: {train_data.shape[0]}")
    print(f"   Test normal samples: {test_normal.shape[0]}")
    print(f"   Test anomaly samples: {test_anomaly.shape[0]}")
    print(f"   Number of features: {train_data.shape[1]}")
    
    # Standardize data
    print("\n2. Standardizing data...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_normal_scaled = scaler.transform(test_normal)
    test_anomaly_scaled = scaler.transform(test_anomaly)
    
    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(train_scaled)
    test_normal_tensor = torch.FloatTensor(test_normal_scaled)
    test_anomaly_tensor = torch.FloatTensor(test_anomaly_scaled)
    
    train_loader = DataLoader(TensorDataset(train_tensor), 
                             batch_size=32, shuffle=True)
    
    # Hyperparameter tuning for beta
    print("\n3. Hyperparameter tuning for beta parameter...")
    beta_values = [0.5, 1.0, 2.0, 4.0]
    tuning_results = hyperparameter_tuning(
        train_data, test_normal, test_anomaly,
        beta_values, epochs=50, device=device
    )
    
    # Select best beta
    best_beta = max(tuning_results.keys(), 
                   key=lambda b: tuning_results[b]['auc_roc'])
    print(f"\n   Best beta: {best_beta}")
    print(f"   Best AUC-ROC: {tuning_results[best_beta]['auc_roc']:.4f}")
    
    # Train final VAE with best beta
    print(f"\n4. Training Beta-VAE with beta = {best_beta}...")
    vae_model = BetaVAE(
        input_dim=train_data.shape[1],
        hidden_dims=[128, 64],
        latent_dim=10
    )
    vae_history = train_vae(vae_model, train_loader, epochs=100, 
                           beta=best_beta, device=device)
    
    # Train baseline autoencoder
    print("\n5. Training baseline Autoencoder...")
    ae_model = StandardAutoencoder(
        input_dim=train_data.shape[1],
        hidden_dims=[128, 64],
        latent_dim=10
    )
    ae_history = train_baseline(ae_model, train_loader, epochs=100, device=device)
    
    # Compute anomaly scores
    print("\n6. Computing anomaly scores...")
    vae_normal_scores = compute_anomaly_scores(vae_model, test_normal_tensor, 
                                               is_vae=True, device=device)
    vae_anomaly_scores = compute_anomaly_scores(vae_model, test_anomaly_tensor, 
                                                is_vae=True, device=device)
    
    ae_normal_scores = compute_anomaly_scores(ae_model, test_normal_tensor, 
                                              is_vae=False, device=device)
    ae_anomaly_scores = compute_anomaly_scores(ae_model, test_anomaly_tensor, 
                                               is_vae=False, device=device)
    
    # Evaluate performance
    print("\n7. Evaluating anomaly detection performance...")
    vae_results = evaluate_anomaly_detection(vae_normal_scores, vae_anomaly_scores)
    ae_results = evaluate_anomaly_detection(ae_normal_scores, ae_anomaly_scores)
    
    print(f"\n   Beta-VAE AUC-ROC: {vae_results['auc_roc']:.4f}")
    print(f"   Baseline AE AUC-ROC: {ae_results['auc_roc']:.4f}")
    print(f"   Improvement: {(vae_results['auc_roc'] - ae_results['auc_roc']):.4f}")
    
    # Generate visualizations
    print("\n8. Generating visualizations...")
    
    plot_training_curves(vae_history, ae_history, 
                        save_path='/home/claude/training_curves.png')
    print("   Saved: training_curves.png")
    
    plot_roc_curves(vae_results, ae_results, 
                   save_path='/home/claude/roc_curves.png')
    print("   Saved: roc_curves.png")
    
    plot_latent_space(vae_model, test_normal_tensor, test_anomaly_tensor,
                     device=device, save_path='/home/claude/latent_space.png')
    print("   Saved: latent_space.png")
    
    # Plot beta comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    betas = sorted(tuning_results.keys())
    aucs = [tuning_results[b]['auc_roc'] for b in betas]
    ax.plot(betas, aucs, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Beta Parameter')
    ax.set_ylabel('AUC-ROC Score')
    ax.set_title('Effect of Beta Parameter on Anomaly Detection Performance')
    ax.grid(True, alpha=0.3)
    plt.savefig('/home/claude/beta_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: beta_comparison.png")
    
    # Save results
    results_summary = {
        'best_beta': best_beta,
        'vae_auc_roc': vae_results['auc_roc'],
        'ae_auc_roc': ae_results['auc_roc'],
        'improvement': vae_results['auc_roc'] - ae_results['auc_roc'],
        'beta_tuning': {str(k): v['auc_roc'] for k, v in tuning_results.items()}
    }
    
    with open('/home/claude/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("   Saved: results_summary.json")
    
    # Save models
    torch.save(vae_model.state_dict(), '/home/claude/vae_model.pth')
    torch.save(ae_model.state_dict(), '/home/claude/ae_model.pth')
    print("   Saved: vae_model.pth, ae_model.pth")
    
    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
