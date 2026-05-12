import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score


def calibrate_balanced(model, attn, X_v, y_v, device):
    """
    Calibrate decision threshold for balanced accuracy.
    
    Args:
        model: The model to evaluate
        attn: Optional attention module
        X_v: Validation features
        y_v: Validation labels
        device: Device to run on
        
    Returns:
        best_t: Optimal threshold for balanced accuracy
    """
    model.eval()
    if attn:
        attn.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_v, dtype=torch.float32).to(device)
        p = model(attn(x_t) if attn else x_t, mode='fine_tune').cpu().numpy()
    
    best_t, best_bal = 0.5, 0
    for t in np.linspace(0.01, 0.9, 100):
        score = balanced_accuracy_score(y_v, (p >= t))
        if score > best_bal:
            best_bal, best_t = score, t
    return best_t


def train_epoch(model, loader, optimizer, criterion, device, accumulation_steps=1):
    """
    Train model for one epoch with gradient accumulation.
    
    Args:
        model: Model to train
        loader: DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits, y) / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        n_batches += 1
    
    # Final step if needed
    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    return total_loss / n_batches


def evaluate_model(model, loader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        
    Returns:
        probs: Predicted probabilities
        labels: True labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    
    return np.concatenate(all_probs), np.concatenate(all_labels)


# Checkerboard Generator
def generate_checkerboard_data(n_samples, img_size=28, noise_level=1.0):
    images, labels = [], []
    for _ in range(n_samples):
        label = np.random.randint(0, 2)
        noise = np.random.normal(0, noise_level, img_size * img_size)
        if label == 1:
            b_size = np.random.randint(3, 6)
            pad = 8
            large_size = img_size + pad
            base = np.indices((large_size // b_size + 1, large_size // b_size + 1)).sum(axis=0) % 2
            checker = base.repeat(b_size, axis=0).repeat(b_size, axis=1)
            y_off, x_off = np.random.randint(0, pad), np.random.randint(0, pad)
            checker = checker[y_off:y_off+img_size, x_off:x_off+img_size]
            checker = (checker.astype(float) * 2) - 1
            mask = np.random.binomial(1, 0.8, size=(img_size, img_size))
            img = noise + (1.0 * (checker * mask).flatten()) 
        else:
            img = noise
        images.append(img); labels.append(label)
    return np.array(images), np.array(labels)