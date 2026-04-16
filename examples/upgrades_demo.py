"""
Upgrade Features Demonstration

This script demonstrates all the new upgrade features added to the system:
1. Advanced Visualization
2. Uncertainty Quantification
3. Data Augmentation
4. Attention-Based Models
5. Model Checkpointing

Run this to see all upgrades in action!
"""

# Import standard libraries
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Import original modules
from src.bio_signal_preprocessing import generate_simulated_emg, BioSignal
from src.intent_inference import generate_simulated_intent_data

# Import UPGRADE modules
from src.utils import (
    # Advanced visualization
    plot_signal_comparison,
    plot_training_history,
    plot_uncertainty_distribution,
    plot_feature_importance,
    # Uncertainty quantification
    MCDropoutModel,
    reject_uncertain_predictions,
    # Data augmentation
    SignalAugmenter,
    create_augmented_dataset,
    # Model checkpointing
    ModelCheckpoint
)

# Import attention classifier
from src.intent_inference import AttentionIntentClassifier


def demo_advanced_visualization():
    """Demonstrate advanced visualization capabilities."""
    print("\n" + "=" * 70)
    print("UPGRADE 1: Advanced Visualization")
    print("=" * 70)
    
    # Generate two different signals
    np.random.seed(42)
    signal1 = generate_simulated_emg(duration=2.0, sampling_rate=1000.0, noise_level=0.2)
    signal2 = generate_simulated_emg(duration=2.0, sampling_rate=1000.0, noise_level=0.1)
    
    print("\n1.1 Signal Comparison with Spectrograms")
    print("-" * 70)
    plot_signal_comparison(
        [signal1, signal2],
        ['High Noise EMG', 'Low Noise EMG'],
        sampling_rate=1000.0
    )
    
    print("\n1.2 Training History Visualization")
    print("-" * 70)
    # Simulate training history
    history = {
        'loss': [0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.22, 0.20, 0.19, 0.18],
        'accuracy': [0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91],
        'val_loss': [0.95, 0.75, 0.55, 0.45, 0.38, 0.35, 0.33, 0.32, 0.31, 0.30]
    }
    plot_training_history(history)
    
    print("\n1.3 Feature Importance")
    print("-" * 70)
    features = {
        'rms': 0.89,
        'zero_crossing_rate': 0.76,
        'dominant_frequency': 0.68,
        'mean_frequency': 0.54,
        'spectral_entropy': 0.42,
        'variance': 0.35,
        'kurtosis': 0.28,
        'skewness': 0.15
    }
    plot_feature_importance(features, top_n=8)
    
    print("\n✓ Advanced visualization complete!")


def demo_uncertainty_quantification():
    """Demonstrate uncertainty quantification."""
    print("\n" + "=" * 70)
    print("UPGRADE 2: Uncertainty Quantification")
    print("=" * 70)
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n2.1 Creating MC Dropout Model")
    print("-" * 70)
    model = MCDropoutModel(input_size=15, hidden_size=64, output_size=5, dropout_rate=0.2)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate training data
    X_train = torch.randn(200, 15)
    y_train = torch.randint(0, 5, (200,))
    
    print("\n2.2 Training with uncertainty estimation")
    print("-" * 70)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    print("\n2.3 Predictions with uncertainty")
    print("-" * 70)
    X_test = torch.randn(50, 15)
    y_test = torch.randint(0, 5, (50,))
    
    # Get predictions with uncertainty
    mean_pred, uncertainty = model.predict_with_uncertainty(X_test, n_samples=30)
    class_pred = torch.argmax(mean_pred, dim=1).numpy()
    avg_uncertainty = uncertainty.mean(dim=1).numpy()
    
    print(f"Average uncertainty: {avg_uncertainty.mean():.4f}")
    print(f"Uncertainty range: [{avg_uncertainty.min():.4f}, {avg_uncertainty.max():.4f}]")
    
    # Visualize uncertainty distribution
    plot_uncertainty_distribution(class_pred, avg_uncertainty, y_test.numpy())
    
    print("\n2.4 Rejecting uncertain predictions")
    print("-" * 70)
    threshold = np.percentile(avg_uncertainty, 70)  # Reject top 30%
    accepted_pred, accepted_idx = reject_uncertain_predictions(
        class_pred, avg_uncertainty, threshold=threshold
    )
    
    print(f"Accepted {len(accepted_pred)} out of {len(class_pred)} predictions")
    
    print("\n✓ Uncertainty quantification complete!")


def demo_data_augmentation():
    """Demonstrate data augmentation."""
    print("\n" + "=" * 70)
    print("UPGRADE 3: Data Augmentation")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n3.1 Creating Signal Augmenter")
    print("-" * 70)
    augmenter = SignalAugmenter(sampling_rate=1000.0)
    
    # Generate test signal
    signal_original = generate_simulated_emg(duration=2.0, sampling_rate=1000.0, noise_level=0.1)
    
    print("\n3.2 Applying different augmentations")
    print("-" * 70)
    
    # Apply various augmentations
    signal_noisy = augmenter.add_noise(signal_original, noise_level=0.2)
    signal_scaled = augmenter.scale_amplitude(signal_original, scale_range=(0.7, 1.3))
    signal_wandered = augmenter.add_baseline_wander(signal_original, wander_amplitude=0.15)
    signal_spiked = augmenter.spike_artifacts(signal_original, n_spikes=5)
    
    # Compare original and augmented
    plot_signal_comparison(
        [signal_original, signal_noisy, signal_scaled, signal_wandered, signal_spiked],
        ['Original', 'Noisy', 'Scaled', 'Baseline Wander', 'Spike Artifacts'],
        sampling_rate=1000.0,
        titles=['Original Signal', 'Added Noise', 'Amplitude Scaled', 
                'Baseline Drift', 'Movement Artifacts']
    )
    
    print("\n3.3 Creating augmented dataset")
    print("-" * 70)
    # Create small dataset
    n_samples = 20
    signals = np.array([generate_simulated_emg(duration=1.0, sampling_rate=1000.0, noise_level=0.1) 
                       for _ in range(n_samples)])
    labels = np.random.randint(0, 3, n_samples)
    
    # Augment it
    aug_signals, aug_labels = create_augmented_dataset(
        signals, labels, augmentation_factor=5, sampling_rate=1000.0
    )
    
    print(f"\nOriginal dataset size: {len(signals)}")
    print(f"Augmented dataset size: {len(aug_signals)}")
    print(f"Expansion: {len(aug_signals) / len(signals):.1f}x")
    
    print("\n✓ Data augmentation complete!")


def demo_attention_model():
    """Demonstrate attention-based classifier."""
    print("\n" + "=" * 70)
    print("UPGRADE 4: Attention-Based Models")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print("\n4.1 Creating Attention-Based Classifier")
    print("-" * 70)
    model = AttentionIntentClassifier(
        input_size=15,
        hidden_size=64,
        num_classes=5,
        num_heads=4,
        num_layers=2
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Architecture: 2 attention layers, 4 heads each")
    
    print("\n4.2 Generating training data")
    print("-" * 70)
    X_train, y_train = generate_simulated_intent_data(
        n_samples=100, sequence_length=100, n_features=15, n_classes=5
    )
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    
    print(f"Training data: {X_train.shape}")
    
    print("\n4.3 Training attention model")
    print("-" * 70)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(15):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_train).float().mean()
        
        history['loss'].append(loss.item())
        history['accuracy'].append(acc.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/15, Loss: {loss.item():.4f}, Accuracy: {acc.item():.2%}")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n4.4 Extracting attention weights")
    print("-" * 70)
    model.eval()
    with torch.no_grad():
        _ = model(X_train[:1])  # Forward pass on one sample
        attention_maps = model.get_attention_maps()
    
    print(f"Number of attention layers: {len(attention_maps)}")
    print(f"Attention map shape: {attention_maps[0].shape}")
    print("Note: Attention weights show which time steps the model focuses on")
    
    print("\n✓ Attention model complete!")


def demo_model_checkpointing():
    """Demonstrate model checkpointing."""
    print("\n" + "=" * 70)
    print("UPGRADE 5: Model Checkpointing")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print("\n5.1 Creating checkpoint manager")
    print("-" * 70)
    checkpoint_dir = Path('./demo_checkpoints')
    checkpoint_mgr = ModelCheckpoint(
        checkpoint_dir=str(checkpoint_dir),
        model_name='demo_model'
    )
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    print("\n5.2 Creating and training a model")
    print("-" * 70)
    model = nn.Sequential(
        nn.Linear(15, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 5)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate training
    X = torch.randn(100, 15)
    y = torch.randint(0, 5, (100,))
    criterion = nn.CrossEntropyLoss()
    
    print("Training and saving checkpoints...")
    for epoch in range(5):
        # Training step
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            acc = (torch.argmax(outputs, dim=1) == y).float().mean()
        
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}, Acc: {acc.item():.2%}")
        
        # Save checkpoint
        checkpoint_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={'loss': loss.item(), 'accuracy': acc.item()},
            metadata={'learning_rate': 0.001}
        )
        
        # Try to save as best model
        is_best = checkpoint_mgr.save_best_model(
            model=model,
            metric_value=acc.item(),
            metric_name='accuracy',
            higher_is_better=True,
            optimizer=optimizer,
            epoch=epoch + 1
        )
    
    print("\n5.3 Listing checkpoints")
    print("-" * 70)
    all_checkpoints = checkpoint_mgr.list_checkpoints()
    print(f"Total checkpoints saved: {len(all_checkpoints)}")
    
    best_path = checkpoint_mgr.get_best_checkpoint()
    print(f"Best model saved at: {best_path}")
    
    print("\n✓ Model checkpointing complete!")


def main():
    """Run all upgrade demonstrations."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "BIO-SIGNAL INTERFACE UPGRADES DEMO" + " " * 19 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    print("\nThis demo showcases 5 major upgrades to the bio-signal interface system:")
    print("  1. Advanced Visualization - Enhanced plots and analysis")
    print("  2. Uncertainty Quantification - Know when model is unsure")
    print("  3. Data Augmentation - Train better models with less data")
    print("  4. Attention-Based Models - State-of-the-art architecture")
    print("  5. Model Checkpointing - Track and manage experiments")
    
    print("\n" + "=" * 70)
    print("ALL UPGRADES MAINTAIN BACKWARD COMPATIBILITY")
    print("Original code continues to work exactly as before!")
    print("=" * 70)
    
    input("\nPress Enter to start demo...")
    
    # Run all demos
    try:
        demo_advanced_visualization()
        input("\nPress Enter to continue to next upgrade...")
        
        demo_uncertainty_quantification()
        input("\nPress Enter to continue to next upgrade...")
        
        demo_data_augmentation()
        input("\nPress Enter to continue to next upgrade...")
        
        demo_attention_model()
        input("\nPress Enter to continue to next upgrade...")
        
        demo_model_checkpointing()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return
    
    # Final summary
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 25 + "DEMO COMPLETE!" + " " * 29 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    print("\n📊 Summary of Upgrades:")
    print("  ✓ Advanced Visualization - 6 new plotting functions")
    print("  ✓ Uncertainty Quantification - MC Dropout + Ensembles")
    print("  ✓ Data Augmentation - 8 augmentation techniques")
    print("  ✓ Attention Models - Multi-head self-attention")
    print("  ✓ Model Checkpointing - Automatic versioning & best model tracking")
    
    print("\n🎯 Key Benefits:")
    print("  • Better model performance and accuracy")
    print("  • Improved safety with uncertainty estimates")
    print("  • Train on less data with augmentation")
    print("  • Interpretable predictions with attention")
    print("  • Professional experiment management")
    
    print("\n📁 New Files Created:")
    print("  • src/utils/advanced_viz.py")
    print("  • src/utils/uncertainty.py")
    print("  • src/utils/augmentation.py")
    print("  • src/utils/checkpoint.py")
    print("  • src/intent_inference/attention_classifier.py")
    
    print("\n💡 Next Steps:")
    print("  1. Explore each module's documentation")
    print("  2. Integrate upgrades into your workflows")
    print("  3. Run individual module demos for deep dives")
    print("  4. Check UPGRADE_ROADMAP.md for more enhancements")
    
    print("\n" + "=" * 70)
    print("Thank you for exploring the upgrades!")
    print("=" * 70)


if __name__ == "__main__":
    main()
