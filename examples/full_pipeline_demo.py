"""
Full Pipeline Demo

This script demonstrates the complete bio-signal interface system including:
1. Bio-signal preprocessing
2. Neuro-adaptive control
3. Sensory feedback encoding
4. Intent inference
5. Adaptive control logic

This brings together all components in a realistic workflow.
"""

# Import standard libraries
import sys  # For system operations
import os  # For path operations
import numpy as np  # Numerical operations
import torch  # PyTorch

# Add src directory to path so we can import our modules
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to reach the project root, then into src
src_path = os.path.join(os.path.dirname(script_dir), 'src')
# Add to Python path
sys.path.insert(0, src_path)

# Import our custom modules
# These are the modules we created for the project
from bio_signal_preprocessing import (
    generate_simulated_emg,  # Generate EMG signals
    create_preprocessing_pipeline,  # Pipeline for cleaning signals
    extract_features  # Extract features from signals
)

from neuro_adaptive_control import (
    NeuroAdaptiveController  # AI controller that learns and adapts
)

from sensory_feedback import (
    SensoryEncoder  # Encodes physical sensations
)

from intent_inference import (
    IntentInferenceSystem,  # Infers user's intentions
    generate_simulated_intent_data  # Generate test data
)

from adaptive_control import (
    AdaptiveControlSystem,  # Adaptive control system
    SimpleSystem  # Simple system to control
)


def main():
    """
    Main function demonstrating the full pipeline.
    """
    # Print header
    print("=" * 70)
    print(" " * 15 + "BIO-SIGNAL INTERFACE SYSTEM")
    print(" " * 20 + "Full Pipeline Demo")
    print("=" * 70)
    print("\nThis demo showcases all five core objectives:")
    print("  O4: Bio-signal preprocessing")
    print("  O5: Intent inference")
    print("  O6: Adaptive control logic")
    print("  O7: Sensory feedback encoding")
    print("  Neuro-Adaptive AI Control")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    # This ensures we get the same "random" results each time we run
    np.random.seed(42)  # NumPy random seed
    torch.manual_seed(42)  # PyTorch random seed
    
    # ========================================================================
    # STEP 1: BIO-SIGNAL PREPROCESSING (O4)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Bio-Signal Preprocessing (O4)")
    print("=" * 70)
    
    # Generate a simulated EMG signal (muscle activity)
    print("\n1.1 Generating simulated EMG signal...")
    # EMG measures electrical activity from muscles
    # We create a 5-second signal sampled at 1000 Hz
    emg_signal = generate_simulated_emg(duration=5.0, sampling_rate=1000.0, noise_level=0.2)
    print(f"    ✓ Generated {emg_signal.get_duration():.1f}s EMG signal")
    print(f"    ✓ Type: {emg_signal.signal_type}")
    print(f"    ✓ Samples: {len(emg_signal.data)}")
    
    # Create a preprocessing pipeline
    print("\n1.2 Creating preprocessing pipeline...")
    # The pipeline will:
    # - Apply bandpass filter (keep frequencies 10-200 Hz)
    # - Remove 50 Hz power line interference
    # - Denoise using wavelets
    # - Normalize to standard scale
    preprocessing_pipeline = create_preprocessing_pipeline(
        lowcut=10.0,  # Remove frequencies below 10 Hz
        highcut=200.0,  # Remove frequencies above 200 Hz
        notch_freq=50.0,  # Remove 50 Hz interference
        normalize_method='zscore'  # Normalize to mean=0, std=1
    )
    print("    ✓ Pipeline created with:")
    print("      - Bandpass filter (10-200 Hz)")
    print("      - Notch filter (50 Hz)")
    print("      - Wavelet denoising")
    print("      - Z-score normalization")
    
    # Apply preprocessing
    print("\n1.3 Applying preprocessing...")
    # This cleans the signal and makes it ready for analysis
    emg_preprocessed = preprocessing_pipeline(emg_signal)
    print(f"    ✓ Signal preprocessed")
    print(f"    ✓ Before: mean={np.mean(emg_signal.data):.3f}, std={np.std(emg_signal.data):.3f}")
    print(f"    ✓ After: mean={np.mean(emg_preprocessed.data):.3f}, std={np.std(emg_preprocessed.data):.3f}")
    
    # Extract features
    print("\n1.4 Extracting features...")
    # Features are numerical characteristics that summarize the signal
    # These will be used as input to AI models
    features = extract_features(emg_preprocessed)
    print(f"    ✓ Extracted {len(features)} features:")
    print(f"      - RMS: {features['rms']:.4f}")
    print(f"      - Dominant frequency: {features['dominant_frequency']:.2f} Hz")
    print(f"      - Zero crossing rate: {features['zero_crossing_rate']:.4f}")
    
    # ========================================================================
    # STEP 2: INTENT INFERENCE (O5)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Intent Inference (O5)")
    print("=" * 70)
    
    # Generate training data for intent classification
    print("\n2.1 Generating intent training data...")
    # We simulate data for 5 different intents (e.g., different movements)
    # Each sample is a sequence of 100 time steps with 15 features
    train_data, train_labels = generate_simulated_intent_data(
        n_samples=200,  # 200 training examples
        sequence_length=100,  # Each sequence has 100 time steps
        n_features=15,  # Each time step has 15 features
        n_classes=5  # 5 different intent classes
    )
    print(f"    ✓ Generated {len(train_data)} training samples")
    print(f"    ✓ Data shape: {train_data.shape}")
    print(f"    ✓ Classes: {np.unique(train_labels)}")
    
    # Create intent inference system
    print("\n2.2 Creating LSTM-based intent inference system...")
    # LSTM (Long Short-Term Memory) networks are good at learning
    # patterns in sequential data like bio-signals
    intent_system = IntentInferenceSystem(
        model_type='lstm',  # Use LSTM architecture
        input_size=15,  # 15 features per time step
        num_classes=5,  # 5 intent classes
        sequence_length=100  # Sequences are 100 steps long
    )
    print("    ✓ LSTM model created")
    
    # Train the system
    print("\n2.3 Training intent classifier...")
    # Train for 10 epochs (passes through the data)
    for epoch in range(10):
        loss = intent_system.train_epoch(train_data, train_labels, batch_size=32)
        # Print progress every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"    ✓ Epoch {epoch+1}/10 - Loss: {loss:.4f}")
    
    # Test the system
    print("\n2.4 Testing intent inference...")
    # Generate some test data
    test_data, test_labels = generate_simulated_intent_data(
        n_samples=50,  # 50 test examples
        sequence_length=100,
        n_features=15,
        n_classes=5
    )
    # Evaluate performance
    metrics = intent_system.evaluate(test_data, test_labels)
    print(f"    ✓ Test accuracy: {metrics['accuracy']:.2%}")
    
    # Make a prediction on a single sample
    sample = test_data[0:1]  # Take first test sample
    prediction = intent_system.predict(sample)
    probabilities = intent_system.predict_proba(sample)
    print(f"    ✓ Sample prediction: class {prediction[0]} (true: {test_labels[0]})")
    print(f"    ✓ Confidence: {probabilities[0][prediction[0]]:.2%}")
    
    # ========================================================================
    # STEP 3: NEURO-ADAPTIVE AI CONTROL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Neuro-Adaptive AI Control")
    print("=" * 70)
    
    # Create neuro-adaptive controller
    print("\n3.1 Creating neuro-adaptive controller...")
    # This controller learns from user's patterns over time
    # It adapts to the individual user
    adaptive_controller = NeuroAdaptiveController(
        input_size=15,  # Takes 15 features as input
        output_size=5,  # Outputs 5 control values
        learning_rate=0.001  # Learning rate
    )
    print("    ✓ Controller created")
    print(f"    ✓ Input size: 15 features")
    print(f"    ✓ Output size: 5 control dimensions")
    
    # Perform initial learning
    print("\n3.2 Initial learning phase...")
    # Generate some simulated feature vectors and control targets
    init_features = np.random.randn(50, 15)  # 50 samples of 15 features
    init_targets = np.random.randn(50, 5)  # 50 target control values
    # Learn for 5 epochs
    losses = adaptive_controller.continual_learn(init_features, init_targets, epochs=5)
    print(f"    ✓ Initial learning complete")
    print(f"    ✓ Final loss: {losses[-1]:.4f}")
    
    # Simulate online adaptation
    print("\n3.3 Simulating online adaptation...")
    # In real use, the controller would continuously adapt to new data
    new_features = np.random.randn(10, 15)  # New data from user
    new_targets = np.random.randn(10, 5)  # Corresponding targets
    loss = adaptive_controller.adapt(new_features, new_targets, use_replay=True)
    print(f"    ✓ Adapted to new data")
    print(f"    ✓ Adaptation loss: {loss:.4f}")
    
    # Make a prediction
    print("\n3.4 Making predictions...")
    test_features = np.random.randn(3, 15)  # 3 test samples
    predictions = adaptive_controller.predict(test_features)
    print(f"    ✓ Predictions shape: {predictions.shape}")
    print(f"    ✓ Sample output: {predictions[0]}")
    
    # Show statistics
    stats = adaptive_controller.get_stats()
    print(f"\n3.5 Controller statistics:")
    print(f"    ✓ Adaptation count: {stats['adaptation_count']}")
    print(f"    ✓ Replay buffer size: {stats['replay_buffer_size']}")
    print(f"    ✓ Mean loss: {stats['mean_loss']:.4f}")
    
    # ========================================================================
    # STEP 4: SENSORY FEEDBACK ENCODING (A3, O7)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Sensory Feedback Encoding (A3, O7)")
    print("=" * 70)
    
    # Create sensory encoder
    print("\n4.1 Creating sensory encoder...")
    # This converts physical sensations to neural encodings
    sensory_encoder = SensoryEncoder(
        input_dim=100,  # 100 sensors (e.g., pressure sensors on a hand)
        latent_dim=10  # Compress to 10 dimensions
    )
    print("    ✓ Encoder created")
    print(f"    ✓ Compression: 100 → 10 dimensions")
    
    # Generate training data (simulated sensory data)
    print("\n4.2 Generating sensory training data...")
    # Simulate sensory readings from 200 different situations
    sensory_training_data = np.random.rand(200, 100)
    print(f"    ✓ Generated {len(sensory_training_data)} samples")
    
    # Train the VAE (Variational Autoencoder)
    print("\n4.3 Training sensory encoder...")
    # The VAE learns to compress and reconstruct sensory data
    train_losses = sensory_encoder.train_vae(sensory_training_data, epochs=15, batch_size=32)
    print(f"    ✓ Training complete")
    print(f"    ✓ Final loss: {train_losses[-1]:.2f}")
    
    # Encode pressure signal
    print("\n4.4 Encoding pressure signal...")
    # Simulate pressure readings from 100 sensors
    pressure_map = np.random.rand(100)
    pressure_encoding = sensory_encoder.encode_pressure(pressure_map)
    print(f"    ✓ Pressure encoded")
    print(f"    ✓ Compression ratio: {pressure_encoding['compression_ratio']:.1f}x")
    print(f"    ✓ Spikes generated: {np.sum(pressure_encoding['spike_train']):.0f}")
    
    # Encode texture signal
    print("\n4.5 Encoding texture signal...")
    # Simulate texture readings (e.g., roughness)
    texture_signal = np.random.rand(100)
    texture_encoding = sensory_encoder.encode_texture(texture_signal)
    print(f"    ✓ Texture encoded using {texture_encoding['encoding_type']} coding")
    print(f"    ✓ Spike train shape: {texture_encoding['spike_train'].shape}")
    
    # Encode temperature
    print("\n4.6 Encoding temperature...")
    # Simulate temperature from 3 sensors
    temp_values = np.array([25.0, 30.0, 35.0])
    # Pad to match encoder dimension
    temp_padded = np.pad(temp_values, (0, 100 - len(temp_values)), 'constant')
    temp_encoding = sensory_encoder.encode_temperature(temp_padded)
    print(f"    ✓ Temperature encoded")
    print(f"    ✓ Values: {temp_values} °C")
    print(f"    ✓ Encoding type: {temp_encoding['encoding_type']}")
    
    # ========================================================================
    # STEP 5: ADAPTIVE CONTROL LOGIC (O6)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Adaptive Control Logic (O6)")
    print("=" * 70)
    
    # Create adaptive control system
    print("\n5.1 Creating adaptive control system...")
    # This compares rule-based (PID) vs learning-based (neural) control
    control_system = AdaptiveControlSystem(
        state_dim=4,  # State: [position, velocity, target, error]
        action_dim=1,  # One control output (e.g., motor command)
        use_neural=True  # Enable neural controller
    )
    print("    ✓ Control system created with:")
    print("      - PID controller (rule-based)")
    print("      - Neural controller (learning-based)")
    
    # Create a simple system to control
    print("\n5.2 Creating simulated system...")
    # This represents a simple physical system (e.g., prosthetic joint)
    from adaptive_control import SimpleSystem
    system = SimpleSystem(initial_position=0.0, damping=0.1)
    print("    ✓ System created (1D position control)")
    
    # Test PID controller
    print("\n5.3 Testing PID controller...")
    setpoint = 1.0  # Target position
    system.position = 0.0  # Start at origin
    system.velocity = 0.0
    control_system.pid_controller.reset()
    
    # Simulate for 50 time steps
    dt = 0.1  # Time step
    for i in range(50):
        time = i * dt
        state = system.get_state(setpoint)
        control = control_system.compute_control(state, setpoint, time, mode='pid')
        system.step(control, dt)
    
    pid_stats = control_system.pid_controller.get_stats()
    print(f"    ✓ PID test complete")
    print(f"    ✓ Mean error: {pid_stats['mean_error']:.4f}")
    print(f"    ✓ Final error: {pid_stats['final_error']:.4f}")
    
    # Train neural controller
    print("\n5.4 Training neural controller...")
    # Train by learning from multiple trials
    for trial in range(30):
        # Reset system to random position
        system.position = np.random.uniform(-0.5, 0.5)
        system.velocity = 0.0
        
        # Run short episode
        for step in range(15):
            time = step * dt
            state = system.get_state(setpoint)
            
            # Use PID as teacher initially, then explore
            if trial < 15:
                action = control_system.compute_control(state, setpoint, time, mode='pid')
            else:
                action = control_system.compute_control(state, setpoint, time, mode='neural')
            
            # Take action
            system.step(action, dt)
            next_state = system.get_state(setpoint)
            
            # Reward is negative squared error (we want small error)
            reward = -next_state[3] ** 2
            
            # Learn from this experience
            control_system.learn_from_experience(state, action, reward, next_state)
    
    print(f"    ✓ Neural controller trained on 30 trials")
    
    # Compare both controllers
    print("\n5.5 Comparing PID vs Neural controller...")
    # Generate test scenarios
    test_states = []
    test_setpoints = []
    test_times = []
    
    system.position = 0.0
    system.velocity = 0.0
    
    for i in range(100):
        time = i * dt
        state = system.get_state(setpoint)
        test_states.append(state)
        test_setpoints.append(setpoint)
        test_times.append(time)
    
    # Compare performance
    comparison = control_system.compare_controllers(test_states, test_setpoints, test_times)
    
    print("    ✓ Comparison results:")
    print(f"      PID: mean error = {comparison['pid']['mean_error']:.4f}")
    if 'neural' in comparison:
        print(f"      Neural: mean error = {comparison['neural']['mean_error']:.4f}")
        print(f"      Winner: {comparison['winner']}")
    
    # Monitor stability
    print("\n5.6 Monitoring system stability...")
    stability = control_system.monitor_stability(test_states)
    print(f"    ✓ Stability analysis:")
    print(f"      Stable: {stability['stable']}")
    print(f"      Converging: {stability['converging']}")
    print(f"      Oscillating: {stability['oscillating']}")
    print(f"      Max error: {stability['max_error']:.4f}")
    
    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("\n" + "=" * 70)
    print(" " * 25 + "DEMO COMPLETE")
    print("=" * 70)
    print("\nAll five objectives demonstrated successfully:")
    print("  ✓ O4: Bio-signal preprocessing with filtering and feature extraction")
    print("  ✓ O5: Intent inference using LSTM neural networks")
    print("  ✓ Neuro-Adaptive AI: Online learning and domain adaptation")
    print("  ✓ A3/O7: Sensory encoding with VAE and spike coding")
    print("  ✓ O6: Adaptive control with PID vs Neural comparison")
    print("\n" + "=" * 70)
    print("\nIMPORTANT DISCLAIMER:")
    print("This is a SIMULATION for research and educational purposes.")
    print("All components are conceptual and designed for academic exploration.")
    print("There are NO claims of real-world deployment or medical applicability.")
    print("=" * 70)


# Run the demo when script is executed
if __name__ == "__main__":
    main()
