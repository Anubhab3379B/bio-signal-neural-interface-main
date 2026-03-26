# Bio-Signal Interface System - Quick Start Guide

This guide will help you get started with the Bio-Signal & Human Interface System.

## Installation

1. **Navigate to the project directory:**
```bash
cd C:\Users\cyber19\.gemini\antigravity\scratch\bio_signal_interface
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**
```bash
# On Windows
venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Demo

Once installed, run the full pipeline demonstration:

```bash
python examples\full_pipeline_demo.py
```

This will demonstrate all five core objectives:
- **O4**: Bio-signal preprocessing
- **O5**: Intent inference
- **O6**: Adaptive control logic
- **O7**: Sensory feedback encoding
- **Neuro-Adaptive AI Control**

## Testing Individual Modules

Each module can be tested independently:

### 1. Bio-Signal Preprocessing
```bash
python src\bio_signal_preprocessing\preprocessing.py
```

### 2. Neuro-Adaptive Control
```bash
python src\neuro_adaptive_control\adaptive_controller.py
```

### 3. Sensory Feedback Encoding
```bash
python src\sensory_feedback\encoder.py
```

### 4. Intent Inference
```bash
python src\intent_inference\classifier.py
```

### 5. Adaptive Control Logic
```bash
python src\adaptive_control\controller.py
```

### 6. Utilities
```bash
python src\utils\helpers.py
```

## Project Structure

```
bio_signal_interface/
├── README.md                          # Main documentation
├── QUICKSTART.md                      # This file
├── requirements.txt                   # Dependencies
├── config/                            # Configuration files
│   └── default_config.yaml           # Default parameters
├── src/                               # Source code
│   ├── bio_signal_preprocessing/      # Signal preprocessing
│   ├── neuro_adaptive_control/        # Adaptive AI control
│   ├── sensory_feedback/              # Sensory encoding
│   ├── intent_inference/              # Intent classification
│   ├── adaptive_control/              # Control logic
│   └── utils/                         # Helper functions
├── examples/                          # Demo scripts
│   └── full_pipeline_demo.py         # Complete pipeline demo
├── data/                              # Data directory (for datasets)
└── notebooks/                         # Jupyter notebooks (optional)
```

## Understanding the Code

Every Python file in this project includes **detailed line-by-line explanations** in simple language. The comments explain:
- What each line does
- Why it's needed
- How it contributes to the overall goal

This makes the code accessible for learning and understanding the concepts.

## Configuration

The system can be configured by editing `config/default_config.yaml`. This file contains all parameters for:
- Signal preprocessing
- Neural network architectures
- Training parameters
- Control system settings

## Important Notes

⚠️ **This is a SIMULATION for research and educational purposes.**

- All components are conceptual and designed for academic exploration
- No claims of real-world deployment
- No medical applicability
- Hardware implementation is explicitly out of scope

## Next Steps

1. Run the full pipeline demo
2. Explore individual modules
3. Read the detailed code comments
4. Experiment with different configurations
5. Modify parameters to see their effects

## Support

For questions or issues, refer to the detailed comments in the code or the main README.md file.
