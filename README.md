-# AIReplacementInstructionCPU
AI-JSON replacement of processor instructions
### Advanced Instruction Analyzer & Optimizer

**Program description**  
This program analyzes executable files (EXE), finds processor instructions (AVX2,FMA,F16C,BMI,BMI2) and replaces them with AVX using neural network models. Key Features:
- Dynamic instruction replacement with validation via emulation
- Performance benchmarking before/after replacement
- Generation of optimized binary files
- Neural network training based on user substitution patterns

---

### Startup Requirements
**Python 3.9+** and the following libraries:
``
pefile==2023.2.7
capstone==5.0.0
lief==0.13.2
tensorflow==2.15.0
numpy==1.26.3
unicorn==2.0.1
keystone-engine==0.9.2
scikit-learn==1.3.2
```

---

### Project structure
#### Main folders:
1. **MNEMONICS**  
   It contains instruction templates for different architectures:
- `SSE.txt `, `AVX.txt `, `AVX2.txt ` and others.  
   *Format: one mnemonic per line (for example, "addps")*

2. **NEO_RABBIT**  
   Knowledge base for replacing instructions (JSON files):
- `AVX2_AVX.json` - replaces AVX2 → AVX
- `F16C_AVX.json` - optimizations for float16
   - `BMI2_AVX.json` - bit operations  
   *Structure: {"original": "instruction", "replacement": ["substitution1", "substitution2"]}*

3. **AI_MODEL**  
   Stores neural network components:
   - `transformer_model.keras' - transformer model
   - `tokenizer.json` - tokenizer dictionary
- `model_config.json' - model parameters
- `feedback_data.json` - user edits

4. **CONFIG**  
   Configuration files:
   - `model_config.ini` - neural network training settings

---

### Strengths
1. **Flexible instruction replacement**
- Combining templates (JSON) and neural network predictions
   - Context-sensitive substitution (analysis of neighboring instructions)

2. **Validation of changes**
- Emulation via Unicorn Engine
   - Checking the identity of the results before/after the replacement

3. **Self-learning system**
- Saving successful substitutions in feedback_data.json`
   - Automatic retraining of the model when changing templates

4. **Secure export**
- Correct modification of PE files via LIEF
- Creation of a new section for long instructions

---

### Weaknesses
1. **Resource requirements**
- Model training requires significant computing power
   - Emulation of large binaries can be slow

2. **Limitations of emulation**
- Does not support complex system calls
- Problems with instructions that change the state of the system

3. **Template dependency**
- The quality of replacements depends on the completeness of the JSON templates
- Requires manual configuration for new architectures

---

### How to use
1. Download the EXE file via the `Open File`
2. Select the target instruction set (for example AVX2)
3. Run the analysis (`Analyze File')
4. Apply an AI replacement (`AI Replace`)
5. Check the results in the `AI Replacement` tab
6. Export the optimized file (`Export Binary`)

---

### Work features
- **Caching of predictions** - acceleration of repeated operations
- **Multi-level validation**:
1. Static analysis via Capstone
  2. Dynamic emulation via Unicorn
  3. Performance benchmarking
- **Adaptive learning** - the program offers to retrain the model when changing JSON templates

> **Important:** All substitutions are saved in the feedback_data.json` - you can add to the knowledge base manually, after which the program will automatically retrain the neural network.

For developers: all neural network parameters are configured via `CONFIG/model_config.ini` (layer sizes, learning rate, etc.).

ATTENTION!!! - the model needs to be constantly updated so that there are no errors in the program
