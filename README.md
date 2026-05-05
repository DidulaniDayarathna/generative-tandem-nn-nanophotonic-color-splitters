# Generative Tandem Neural Network for Inverse Design of Color Splitter Structures

This repository implements a **Generative Tandem Neural Network (GTNN)** for the inverse design of **color splitter nanophotonic structures**, published in:

> *A generative tandem neural network framework for inverse design of optical color splitters*  
> (IOP Publishing: https://iopscience.iop.org/article/10.1088/1402-4896/adba19/meta)

---

## Project Overview

Inverse design in nanophotonics is challenging due to the highly non linear relationship between structure geometry and optical response.

This project implements a **data-driven inverse design framework** using a **tandem neural network architecture**, where:

- A **forward model** (pretrained ResNet50-based network) learns the mapping:
  - **Structure --> Spectral/optical response**
- An **inverse model (generator)** learns:
  - **Spectral/optical response --> Structure parameters**
- The models are combined into a **tandem network** to optimize inverse design using forward model feedback.

---

## Key Idea

Instead of directly solving an ill posed inverse problem, the approach:

1. Generates structure candidates using a neural network
2. Feeds them into a frozen forward model (ResNet50 based surrogate)
3. Compares predicted output with target response
4. Backpropagates error through the forward model to train the inverse network

This enables stable training for inverse design problems that are otherwise non unique.

---

## Model Architecture

### Forward Model
- Pretrained **ResNet50 backbone**
- Acts as a surrogate physical model
- Learns mapping from structure representation --> optical response

### Inverse Model (Generator)
- Inputs:
  - Noise vector
  - Target spectral response
  - Channel/conditioning information
- Outputs:
  - Predicted structure representation

### Tandem Network
- Combines inverse + forward model
- Forward model weights are frozen
- Loss computed on the final predicted optical response vs ground truth

---

## Training Pipeline

The training workflow follows:

1. Load dataset (structure response pairs)
2. Normalize input/output data
3. Train test split 
4. Build inverse model
5. Attach frozen forward model (ResNet50)
6. Train tandem network using:


generative-tandem-nn-nanophotonic-color-splitters/
├── models/
│   ├── inverse_network_tuner.py
│   ├── forward_network_tuner.py
├── training/
│   ├── train_inverse.py
│   ├── train_forward.py

---

### Citation
If you use this repository in your research, please cite the following work:

@article{acharige2025generative,
  title={Generative tandem neural network for optimization of nanophotonic color splitters},
  author={Acharige, Didulani and Johlin, Eric},
  journal={Physica Scripta},
  volume={100},
  number={4},
  pages={046006},
  year={2025},
  publisher={IOP Publishing}
}
