# Emulator 3pt

This package illustrates the use of HMC or nested sampler using emulator of 2x3pt shear aperture statistics.

---

## 🛠️ Installation

We recommend using a dedicated `conda` environment for clean dependency management.

### 1. Clone the Repository

```bash
git clone https://gitlab.euclid-sgs.uk/DR1-KP/WL-8/cosmoanalysis.git
cd cosmoanalysis
```

### 2. Create and Activate a Conda Environment

```bash
conda create -n emulator_3pt_env python=3.10 
conda activate emulator_3pt_env
```

### 3. Install the Package

Install the package and its dependencies using `pip`:

```bash
pip install .
```

If you need editable/development mode:

```bash
pip install -e .
```

---

## 📦 Dependencies

The core dependencies (automatically installed) include:

* numpy
* scipy
* keras
* jax
* jaxlib
* blackjax
* gdown
* matplotlib
* nautilus-sampler
* datetime
* getdist


---

## 📂 Structure

* `emulator_3pt/`: Code that computes the predictins using jax
* `outputs_gitlab/`: Example emulator weights (downloaded if needed)
* `examples/`: Notebooks to demonstrate emulator use


---

## 📬 Contact

For questions, please contact:
Pierre Burger – [pierre.burger@uwaterloo.ca](mailto:pierre.burger@uwaterloo.ca)
