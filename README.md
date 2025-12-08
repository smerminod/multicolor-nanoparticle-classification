# Multicolor Nanoparticle Classification Pipeline

A **physics-informed, configuration-driven** image analysis pipeline for classifying rare-earth doped, multicolor nanoparticles using **simultaneous cathodoluminescence (CL) and secondary electron (SE) imaging**, co-registered with higher-resolution **scanning electron microscopy (SEM)**. 

This architecture leverages **multi-modal data fusion** to solve the inverse problem of characterizing heterogeneous mixtures at the single-particle level, utilizing **Bayesian inference** to rigorously decode element-specific optical signatures. The nanoparticle synthesis, imaging technique development, and data acquisition were performed at the [Prigozhin Lab at Harvard University](https://prigozhin.hsites.harvard.edu).


[![Python CI](https://github.com/smerminod/multicolor-nanoparticle-classification/actions/workflows/ci.yaml/badge.svg)](https://github.com/smerminod/multicolor-nanoparticle-classification/actions)
[![codecov](https://codecov.io/gh/smerminod/multicolor-nanoparticle-classification/graph/badge.svg)](https://codecov.io/gh/smerminod/multicolor-nanoparticle-classification)
[![Python 3.12-3.13](https://img.shields.io/badge/python-3.12--3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Quick Navigation

**Key Sections**: 
[The Scientific Problem](#the-scientific-problem) · [Key Computational Challenges](#key-computational-challenges) · [Methodology](#methodology) · [Quick Start](#quick-start) · [Docker Usage](#docker-usage) · [Configuration System](#configuration-system) · [Citation](#citation)

> 📖 **Full Navigation**: Use GitHub's table of contents (click the ≡ menu icon in the top-right corner) to jump to any section.

---

## The Scientific Problem

Rare-earth doped nanoparticles (NaF₄:RE³⁺)—often referred to in literature as Lanthanide Nanoparticles (LNPs)—are critical materials for applications in bioimaging, photonics, and quantum technologies. However, **determining the elemental composition of individual nanoparticles in heterogeneous mixtures remains a significant challenge**. Traditional spectroscopic methods do not provide single-particle optical properties, while electron microscopy alone only provides structural information.

This project addresses this gap by developing a computational pipeline to analyze a **novel multi-modal dataset** where each nanoparticle is imaged using:
1) **Scanning Electron Microscopy (SEM)**, for high-resolution structural information
2) **Simultaneous CL-SE Imaging:** A lower-resolution scan capturing:
    - **Cathodoluminescence (CL):** Electron-beam induced photon emission, captured in three optical channels correlating with dopant atomic transitions
    - **Secondary Electrons (SE):** For correlated structural information

This multi-modal imaging enables **direct mapping of optical properties to physical structure at the single-particle level**, opening new possibilities for composition analysis and population decoding.

### Cathodoluminescence (CL)

**CL acts as a spectral fingerprint:** when excited by the electron beam, rare-earth dopants exhibit **element-specific photon emission**. This pipeline classifies particles by matching their observed spectral signatures to known elemental compositions.

| Element | Peak Emission Wavelength | Detection Filter Range | CL Channel Name | 
|---------|-------------------|------------|------------------|
| **Tb** (Terbium) | 547 nm (green) | 530—560 nm | "Blue" |
| **Dy** (Dysprosium) | 573 nm (green-yellow) | 563—588 nm | "Green" |
| **Ho** (Holmium) | 646 nm (red) | 630—660 nm | "Red" |
| **Gd** (Gadolinium) | No visible CL | — | — |
| **Y** (Yttrium) | No visible CL | — | — |

> [!NOTE]
> "CL Channel Name" serves as a relative configuration alias rather than a fixed physical descriptor. Mapping Terbium (547 nm) to "Blue" not only distinguishes it from Dysprosium ("Green") but also enforces a generalized, element-agnostic schema. This allows the pipeline to readily adapt to future material systems with different spectral profiles without modifying the core codebase.

### Representative Results

<p align="center">
  <img src="assets/Figure_AbdulRehman_2025_SI-Fig19_Classification.png" width="800" alt="Bayesian Classification Framework">
</p>
<p align="left">
  <strong>Figure 1: Probabilistic Bayesian classification framework.</strong> Representative results from a 2-color dataset (Dysprosium displayed as green, Holmium as magenta) using a prototype analysis pipeline. <strong>(a-c)</strong> Automated exclusion masking isolates single-particle signals. <strong>(d-h)</strong> Maximum Likelihood Estimation (MLE) of CL photon rates and posterior probability calculation for dopant assignment. <strong>(i)</strong> Class-based false-color SE image showing the spatial distribution of assigned dopants. <strong>(j–o)</strong> Population statistics and monotonic brightness-to-size scaling laws. <em>Reproduction from Abdul Rehman et al., Nature Communications (2025), Supplementary Figure 19</em>.
</p>

---

## Key Computational Challenges

Our analysis of cathodoluminescence (CL) datasets necessitates robust data processing strategies. This involves specific signal processing challenges with analogies to those found in neuroimaging:
1. **Multi-Modal Data Fusion**: Integrating functional data (low-resolution CL) with structural data (high-resolution SEM) to bridge the resolution gap between elemental identity and physical morphology, **similar to fMRI-to-MRI registration**.
2. **Signal Unmixing**: High particle density leads to CL signal overlap between adjacent nano-emitters. Furthermore, non-local electron excitation creates long-range spatial crosstalk **similar to volume conduction** in electrophysiology, where the recorded signal is a superposition of local and distant sources.
3. **Probabilistic Population Decoding**: Going beyond simple classification to **probabilistic decoding of heterogeneous populations** of nanoparticles. The challenge is to infer a latent variable (elemental composition) from noisy observations (CL photon counts).

---

## Methodology

We address these challenges through a two-phase approach, beginning with our established processing pipeline and evolving into an end-to-end deep learning framework (from `v2.0`).

### Core Methodology (`v1.0` target)

The pipeline is designed to implement an automated dopant assignment framework adapted from Supplementary Note 13 of Abdul Rehman *et al.*, *Nature Communications* (2025):
1. **Semi-Automated Co-Registration**: We utilize structural landmarks (large nanoparticles) visible in both SEM and SE modalities to compute an affine transformation matrix. This maps the high-resolution structural images (SEM) onto the low-resolution functional images (SE/CL).
2. **Spatial Demixing with Structural Priors**: We employ particle-specific exclusion masks derived from bivariate Gaussian modeling of the structural SE image. This acts as a spatial filter, isolating intrinsic particle signals from neighbor contamination by excluding pixels shared between adjacent emitters.
3. **Probabilistic Decoding: Bayesian Inference**
    - **Forward Modeling (Encoding)**: We first estimate photon emission rates using MLE on the filtered emitter signals. We then fit these rates to quadratic surface models calibrated by particle diameter (structural feature) and integration time. This establishes the encoding model `P(Data∣Class),` where `Data` is the observed photon rate (conditioned on particle size and imaging time) and `Class` is the elemental composition.
    - **Bayesian Inversion (Decoding)**: We compute the posterior probability `P(Class∣Data)` for each particle. This combines the likelihood (from the encoding model) with priors based on sample stoichiometry, enabling the probabilistic resolution of ambiguous signals.

### Future Roadmap (`v2.0+`)

From version `v2.0`, we will transition to a **Probabilistic Video Vision Transformer architecture (ViViT + Bayesian Layer)**. This end-to-end deep learning framework will learn to **perform simultaneous signal unmixing and probabilistic decoding**, replacing explicit physical forward models with learned representations. It will resolve complex, non-linear environmental interactions that heuristic masks cannot address. This model will be trained on large-scale synthetic CL/SE datasets generated via the Quantum Monte Carlo program [CASINO](https://vallico.net/casinoqmc/).

---

## Why This Architecture?

The **declarative YAML-based configuration system** ensures reproducibility and scalability. Furthermore, the automated directory hierarchy adopts **data organization principles analogous to BIDS (Brain Imaging Data Structure)**: it enforces a strict separation of structural (SEM, SE) and functional (CL) modalities, isolates immutable `raw/` acquisitions from `processed/` derivatives, and strictly couples imaging data with centralized metadata registries to ensure full data provenance.

### Key Features

- **Configuration-Driven Architecture**: Three-tier YAML system (class definitions, processing templates, experiment registry)
- **Probabilistic Multivariate Classification**: Implements the Bayesian framework described above
- **Full Data Provenance**: The configuration system creates an unbreakable audit trail. Every processed output can be traced back to the exact experimental parameters and software version used to generate it.
- **Comprehensive Validation**: Pydantic schemas with cross-file reference checking and scientific constraint validation
- **Automated Path Management**: Intelligent directory structure generation following best practices conventions
- **Extensive Testing**: 120 tests with 93% coverage, including integration tests with real configuration files
- **CI/CD Pipeline**: Automated testing, linting (Ruff), and coverage reporting (Codecov)
- **Containerization**: Reproducible Docker environment for both production pipelines and interactive development

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/smerminod/multicolor-nanoparticle-classification.git
cd multicolor-nanoparticle-classification

# Install dependencies
pip install -r requirements.txt

# Or install in development mode with test dependencies
pip install -e ".[test]"
```

### Run the demo

> [!NOTE]
> The current demo runs in metadata-only mode to validate the **configuration architecture** and **path generation logic**. Raw `.tif` files are excluded from the repository due to size, though a minimal sample dataset will be added in an upcoming release.

```bash
# Show available experiments and configuration validation
python main.py

# Demonstrate configuration system for a specific experiment
python main.py 2024-06-29_Dy100
```

What the demo shows:
- ✅ Comprehensive YAML validation (cross-file references, elemental fractions, priors)
- ✅ Experiment discovery and metadata loading
- ✅ Automatic path generation for raw/processed/metadata files
- ✅ Channel-specific configuration queries
- ✅ Directory structure that the pipeline expects

**To process actual data**: Follow the same directory structure as `data/training/physical/single_particle/<experiment_id>/` and add `.tif` files to process. 
Full datasets will be made available upon publication.

### Interactive Demo

An interactive walkthrough of the configuration system architecture and path management utilities:

```bash
# Launch Jupyter and open the configuration system demo
jupyter notebook notebooks/00_configuration_system_demo.ipynb
```

---

## Docker Usage

The project is **fully containerized** to **ensure reproducibility** across different computing environments. The Dockerfile supports both a lightweight **Production** mode (runtime only) and a **Development** mode (testing tools & Jupyter).

### Build & Run

```bash
# Option A: Production build
docker build -t nanoparticle-classifier .

# Option B: Development build
docker build --build-arg INSTALL_DEV=true -t nanoparticle-classifier:dev .

# Run the demo
docker run --rm nanoparticle-classifier

# Run tests (dev image only)
docker run --rm nanoparticle-classifier:dev pytest
```

### Process Local Data

Mount your local data directory to **process your own experiments**.

```bash
docker run \
  -v "/path/to/my_data:/app/data" \
  -v "/path/to/my_experiments.yaml:/app/experiments.yaml" \
  -v "/path/to/my_class_definitions.yaml:/app/class_definitions.yaml" \
  nanoparticle-classifier <my_experiment_id>
```
**Prerequisites**: External data must follow the same directory structure as `data/training/physical/single_particle/<experiment_id>/`

### Interactive Tuning

Mount your local source directory into the container to edit configurations locally and test immediately without rebuilding.

```bash
# Launch a bash shell with your current directory mounted to /app
docker run --rm -it -v "$(pwd)":/app nanoparticle-classifier:dev /bin/bash

# Inside the container:
# 1. Edit e.g. config.yaml on your host machine (changes appear instantly)
# 2. Run the pipeline
python main.py <experiment_id>
# 3. Exit
exit
```

---

## Data Structure

Each experiment's data is organized hierarchically by **"region"** and **acquisition type**. Note that we use the term **"region"** (instead of "field of view") for a specific sample location to create a **unified ID** that groups the slightly offset field of views from the CL/SE versus SEM acquisitions.

```
data/training/physical/single_particle/2024-06-29_Dy100/  # Example experiment
├── raw/
│   └── region_1/                    # First imaging region
│       ├── CL_SE/                   # Combined CL + SE acquisition
│       │   ├── CL/
│       │   │   ├── blue/            # "blue" filter frames (Tb emission)
│       │   │   ├── green/           # "green" filter frames (Dy emission)
│       │   │   └── red/             # "red" filter frames (Ho emission)
│       │   └── SE/                  # Secondary electron frames
│       └── SEM/                     # High-resolution SEM image
├── processed/
│   └── summed_images/               # Processing method
│       └── region_1/
│           ├── CL_SE/CL/blue/
│           └── ...
└── metadata/
    └── 2024-06-29_Dy_regions.csv   # Region coordinates and acquisition metadata
```

> [!NOTE]
> The directory of sample experiment `2024-06-29_Dy100` currently contains `.gitkeep` placeholders to preserve the BIDS-like hierarchy. Full datasets will be made available upon publication.

---

## Project Structure

```
.
├── class_definitions.yaml      # Nanoparticle class definitions with elemental fractions
├── config.yaml                 # Processing templates and configuration anchors
├── experiments.yaml            # Experiment registry with metadata
├── main.py                     # Demo/smoke test entry point
│
├── src/
│   ├── data/
│   │   ├── schemas.py          # Pydantic validation schemas (89 tests)
│   │   └── config_loader.py    # Configuration loading and merging (31 tests)
│   ├── processing/             # Image preprocessing (in development)
│   ├── features/               # Feature extraction (planned)
│   ├── models/                 # Classification models (planned)
│   └── visualization/          # Plotting utilities (planned)
│
├── tests/
│   ├── test_schemas.py         # Schema validation tests
│   └── test_config_loader.py   # Config loader validation tests
│
├── notebooks/
│   └── 00_configuration_system_demo.ipynb  # Interactive demo
│
├── data/                       # Experimental data (structure auto-generated)
│   ├── training/
│   │   └── physical/single_particle/2024-06-29_Dy100/  # Example experiment
│   └── inference/
│       └── physical/multi_particle/2025-07-02_6-component-mixture/  # Example experiment
│
└── scripts/                    # (Planned)
```

---

## Configuration System

The pipeline uses a **3-tier YAML configuration system** to separate concerns:
1. **`class_definitions.yaml`**: Defines elemental identity (e.g., `Dy100` = 100% Dysprosium)
2. **`config.yaml`**: Defines processing templates and logical channel identifiers (`blue`, `green`, `red`) to abstract hardware specifics
3. **`experiments.yaml`**: The registry of all datasets, linking metadata to specific configurations

The configuration system performs **comprehensive cross-validation**, including:
- ✅ All `composition_label` references from `experiments.yaml` exist in `class_definitions.yaml`
- ✅ All config template names from `experiments.yaml` exist in `config.yaml`
- ✅ Elemental fractions in `class_definitions.yaml` sum to 1.0
- ✅ Mixture priors in `experiments.yaml` (for inference experiments) sum to 1.0

---

## Developer Guide

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Fail if coverage drops below 80%
pytest tests/ --cov=src --cov-fail-under=80

# Run only unit tests (skip integration tests)
pytest tests/ -m "not integration"
```

### Code Quality

```bash
# Lint with Ruff
ruff check src tests main.py

# Auto-fix linting issues
ruff check --fix src tests main.py
```

---

## Project Status

**Current Phase:** Architecture & Configuration Setup (v0.1.0 - Architectural Commit)
- ✅ **Completed**:
    - Configuration system
    - Testing (93% coverage)
    - Docker
    - CI/CD
- 🚧 **In Development**:
    - Image preprocessing (drift correction)
    - Feature extraction (particle morphology, de-mixed photon emission rates)
- 📋 **Planned**:
    - Bayesian inference model (calibrated on `physical` experiments)
    - Probabilistic deep learning model (ViViT + Bayesian Layer, using both `physical` and `simulated` experiments as training data)

---

## Development History

This repository represents a complete architectural rewrite and modernization of the original MATLAB algorithms used to generate the results in Abdul Rehman *et al.* (2025). 

While the original prototype validated the Bayesian classification framework, this Python implementation introduces:
- **Reproducibility**: Transitioning from manual script execution to a configuration-driven pipeline.
- **Maintainability**: Adoption of strict type-hinting, unit testing, and modular design.
- **Accessibility**: Removal of proprietary license requirements (MATLAB) to foster open-source collaboration.

---

## Documentation

- **[README.md](README.md)**: Architecture documentation, design decisions, development guidelines, and testing strategy
- **[notebooks/00_configuration_system_demo.ipynb](notebooks/00_configuration_system_demo.ipynb)**: Interactive demonstration of the configuration system with executable examples
- **Inline Documentation**: Comprehensive Google-style docstrings throughout source code

---

## Requirements

- **Python 3.12-3.13** (uses modern type hinting and pattern matching features)
- **Docker** (optional, for containerized execution)

### Core Dependencies
- `pydantic` (≥2.0) — Data validation
- `pyyaml` — Configuration file parsing
- `scikit-image` — Image processing
- `numpy` — Numerical computing
- `pandas` — Metadata handling

See [`requirements.txt`](requirements.txt) for pinned versions and complete dependency tree.

---

## Contributing

This is a research project under active development. For questions, bug reports, or collaboration inquiries, please [open an Issue](https://github.com/smerminod/multicolor-nanoparticle-classification/issues).

---

## License

MIT License — See [**LICENSE**](LICENSE) for details.

---

## Citation

- **Underlying Research:** This pipeline implements and expands the probabilistic classification framework from the following publication:<br>

  > Abdul Rehman\*, Conway\*, ..., **Merminod**, ..., & Prigozhin (2025). "Multicolor cathodoluminescence imaging of single lanthanide nanoparticles." *Nature Communications*, 16, 9632. DOI: 10.1038/s41467-025-64409-8

  ```bibtex
  @article{AbdulRehman2025,
    title = {Multicolor cathodoluminescence imaging of single lanthanide nanoparticles},
    volume = {16},
    issn = {2041-1723},
    url = {https://doi.org/10.1038/s41467-025-64409-8},
    doi = {10.1038/s41467-025-64409-8},
    number = {1},
    journal = {Nature Communications},
    author = {Abdul Rehman, Sohaib and Conway, Jeremy B. and Nichols, Amy and Soucy, Edward R. and Dee, Amanda and Stevens, Kristal and Merminod, Simon and MacNaughton, Isabella and Curtis, Abigail and Prigozhin, Maxim B.},
    month = oct,
    year = {2025},
    pages = {9632},
  }
  ```

- **Software:** If you use this codebase, please cite it as:

  > **Merminod, S.** (2025). *Multicolor Nanoparticle Classification Pipeline* (Version 0.1.0) [Computer software]. GitHub. https://github.com/smerminod/multicolor-nanoparticle-classification   

---

## Author

**Simon Merminod** | Visit my [website](https://www.simonmerminod.com)

---

## Acknowledgments

- Configuration architecture inspired by best practices in scientific computing and reproducible research
- Testing approach follows modern Python conventions (pytest, parametrization, fixtures)
- CI/CD setup based on GitHub Actions best practices for Python projects
- Nanoparticle synthesis, multi-modal SEM-based imaging technique, and data acquisition performed by collaborators at the [Prigozhin Lab at Harvard University](https://prigozhin.hsites.harvard.edu) (see Author Contributions section in Abdul Rehman *et al.* (2025)).
