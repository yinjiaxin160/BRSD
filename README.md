# BRSD

> **Boundary-aware Risk-driven Test Prioritization for Medical Image Segmentation Systems**

This repository contains anonymized supplementary material for the paper:

**“BRSD: Boundary-aware Risk-driven Test Prioritization for Medical Image Segmentation Systems.”**

The repository contains the anonymized implementation and experimental
materials used for evaluating BRSD. Some implementation details that are
not required for understanding the experimental results are omitted.

The complete source code and reproduction package will be made publicly available upon acceptance.

---

## Overview

BRSD is a test-input prioritization method for DNN-based medical image segmentation systems. Given a candidate test pool and a limited testing budget, BRSD prioritizes inputs that are more likely to reveal failures during subsequent perturbation-based robustness testing.

This repository contains the anonymized implementation of BRSD together with the five baseline prioritization methods evaluated in the paper.

The selector implementations are located in `fuzzing/selectors/`:

| File | Method | Type |
|---|---|---|
| `random_selector.py` | Random | Baseline |
| `entropy_selector.py` | Entropy | Baseline |
| `deepgini_selector.py` | DeepGini | Baseline |
| `moo_nci_selector.py` | MOO-NCI | Baseline |
| `moo_cf_selector.py` | MOO-CF | Baseline |
| `brsd_selector.py` | **BRSD** | Proposed |

The selectors follow a common prioritization interface and can be integrated into the same evaluation pipeline.

---

## Repository Layout

```text
.
├── fuzzing/
│   ├── attacks/       # Perturbation implementations used for evaluation
│   ├── evaluators/    # Evaluation on clean and perturbed inputs
│   └── selectors/     # Test-input prioritization methods
│
├── model/             # Segmentation model implementations
│
├── utils/             # Dataset loading, losses, and evaluation utilities
│
├── trainroot/         # Training scripts
│
├── requirements.txt
├── Makefile
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Check the installation

```bash
make smoke
```

This command checks the basic repository environment and selector imports. No dataset or trained checkpoint is required for the smoke test.

### 3. Selector Interface

The prioritization methods operate on a trained segmentation model and a candidate test loader and return a priority ordering of candidate inputs.

A typical workflow is:

```python
selector = Selector(budget=K)

priority = selector.get_priority_sequence(
    dataloader=loader,
    model=model,
    device="cuda",
    criterion=criterion,
)

selected_inputs = priority[:K]
```

The concrete constructor and optional arguments may differ slightly across selectors. Please refer to the corresponding implementation file for the available interface.

---

## BRSD

BRSD estimates the failure susceptibility of candidate test inputs before allocating the downstream robustness-testing budget.

At a high level, the method contains three stages.

### 1. Risk Evidence Extraction

BRSD extracts complementary information from the model prediction and intermediate representation, including uncertainty-related, boundary-related, structural, and semantic information.

### 2. Local-Response Assessment

BRSD evaluates how a candidate segmentation responds to a weak localized probe. The resulting prediction response provides additional evidence about whether the candidate is susceptible to failure under subsequent perturbation-based testing.

### 3. Budgeted Prioritization

The collected risk evidence is aggregated to prioritize high-risk candidates. Semantic information is additionally used during selection to reduce redundant choices under the fixed testing budget.

The formal definitions and methodological details are described in the paper.

---

## Anonymous Artifact Notice

This repository is intended for anonymous peer review.

Accordingly, the public review version of `brsd_selector.py` preserves the overall organization, interfaces, and conceptual processing stages of BRSD, while withholding selected implementation details of the proposed method.

In particular, the anonymous version does **not** disclose:

- exact component weights and numerical hyperparameters;
- probe-specific perturbation settings;
- internal thresholds and candidate-selection parameters;
- the complete implementation of the boundary-aware local-response probe;
- the exact risk aggregation and selection configuration.

The corresponding locations in the source code contain explanatory placeholders describing their role in the overall method.

These omissions are limited to the proposed BRSD implementation. The paper provides the methodological definitions necessary to understand the approach and evaluate the reported experimental results.

The complete implementation, configuration, and reproduction package will be released publicly upon acceptance.

---

## Mapping from the Paper to the Repository

| Paper component | Repository location |
|---|---|
| BRSD prioritization framework | `fuzzing/selectors/brsd_selector.py` |
| Baseline prioritization methods | `fuzzing/selectors/` |
| Perturbation-based evaluation | `fuzzing/attacks/` |
| Evaluation pipeline | `fuzzing/evaluators/` |
| Segmentation models | `model/` |
| Dataset and evaluation utilities | `utils/` |

For BRSD-specific components whose implementation is withheld in the anonymous artifact, the corresponding functions contain high-level descriptions of their purpose and role.

---

## Reproducibility and Code Availability

The current repository is an anonymized review artifact rather than the final reproduction package.

It is intended to expose the experimental code organization, baseline implementations, evaluation pipeline, and the integration structure of BRSD without disclosing selected implementation-specific details of the unpublished method.

Upon acceptance, the complete release will include the full BRSD implementation and the configuration required to reproduce the experiments reported in the paper.

---

## License

The complete public release is planned to be provided under the MIT License.
