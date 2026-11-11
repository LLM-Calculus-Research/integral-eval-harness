# Integral Eval Harness

This repository is used to evaluate language models on symbolic integration tasks.

## Datasets

### 1. Custom SymPy-based dataset (`datasets/simple_own_dataset`)

This is a lightweight dataset generated from SymPy expressions and designed for testing multivariable integration tasks.

- Location: `datasets/simple_own_dataset/multivariable_integrals_dataset.json`
- Origin: generated from SymPy symbolic expressions using the project script `create_dataset_integrals.py`
- Data type: symbolic integration problems with multiple variables
- Record fields:
  - `id`
  - `category`
  - `variables_order`
  - `expression_str`
  - `prompt_latex`
  - `ground_truth_latex`
- Content: mainly polynomial expressions and simple mixed 2D/3D cases involving `x`, `y`, and `z`, with trigonometric and exponential terms.
- Purpose: to test whether a model can correctly compute an integral and return the result in LaTeX format.

The dataset is derived from the SymPy library and the generated expressions are produced programmatically for controlled symbolic-integration evaluation.

### 2. Rubi dataset (`datasets/rubi`)

This is a larger and more comprehensive benchmark for symbolic integration.

- Location: `datasets/rubi/`
- Source: the Rubi rule-based integration test suite
- Size: 64,740 indefinite integration problems
- Structure: chapter-based directories and JSONL files containing integration cases
- Coverage: 8 chapters including algebraic, exponential, logarithmic, trigonometric, inverse trigonometric, hyperbolic, inverse hyperbolic, and special functions
- Additional metadata: `integral`, `num_steps`, `source`, and provenance information for each problem

Rubi is a well-known benchmark for comparing symbolic integration systems. In this project, it serves as a reference dataset for evaluating model performance.

## Project summary

This repository compares language-model outputs against reference symbolic solutions from:

- a custom SymPy-generated dataset,
- the Rubi benchmark.

The script `llm_eval.py` loads tasks, sends them to an Ollama model, and compares the generated result against the correct symbolic solution.
