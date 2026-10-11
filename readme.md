# ReTrace

ReTrace is a step-level hallucination detection and selective verification
framework for reasoning LLMs.

## Project Structure

data/
    Raw and processed datasets.

models/
    Qwen3-8B model references, probes, trained classifiers, etc.

features/
    Extracted hidden-state and intermediate feature representations.

signals/
    Individual hallucination signals:
    ARS
    Deviation
    Consistency
    Causal

training/
    Training scripts and training matrices.

calibration/
    Threshold and probability calibration.

evaluation/
    Evaluation scripts and metrics.

notebooks/
    Experiments and exploratory analysis.