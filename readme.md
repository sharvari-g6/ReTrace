🎯 7-Day Goal
By the end of Day 7, your team should have:
ProcessBench + PRM800K + HaluEval
              ↓
       Qwen3-8B hidden states
              ↓
     4 implemented signals
              ↓
     Training feature matrix
              ↓
       Step Probe trained
              ↓
         GeoReason R₀
              ↓
       Normalization stats
              ↓
     Logistic RFS model
              ↓
       RFS probabilities
              ↓
      Threshold θ + calibration
Final saved artifacts:
	step_probe
	R0
	normalization_parameters
	rfs_weights
	threshold_theta
	probability_calibration_model
These are exactly the outputs specified by your finalized plan.
________________________________________
👥 Work Division — 3 Members
I would divide the work by technical component, not simply "Phase 1–3 to Person 1", because the four signals are largely independent and can be developed in parallel.
Let's call you:
	Member A — You: Training + RFS Fusion + Calibration
	Member B - Nitish: Hidden-state/data pipeline + Step Probe + Consistency
	Member C - Shaishavi: ARS + GeoReason + Causal signal
This makes the division much more balanced.
________________________________________
📅 DAY 1 — Setup + Dataset Pipeline
👩‍💻 Member A — You
Phase 1 + training framework
	Set up project repository
	Python environment
	PyTorch
	Transformers
	sklearn
	scipy
	datasets
	numpy/pandas
	Define folder structure
	Define train/validation/test split strategy
	Create common feature format
Deliverable:
retrace/
├── data/
├── models/
├── features/
├── signals/
├── training/
├── calibration/
├── evaluation/
└── notebooks/
Also define the final training matrix:
sample_id
step_id
ARS
Deviation
Consistency
Causal
label
________________________________________
👩‍💻 Member B
Phase 2 + Phase 3
Build:
Dataset
   ↓
Qwen3-8B
   ↓
Reasoning trajectory
   ↓
Step segmentation
   ↓
step labels
Work on:
	ProcessBench loader
	PRM800K loader
	HaluEval loader
	common format
	step labels
	first-error labels
	answer labels
The implementation plan explicitly requires these three datasets to be converted into a unified structure, while noting that their annotation formats cannot simply be assumed identical.
Deliverable:
{
    "question": ...,
    "steps": [...],
    "step_labels": [...],
    "first_error": ...,
    "answer": ...,
    "answer_label": ...
}
________________________________________
👩‍💻 Member C
Phase 2 hidden-state extraction
Build the actual Qwen3-8B extraction pipeline:
Question
   ↓
Qwen3-8B
   ↓
tokens
   ↓
reasoning steps
   ↓
hidden states
Save:
question
steps
token_ids
step_boundaries
hidden_states
answer
Important: Don't duplicate Member B's dataset work. C focuses on model execution + hidden-state extraction.
________________________________________
📅 DAY 2 — Finish Data + Start Signals
👩‍💻 You — Member A
Start Phase 9 skeleton.
Create:
training/
    train_step_probe.py
    train_rfs.py
    generate_features.py

calibration/
    normalize.py
    threshold.py
    calibrate.py
Also decide the exact label convention:
0 = correct
1 = hallucinated/error
And define the train/validation/test separation before fitting anything.
This is important because normalization statistics, RFS weights and threshold must not leak information from the test set.
________________________________________
👩‍💻 Member B
Finish:
Phase 4 — Hidden-State Dataset Generation
Run:
ProcessBench
PRM800K
HaluEval
       ↓
Qwen3-8B
       ↓
hidden states
Create a reusable dataset such as:
train_features_raw/
    processbench.pt
    prm800k.pt
    halu_eval.pt
________________________________________
👩‍💻 Member C
Start Phase 5A — ARS.
Implement:
h_t
 ↓
latent perturbation
 ↓
counterfactual answer
 ↓
compare with original
 ↓
agreement rate
 ↓
ARS_t
The ARS paper specifically constructs counterfactual answers by perturbing the trace-boundary hidden representation and labels the resulting samples according to answer agreement/disagreement.
Deliverable:
ars_scores[trajectory][step]
________________________________________
📅 DAY 3 — Four Signal Implementation
Member A — You
Phase 9.2 — Step Probe
Start training:
h_t
 ↓
Logistic Regression / Step Probe
 ↓
P(error | h_t)
Use the finalized step-level supervision.
Output:
step_probe.pkl
Also generate:
Consistency_t
from the probe outputs across the trajectory.
The ACL paper's formulation explicitly separates local step-level evidence from the accumulated prefix-level hallucination state.
________________________________________
Member B
Phase 6 — GeoReason
Implement:
correct trajectories
       ↓
hidden-state transitions
       ↓
reference distribution R₀
       ↓
current transition
       ↓
Deviation_t
Need to produce:
R0
deviation_scores
GeoReason treats reasoning as a hidden-state trajectory and identifies errors through deviations in the geometry of transitions.
________________________________________
Member C
Finish:
Phase 5A — ARS
and start:
Phase 8 — Causal Signal
original h_t
    ↓
intervention
    ↓
new trajectory
    ↓
new answer
    ↓
compare A vs B
    ↓
Causal_t
Output:
causal_scores
________________________________________
📅 DAY 4 — Signal Integration
👩‍💻 You
Take everyone's outputs and build:
ARS
Deviation
Consistency
Causal
into ONE feature table:
trajectory	step	ARS	Deviation	Consistency	Causal	label
T1	1	...	...	...	...	0
T1	2	...	...	...	...	0
T1	3	...	...	...	...	1
This is Phase 9.1.
Also perform sanity checks:
	missing values
	NaNs
	extreme values
	class imbalance
	signal distributions
	correct/error separation
________________________________________
👩‍💻 Member B
Help validate GeoReason + Consistency.
Check:
	Does R0 contain only correct training trajectories?
	Does deviation increase on known erroneous steps?
	Does Step Probe produce sensible probabilities?
	Plot distributions.
Example:
Correct steps       Error steps
    ████                 ██
    █████                ████
    ██████               ███████
________________________________________
👩‍💻 Member C
Validate ARS + causal.
For ARS:
stable answer → low instability
unstable answer → high instability
For causal:
weak influence → low score
strong answer change → high score
Also run a small sample manually and verify the numbers.
This is critical. Don't immediately run everything on thousands of samples.
________________________________________
📅 DAY 5 — RFS Training
This is your main day.
👩‍💻 YOU — Lead this entire day
Phase 9.5 — Normalization
Fit statistics ONLY on training data:
μ_ARs, σ_ARs
μ_dev, σ_dev
μ_cons, σ_cons
μ_causal, σ_causal
Then:
z_i=(x_i-μ_i)/σ_i 
Save:
normalization.json
________________________________________
Phase 9.6 — Logistic Regression
Input:
[zARS,
 zDeviation,
 zConsistency,
 zCausal]
↓
Logistic Regression
↓
w1
w2
w3
w4
bias
↓
RFS
Train with binary cross-entropy/log loss.
The finalized implementation specifically chose logistic regression because there are only four signals and the resulting weights remain interpretable and easy to analyze/ablate.
________________________________________
Member B
Support you with:
	feature quality checks
	class balancing if necessary
	train/validation split verification
	correlation analysis
	feature distributions
	checking whether any feature dominates
Produce:
correlation_matrix.png
feature_distributions.png
________________________________________
Member C
Run signal ablations:
All 4
ARS only
Deviation only
Consistency only
Causal only

All except ARS
All except Deviation
All except Consistency
All except Causal
This will later be very useful for your paper.
________________________________________
📅 DAY 6 — Calibration
👩‍💻 You — Lead
Phase 10.1 — Generate validation RFS
Use:
validation data
      ↓
four signals
      ↓
normalization
      ↓
RFS model
      ↓
RFS probability
Example:
Step	RFS
1	0.08
2	0.12
3	0.19
4	0.78
5	0.91
________________________________________
Phase 10.2 — Determine threshold
Don't arbitrarily choose:
θ = 0.5
Instead evaluate thresholds across validation data.
For example:
θ = 0.30
θ = 0.40
θ = 0.50
θ = 0.60
θ = 0.70
θ = 0.80
Calculate:
	Precision
	Recall
	F1
	TPR
	FPR
Then select the threshold according to your project's chosen objective.
Save:
threshold.json
________________________________________
Phase 10.3 — Probability calibration
Compare:
Raw logistic probability
vs.
Platt/sigmoid calibration
vs.
Isotonic regression
Use validation data and calibration metrics/plots such as:
	reliability diagram
	Brier score
	Expected Calibration Error
Then choose the calibration method based on validation performance, as your finalized plan specifies.
________________________________________
Member B
Evaluate:
	calibration curves
	threshold curves
	confusion matrices
	AUROC
	F1
	precision/recall
Create the plots.
________________________________________
Member C
Stress-test the system:
	very short reasoning
	very long reasoning
	no hallucination
	early hallucination
	late hallucination
	multiple errors
	recovery after an error
	contradictory reasoning
Check whether the four signals behave sensibly.
________________________________________
📅 DAY 7 — Integration + Freeze
This should NOT be a coding-heavy day.
It should be your verification day.
👩‍💻 You
Run the complete:
TRAIN
 ↓
Step Probe
 ↓
R₀
 ↓
Normalization
 ↓
RFS
 ↓
Calibration
pipeline from scratch.
Make sure someone can run:
python train_step_probe.py
python build_R0.py
python generate_features.py
python train_rfs.py
python calibrate_rfs.py
and reproduce the outputs.
________________________________________
👩‍💻 Member B
Create final training/calibration evaluation report:
Step Probe:
AUROC
F1
Precision
Recall

RFS:
AUROC
F1
Precision
Recall

Calibration:
Brier
ECE
Reliability curve
________________________________________
👩‍💻 Member C
Create final signal sanity report.
For 5–10 manually inspected trajectories:
Step
ARS
Deviation
Consistency
Causal
RFS
Ground Truth
Prediction
Example:
Step	ARS	Dev.	Cons.	Causal	RFS	GT
1	.08	.12	.05	.02	.07	Correct
2	.12	.19	.08	.04	.11	Correct
3	.71	.82	.76	.81	.87	Error
4	.75	.85	.82	.88	.92	Error
________________________________________
🔥 Your Personal Work — More Specifically
Since you asked "give me", I'd make your role the central training/calibration role.
Your 7-day responsibility
Day	Your main task
Day 1	Environment + training architecture
Day 2	Training pipeline + feature schema
Day 3	Step Probe
Day 4	Integrate 4 signals into feature matrix
Day 5	Normalization + Logistic RFS
Day 6	Threshold + probability calibration
Day 7	Full pipeline + freeze artifacts
Your main ownership is therefore:
                  YOU
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
 Step Probe              RFS Training
        │                     │
        ↓                     ↓
 Consistency            Normalization
                              │
                              ↓
                         Logistic RFS
                              │
                              ↓
                         Calibration
                              │
                              ↓
                        Threshold θ
The other two members feed you the signal outputs.
________________________________________
⚠️ One Important Change I Recommend
Don't interpret "7 days" as 7 days of writing final code.
Your dependency chain is:
Phase 1
   ↓
Phase 2
   ↓
Phase 3
   ↓
Phase 4
   ↓
Phase 5–8  ← parallel
   ↓
Phase 9
   ↓
Phase 10
But Phases 5–8 should be parallelized:
                Phase 4
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
       ARS      GeoReason   Step Probe
        │          │          │
        └──────────┼──────────┘
                   ↓
             Causal Signal
                   ↓
             Feature Matrix
                   ↓
              Phase 9
                   ↓
              Phase 10
So you absolutely should not wait until Day 5 to start Phase 9. Start building its code skeleton on Day 1, while the other members are producing the signal modules.
By the end of 7 days, the exact checkpoint should be:
DONE
	Qwen3-8B extraction
	ProcessBench processed
	PRM800K processed
	HaluEval processed
	Hidden-state dataset generated
	ARS working
	GeoReason working
	Step Probe working
	Causal signal working
	Four-signal feature matrix generated
	Step Probe trained
	R_0learned
	Normalization parameters learned
	Logistic RFS trained
	RFS weights obtained
	Threshold selected
	Calibration tested
	Final artifacts frozen
NOT YET
	❌ KB
	❌ FAISS
	❌ selective verification
	❌ correction
	❌ re-tracing
	❌ FastAPI/UI
Those belong to the later phases of your implementation plan.
One thing I would do next: make a Day 1–7 task sheet with exact tasks/subtasks for you, Member 2, and Member 3, including what file/code each person should create and what output they must hand over each day. That will make this much easier to execute without overlap.

