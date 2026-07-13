"""Shared source package for the foveated-vision thesis notebooks.

Modules are added incrementally, alongside the notebook that first needs them:
  - common.py         (00_setup_and_data)       seeding, config/checkpoint I/O, env stamp
  - datasets.py        (00_setup_and_data)       dataset prep + normalization conventions
  - eval_harness.py    (00_setup_and_data on)    evaluation metrics (mCE now; PGD-with-EOT
                                                  added in 01; Brain-Score and shape-bias
                                                  added in 03/05)
  - models.py           (01_baseline_reproduce)   B0-B3 baseline builders, raw-pixel-input
                                                  wrapper for attacks
  - overrides.py        (01_baseline_reproduce)   monkey-patch fix for VOneBlock's
                                                  input-gradient bug (needed for PGD/EOT);
                                                  V1-noise ablation flags added in 03
  - foveation.py         (02_foveation_...)        R-Blur, Watson pooling, SNR(r), trace-based
                                                  periphery synthesis
  - it_feedback.py       (04_it_feedback_...)      FoveatedModel loop, fixation policy, halting
  - mftma.py             (05_mftma_certification)  manifold capacity/radius/dimension wrapper

Nothing in this package edits code under external/ (the read-only clones of VOneNet,
CORnet, FOVEA, ...). Behavioral changes are subclasses, monkey-patches or wrapper
nn.Modules defined here, per TEKNIK_REHBER.md Sec. 3 (override strategy).
"""
