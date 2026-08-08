# Python port of the Secretary Bird Optimization Algorithm (SBOA) MLP training

This folder contains a minimal Python re-implementation of the SBOA MLP training code
ported from the R repository. The goal is to reproduce the experiments and provide a
clean Python API.

Structure added in branch python-port/sboa:
- mlp_sboa/: core modules (model, optimizers, fitness, metrics, utils)
- examples/: a runnable example script that trains SBOA on a chosen dataset
- requirements.txt: Python dependencies

How to run the example (locally)
1. Install requirements: pip install -r requirements.txt
2. Place datasets (unzipped) under `data/` at the repository root (we also accept data.zip)
3. Run the example script: python examples/sboa_example.py

Notes
- The example is configured to look for datasets in `data/` by default and will try to
  unzip `data.zip` if present.
- For quick experimentation the example uses reduced population/iteration counts.
