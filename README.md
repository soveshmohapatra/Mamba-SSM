# Deconstructing State Space Models (Mamba) 🐍

*A "Build in Public" mini-series breaking down State Space Models (SSMs) and Mamba.*

![Banner](Post-1-Banner.png)

## Overview

Transformers have a fatal flaw: the Attention mechanism scales quadratically. This repository is part of a "Build in Public" mini-series completely deconstructing State Space Models (SSMs). 

SSMs compress the sequence context into a hidden "state" that updates linearly as new data flows in—providing fast inference, linear scaling, and infinite context potential.

### Series Outline
1. **Part 1**: The fundamental math (Control Theory 101 for AI)
2. **Part 2**: Writing a 1D State Space layer in under 100 lines of pure PyTorch
   ![Part 2 Banner](Post-2-Banner.png)
3. **Part 3**: Scaling it up and testing it against a baseline Transformer on long sequences

## Code Structure

### `simple_ssm.py`
A minimal, educational 1-Dimensional Linear Time-Invariant (LTI) State Space Model layer in pure PyTorch.

**Features:**
- Continuous-time parameters mapped to discrete transitions via Zero-Order Hold (ZOH).
- **Fast Convolutional Training**: O(L log L) path using FFT-backed convolution for massive parallel training.
- **O(1) Memory Recurrence**: Autoregressive generation path for deployment.
- Strict validation that both paths yield mathematically identical results.

## Quick Start

You can run the minimal SSM implementation directly:

```bash
python simple_ssm.py
```

This will run a quick sanity check validating the mathematical LTI duality between the Convolutional and Recurrent paths.

## Articles and Posts
The detailed write-ups for the series are available online:

**Part 1: The Intuition and Mathematics**
- 🔗 [Read the Full Blog Post](https://soveshmohapatra.com/projects/mamba-ssm-theory/)
- 🔗 [Join the Discussion on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7431122409711374336/?originTrackingId=wQPcaqKdn4pVi9A7h0mXGg%3D%3D)

**Part 2: Pure PyTorch Implementation**
- 🔗 [Read the Full Blog Post](https://soveshmohapatra.com/projects/mamba-ssm-live-code/)
- 🔗 [Join the Discussion on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7431498487692713984/?originTrackingId=FA1vwkjcBX3jfEk9C%2Bq9aw%3D%3D)

Check out my [LinkedIn](https://www.linkedin.com/in/sovesh) to follow along with the live updates for future parts!
