## Architecture

This repository implements a latent generative pipeline to predict **edentulous sagittal CBCT slices** from **dentate inputs**. The method is structured into three blocks:

---

### A) VAE + Regression Head (latent representation + metric inference)

We first train a **Variational Autoencoder (VAE)** to obtain a compact representation of each 256×256 sagittal slice. The encoder maps an input slice `x` to a **spatial latent tensor** `z` (4 channels at 32×32 resolution), and the decoder reconstructs the slice `x_hat`.

Following the ablation study, the VAE used for the diffusion stage is **VAE-both** (trained jointly on dentate + edentulous images **without adversarial loss**) because it provides the best reconstruction trade-off across both domains.

**VAE training objective (high-level).** The VAE is optimized with:
- a **reconstruction term** that enforces pixel-level fidelity between `x` and `x_hat`,
- a **KL regularization term** that keeps the latent distribution close to a unit Gaussian prior,
- a **perceptual term** that compares deep feature representations of `x` and `x_hat` (extracted from selected layers of a fixed feature network).

**Regression head.** Because the AR-VAE variant was not stable in our setting, we instead train a **decoupled regression head** on top of a **frozen dentate encoder**. Given a dentate latent `z0*`, the regression head (Flatten + MLP with a 4096-unit hidden layer) predicts six edentulous geometric metrics:

- `a_hat = [height_0, width_0, width_1, width_2, width_3, width_4]`

These predicted metrics are used as **geometric constraints** to condition the diffusion model.

**Where to place the figure (A).** Put the “VAE + Regression Head training loops” diagram here, right after introducing both components.

![VAE and Regression Head Training Loops](figures/vae_regression_training_loop.svg#gh-light-mode-only)
![VAE and Regression Head Training Loops](figures/vae_regression_training_loop_dark.svg#gh-dark-mode-only)

---

### B) LDM (partial diffusion initialization + conditioning via metrics)

We train a **Latent Diffusion Model (LDM)** operating in the latent space of the selected VAE (**VAE-both**). The LDM learns to generate an edentulous latent `z_hat0`, which is then decoded by the VAE decoder to obtain the final edentulous slice.

#### B.1 Partial diffusion initialization (why and how)

A standard conditional diffusion setup typically starts reverse diffusion from **pure Gaussian noise**. In our experiments, this produced excessive stochasticity and did not preserve patient-specific **basal bone**. We therefore adopt a **partial diffusion** initialization:

1. Encode the dentate input to get the latent `z0*`.
2. Run a **forward diffusion/noising** step up to a chosen diffusion time `t`, but inject noise **non-uniformly** using a **vertical gradient mask**:
   - low noise in the basal region (to preserve patient identity),
   - higher noise towards the ridge/crest region (to allow remodeling variability).
3. Run the **reverse denoising** process to obtain `z_hat0`.

This design encourages the model to modify primarily the high-noise ridge region while keeping low-noise basal structures consistent.

#### B.2 Conditioning mechanism (CondEnc + cross-attention)

To guide the denoising toward patient-specific geometry, the U-Net denoiser is conditioned on the six metrics `a_hat`.

- A **Conditioning Encoder (CondEnc)** transforms the 6D metrics vector into a higher-dimensional embedding.
- This embedding is injected into the U-Net via **cross-attention**, allowing the diffusion process to respect target geometry while still sampling plausible anatomy.

**LDM training objective (high-level).** The diffusion model is trained with the standard diffusion objective: the network learns to predict the noise that was injected during the forward process (given the noised latent, the conditioning metrics, and the diffusion timestep). The learnable parameters include **both** the U-Net and CondEnc.

**Where to place the figure (B).** Put the “full LDM training loop (partial diffusion + conditioning)” diagram here, after describing partial diffusion + CondEnc.

![LDM Training Loop (Partial Diffusion + Conditioning)](figures/ldm_training_loop.svg#gh-light-mode-only)
![LDM Training Loop (Partial Diffusion + Conditioning)](figures/ldm_training_loop_dark.svg#gh-dark-mode-only)

---

### C) Inputs / Outputs and what is frozen vs. trained

This section clarifies what flows through the pipeline and what is optimized at each stage.

#### C.1 Inputs and outputs

**Inputs**
- **Dentate image** `x*` (256×256)
- **Training only:** edentulous target image `x_ed`
- **Training only:** ground-truth metrics `a` (6 values) extracted from the edentulous mask
- **Inference / LDM stage:** predicted metrics `a_hat` produced by the regression head

**Outputs**
- **VAE reconstruction** `x_hat`
- **Predicted metrics** `a_hat`
- **Predicted edentulous latent** `z_hat0`
- **Predicted edentulous image** obtained by decoding `z_hat0` with the VAE decoder

#### C.2 What is trained vs. frozen (by stage)

**Stage 1 — Train the VAE**
- **Trained:** VAE encoder + decoder (end-to-end)
- **Selected for pipeline:** VAE-both (best joint-domain reconstruction trade-off)

**Stage 2 — Train the regression head**
- **Frozen:** VAE dentate encoder (used to produce `z0*`)
- **Trained:** regression head (Flatten + MLP) to predict `a_hat`

**Stage 3 — Train the LDM**
- **Frozen:** VAE encoder + decoder (latent space is fixed once VAE is selected)
- **Typically frozen:** regression head (used only to provide `a_hat` conditioning)
- **Trained:** U-Net denoiser + CondEnc (diffusion training)