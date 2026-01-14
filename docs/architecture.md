## Architecture

This repository implements a latent generative pipeline to predict **edentulous sagittal CBCT slices** from **dentate inputs**. The method is structured into three blocks:

______________________________________________________________________

### A) VAE + Regression Head (latent representation + metric inference)

We first train a Variational Autoencoder (VAE) to obtain a compact representation of each (256\\times256) slice. The encoder maps an input slice (x) to a spatial latent tensor
\[
z \\in \\mathbb{R}^{4\\times 32\\times 32},
\]
and the decoder reconstructs (\\hat{x}). Following the ablation results reported in the paper, the VAE used for the diffusion stage is **VAE-both** (trained jointly on dentate + edentulous images **without adversarial loss**) because it provides the best reconstruction trade-off across both domains.

# The VAE is trained with a reconstruction + regularization objective: \[ \\mathcal{L}\_{\\mathrm{VAE}}

|x-\\hat{x}|_{1}
\+
\\lambda_{\\mathrm{KL}},D\_{\\mathrm{KL}}!\\left(q(z\\mid x),|,\\mathcal{N}(0,I)\\right)
\+
\\lambda\_{\\mathrm{perc}}
\\sum\_{\\ell \\in \\mathcal{S}}
\\big|\\psi\_{\\ell}(x)-\\psi\_{\\ell}(\\hat{x})\\big|_{2}^{2}.
\]
Here, the KL term enforces a unit-Gaussian prior over the latent distribution, while the perceptual term compares features (\\psi_\\ell(\\cdot)) extracted at selected layers (\\ell \\in \\mathcal{S}).

**Regression head.** Because the AR-VAE variant did not produce a stable latent structure in our setting, we instead train a **decoupled regression head** on top of a **frozen dentate encoder**. Given a dentate latent (z_0^\*), the regression head (Flatten + MLP with a 4096-unit hidden layer) predicts six edentulous geometric metrics:
\[
\\hat{a} ;=; [\\hat{h}\_0,\\hat{w}\_0,\\hat{w}\_1,\\hat{w}\_2,\\hat{w}\_3,\\hat{w}\_4] \\in \\mathbb{R}^{6}.
\]
These metrics act as geometric constraints used to condition the diffusion model.

**Where to place the figure (A).** Put the “VAE + Regression Head training loops” diagram here, right after introducing both components.

![VAE and Regression Head Training Loops](figures/vae_regression_training_loop.svg#gh-light-mode-only)
![VAE and Regression Head Training Loops](figures/vae_regression_training_loop_dark.svg#gh-dark-mode-only)

______________________________________________________________________

### B) LDM (partial diffusion initialization + conditioning via metrics)

We train a Latent Diffusion Model (LDM) operating in the latent space of the selected VAE (**VAE-both**). The LDM learns to generate an edentulous latent (\\hat{z}_0), which is decoded into an edentulous slice:
\[
\\hat{x}_{\\mathrm{ed}} = \\mathrm{Dec}(\\hat{z}\_0).
\]

#### B.1 Partial diffusion initialization (why and how)

A standard conditional diffusion setup would start reverse diffusion from pure Gaussian noise. In our experiments, this produced excessive stochasticity and did not preserve patient-specific **basal bone**. We therefore adopt a **partial diffusion** initialization: we start from the dentate latent (z_0^\*), apply a forward diffusion step (t) using spatially selective noise injection, and then denoise to obtain (\\hat{z}\_0).

Concretely, with (\\epsilon \\sim \\mathcal{N}(0,I)) and a mask (\\text{mask}\\in[0,1]^{H\\times W}) (low values in the basal region, higher values towards the crest), we form:
\[
\\hat{z}\_t ;=; \\sqrt{\\bar{\\alpha}\_t}, z_0^\* ;+; \\sqrt{1-\\bar{\\alpha}\_t},(\\epsilon \\odot \\text{mask}),
\]
where (\\bar{\\alpha}\_t) is provided by the diffusion scheduler. This encourages the model to modify primarily the high-noise ridge region while keeping low-noise basal structures consistent.

#### B.2 Conditioning mechanism (CondEnc + cross-attention)

# To guide denoising toward patient-specific geometry, the U-Net denoiser is conditioned on the six metrics (\\hat{a}). A Conditioning Encoder (CondEnc) maps (\\hat{a}) to an embedding (c\\in\\mathbb{R}^{512}) injected via cross-attention: \[ c = W_2 \\cdot \\mathrm{SiLU}\\big(W_1 \\cdot \\mathrm{Dropout}(\\hat{a})\\big). \] The denoiser is trained with the standard diffusion objective by predicting the injected noise: \[ \\mathcal{L}\_{\\mathrm{LDM}}

\\mathbb{E}_{t,\\epsilon}\\left\[\\left| \\epsilon - \\hat{\\epsilon}_\\theta(\\hat{z}\_t,\\hat{a},t)\\right|^2\\right\],
\]
where (\\theta) denotes **both** U-Net and CondEnc parameters.

**Where to place the figure (B).** Put the “full LDM training loop (partial diffusion + conditioning)” diagram here, after describing partial diffusion + CondEnc.

![LDM Training Loop (Partial Diffusion + Conditioning)](figures/ldm_training_loop.svg#gh-light-mode-only)
![LDM Training Loop (Partial Diffusion + Conditioning)](figures/ldm_training_loop_dark.svg#gh-dark-mode-only)


______________________________________________________________________

### C) Inputs / Outputs and what is frozen vs. trained

This section clarifies what flows through the pipeline and what is optimized at each stage.

#### C.1 Inputs and outputs

**Inputs.** The main input is the dentate image (x^\*\\in\\mathbb{R}^{256\\times256}). During training only, we also have the edentulous target image (x\_{\\mathrm{ed}}) and the ground-truth metrics (a\\in\\mathbb{R}^6) extracted from the edentulous mask. In the LDM stage and at inference time, conditioning uses the predicted metrics (\\hat{a}) produced by the regression head.

**Outputs.** The pipeline produces VAE reconstructions (\\hat{x}), predicted metrics (\\hat{a}), a predicted edentulous latent (\\hat{z}_0), and finally the predicted edentulous slice:
\[
\\hat{x}_{\\mathrm{ed}} = \\mathrm{Dec}(\\hat{z}\_0).
\]

#### C.2 What is trained vs. frozen (by stage)

**Stage 1 — Train the VAE.** The VAE encoder and decoder are trained end-to-end (VAE-both is selected for the final pipeline).

**Stage 2 — Train the regression head.** The VAE encoder is frozen (dentate encoder used to produce (z_0^\*)), and the regression head (Flatten + MLP) is trained to predict (\\hat{a}).

**Stage 3 — Train the LDM.** The VAE encoder and decoder are frozen (the latent space is fixed once the VAE is selected). The regression head is typically frozen as well and used only to provide (\\hat{a}) for conditioning. The U-Net denoiser and CondEnc are trained using (\\mathcal{L}\_{\\mathrm{LDM}}).
