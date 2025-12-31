## Goal

Implement the Latent Diffusion Model (LDM) part of the project with:

- A pretrained VAE that is frozen (encode/decode only, no training).
- A pretrained regression head (MLP) that is frozen, used to predict edentulous metrics.
- NO ground-truth metrics used as conditioning (conditioning uses ONLY metrics predicted by the frozen MLP).

We train a conditional diffusion model in VAE latent space to map:
dentate image x\* -> predicted metrics â = g(E(x\*)) and latent condition z0\* = E(x\*)
and generate the edentulous latent z0 = E(x) via denoising.
Final output image is x_hat = D(z0_hat).

## Notation

- Dentate image: x\* (pre-extraction)
- Edentulous image: x (post-extraction)
- Frozen VAE encoder/decoder: E(.), D(.)
- Frozen regression head: g(.) (MLP)
- Latents:
  z0\* = E(x\*) (dentate latent)
  z0 = E(x) (edentulous latent)
- Predicted metrics (conditioning vector):
  â = g(z0\*) (shape: [B, 6] typically: 1 height + 5 widths)
- Diffusion timestep: t ∈ {1..T}
- Noise: ε ~ N(0, I)
- Diffusion schedule: {βt}, αt = 1-βt, ᾱt = Π\_{s=1..t} αs
- Noisy latent:
  zt = sqrt(ᾱt) z0 + sqrt(1-ᾱt) ε
- Denoiser U-Net: εθ(zt, t, cond) predicts ε (or v depending on parameterization)

## High-level training objective

Train a conditional denoiser to predict the noise added to the edentulous latent z0.
Conditioning uses:

1. Dentate latent z0\* (spatial condition).
2. Predicted metrics â (vector condition) produced by frozen g.

Loss (ε-prediction):
L = E\_{(x\*,x), t, ε} [ || ε - εθ(zt, t, z0\*, â) ||^2 ]

## Pipeline overview (training step)

Inputs per sample: paired (x\*, x)

Step 0) Freeze pretrained modules

- Freeze VAE E and D: requires_grad = False, eval() mode.
- Freeze MLP g: requires_grad = False, eval() mode.
- Train only diffusion U-Net parameters θ.

Step 1) Encode to latent space (no grad)

- z0\* = E(x\*) # dentate latent, shape [B, C, H, W]
- z0 = E(x) # edentulous latent, shape [B, C, H, W]

(Important: use the same VAE for both encodings unless you have a strong reason not to.
If you use different VAEs, document and keep consistent decode target.)

Step 2) Compute conditioning metrics (no grad)

- â = g(z0\*) # shape [B, A] where A=6

No GT metrics used: do NOT load/compute a in the training loop.

Step 3) Sample diffusion timestep and noise

- t ~ Uniform({1..T}) (per-sample)
- ε ~ N(0, I) (same shape as z0)

Step 4) Forward diffusion (add noise to edentulous latent)

- zt = sqrt(ᾱt) * z0 + sqrt(1-ᾱt) * ε

Step 5) Prepare conditioning for U-Net
We use two conditioning paths:

(A) Spatial conditioning: z0\*

- Option A1 (simple): concatenate channels
  u_in = concat(zt, z0\*) along channel dim → shape [B, 2C, H, W]
- Option A2 (better): cross-attention
  Convert z0\* to tokens and use cross-attention in U-Net blocks.

(B) Vector conditioning: â
Use FiLM / AdaGN modulation inside ResBlocks:

- Compute an embedding e_a = MLP_embed(â) to match model dimension d.
- Compute time embedding e_t = TimeEmbed(t).
- Combine: e = e_t + e_a.
- In each ResBlock:
  u = GN(h)
  γ = Wγ(e), β = Wβ(e)
  h_mod = u * (1 + γ) + β
  This ensures â influences all layers consistently.

Minimal implementation to start:

- Use A1 (concat) for z0\* and FiLM for â.

Step 6) Predict noise and compute loss

- ε_hat = εθ(u_in, t_embed=e_t, cond_embed=e_a) # depends on your U-Net API
- L = MSE(ε, ε_hat)

Step 7) Backprop and update θ

- optimizer.step() for θ only

## Model architecture (recommended, practical)

1. Latent-space U-Net

- Input: zt (and z0\* if concatenated)
- Typical latent sizes: C=4, H=W=16 (from your refined VAE)
- Use a standard DDPM/Stable-Diffusion-like U-Net backbone in latent space.

2. Conditioning mechanisms

- Time embedding: sinusoidal + MLP (standard).
- Metrics embedding:
  â ∈ R^6 → Linear(6→d) → SiLU → Linear(d→d)
- Dentate latent condition:
  Start with channel concatenation: [zt; z0\*]
  (Optionally later migrate to cross-attention tokens.)

3. Output head

- Predict ε (same shape as z0)
- Alternative: predict v (velocity) for better stability; start with ε for simplicity.

4. Sampling

- Use DDPM or DDIM sampler in latent space.
- At the end, decode with the frozen VAE decoder:
  x_hat = D(z0_hat)

## Inference pipeline (generation)

Input: dentate image x\*
Output: generated edentulous image x_hat

Step 1) Encode dentate latent (no grad)

- z0\* = E(x\*)

Step 2) Predict metrics (no grad)

- â = g(z0\*)

Step 3) Initialize latent noise

- zT ~ N(0, I)

Step 4) Reverse diffusion for t = T..1
At each step:

- ε_hat = εθ( concat(zt, z0\*), t, â )
- Compute z\_{t-1} using your chosen sampler (DDPM or DDIM) and schedule.

Step 5) Decode final latent

- x_hat = D(z0_hat)

## Implementation details / gotchas

1. Normalization of â

- During regression-head training you likely normalized targets.
- Therefore â is in "normalized space". Keep it consistent:
  - Either keep â normalized and train diffusion with normalized â.
  - Or unnormalize â before embedding (requires storing mean/std used).
    Pick one and stay consistent. Easiest: keep normalized â.

2. Conditioning dropout (recommended)
   To avoid the model overfitting or ignoring some conditions:

- With small probability p (e.g., 0.1), replace â with 0-vector.
- With small probability q (e.g., 0.1), replace z0\* with 0-tensor or drop the concat branch.
  This improves robustness and allows “classifier-free guidance” style sampling if desired.

3. Freezing correctness

- Confirm E, D, g are in eval() to freeze BatchNorm/Dropout behavior.
- Wrap their forward passes with torch.no_grad().

4. Loss scaling / mixed precision

- Use AMP (fp16/bf16) for U-Net; keep schedule scalars in fp32.
- Use gradient clipping if instability appears.

5. Data pairing

- Ensure x\* and x are correctly paired per slice/patient.
- Keep the same preprocessing pipeline used for VAE training (resize, intensity scaling, etc.)

6. Latent distribution mismatch

- If VAE encoder outputs posterior mean/variance:
  - Prefer using the mean (deterministic) for diffusion training:
    z = μ
  - Or sample from posterior for augmentation:
    z = μ + σ ⊙ η
    Be consistent. Start with μ to reduce noise sources.

7. Where â acts in training
   â does NOT change the forward diffusion equation.
   It intervenes ONLY inside εθ via the conditioning path (FiLM/AdaGN or embedding injection).
   So the training equation remains:
   zt = sqrt(ᾱt) z0 + sqrt(1-ᾱt) ε
   ε_hat = εθ(zt, t, z0\*, â)
   L = ||ε - ε_hat||^2

8. Conditioning modules must be trained and checkpointed
   The metric embedder and condition builder are part of the LDM and should not be in no_grad.
   Save and load their weights alongside the UNet to keep train/inference conditioning consistent.

## Sanity checks (must pass)

1. Shapes:

- z0\*, z0: [B, C, H, W]
- â: [B, 6]
- concat input: [B, 2C, H, W]

2. Frozen modules:

- gradients w.r.t. VAE and reg head must be None.
- only U-Net params updated.

3. Overfit test:

- On a tiny subset (e.g., 16 samples), loss should decrease and samples should look less noisy over training.

4. Conditioning test:

- Fix x\* and change â (manually add ±kσ on some metric dims) and verify generation changes accordingly
  (only possible if you allow overriding â at inference).

## What you need from existing repo

- Paths to VAE checkpoint and regression-head checkpoint.
- The exact preprocessing used during VAE training.
- Latent scaling (if any) used by the VAE (some implementations scale latents).
- The metric normalization stats used during regression-head training (mean/std).
- The latent shape (C,H,W) for your chosen VAE.

## End result

A trained LDM that:

- Takes a dentate slice x\*,
- Computes z0\* = E(x\*) and â = g(z0\*),
- Samples an edentulous latent z0_hat conditioned on (z0\*, â),
- Outputs the final image x_hat = D(z0_hat).
