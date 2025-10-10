VAE_LDM — README


DESCRIPTION
Pipeline pour :

Entraîner un VAE sur images TIF float32 (édentée → édentée)

Entraîner un LDM conditionné (dentée → édentée) en s’appuyant sur le VAE

Faire de l’inférence (VAE) -> Inférence pour le LDM à faire, se baser sur le scrit du VAE

Calculer des métriques sur les résultats du LDM (PSNR, SSIM, Dice, IoU, métriques géométriques)

Visualiser l’espace latent (UMAP / t-SNE)

Les images sont des TIF float32 1 canal (256×256). Le LDM est conditionné par l’image dentée encodée par le VAE et génère une image édentée.

DOCUMENT DE SYNTHÈSE

Un document de synthèse (Document_de_synthese.pdf) résume la méthodologie, les configurations, les données, ainsi que les résultats d’entraînement et d’inférence 
(métriques, graphiques, UMAP/TSNE) et les recommandations de reproductibilité.

Ne pas hésiter à me contacter en cas de questions : tv.pham1996@gmail.com (Tuong Vy PHAM)

ARBORESCENCE DU DOSSIER

VAE_LDM/
	config/
		config_train_16g_cond.json (config entraînement LDM conditionné)
		environment_tif.json (chemins/env pour TIF)
train_autoencoder_tif.py (entraînement VAE)
train_diffusion_tif_cond.py (entraînement LDM conditionné)
inference_vae_tif.py (inférence VAE)
compute_metrics_class_tif.py (métriques + distributions)
visualize_image.py (prévisualisations simples)
umap_latent_vae.py (UMAP du latent VAE)
tsne_latent_vae.py (t-SNE du latent VAE)
utils_tif.py (dataset/augmentations/IO TIF)
utils_tif_no_augment.py (variante sans augmentations)
requirements.txt (dépendances Python)
data_cs_1_dm_encastre_tif_32_bits_22_05_2025.zip (données entraînement)
data_cs_1_dm_encastre_tif_04_06_2025_lot_inference_res_150.zip (données inférence)
12_06_2025_vae_pour_ldm_edente_edente.zip (poids VAE édentée/édentée)
Document_de_synthese.pdf

INSTALLATION

Python recommandé : 3.10

Créer et activer un environnement virtuel
python -m venv .venv
(Linux/macOS) source .venv/bin/activate
(Windows) .venv\Scripts\activate

Installer les dépendances Python
pip install -r requirements.txt

Installer PyTorch / torchvision selon votre OS et CUDA
Voir la commande officielle : https://pytorch.org/get-started/locally/

Notes :

TensorBoard recommandé pour le suivi de l’entraînement.

DONNÉES ET POIDS

Dézipper les jeux de données et les poids aux emplacements souhaités, puis adapter les chemins dans config/environment_tif.json si besoin.

Entraînement VAE :
data_cs_1_dm_encastre_tif_32_bits_22_05_2025.zip → ex. data/train/…

Inférence :
data_cs_1_dm_encastre_tif_04_06_2025_lot_inference_res_150.zip → ex. data/inference/…

Poids VAE édentée/édentée :
12_06_2025_vae_pour_ldm_edente_edente.zip 

DÉMARRAGE RAPIDE

Entraîner le VAE (édentée → édentée)
python train_autoencoder_tif.py --config config/environment_tif.json
Sorties : trained_weights/autoencoder/*.pth, tfevent/, validation_samples/

Entraîner le LDM conditionné (dentée → édentée)
python train_diffusion_tif_cond.py --config config/config_train_16g_cond.json --env config/environment_tif.json
Le script charge le VAE édentée/édentée pour encoder l’input/condition.

Sorties typiques (par epoch) :
<run_dir>/validation_samples/epoch_<N>/edente/ (GT)
<run_dir>/validation_samples/epoch_<N>/edente_synth/ (prédictions)
step000.tif, step001.tif, …

Inférence VAE seule
python inference_vae_tif.py --weights trained_weights/vae/xxx.pth --input data/inference/ --output results/vae_recon/

Calcul des métriques
Le script attend des paires GT/Pred ayant le même nom dans deux dossiers frères :
…/epoch_<N>/edente/stepNNN.tif (GT)
…/epoch_<N>/edente_synth/stepNNN.tif (Pred)
Paramétrage conseillé dans compute_metrics_class_tif.py (section main, à la fin du script) :
folder_path = "<run_dir>"
num_epoch = "<N>"
folder_path_validation = f"{folder_path}/validation_samples/epoch_{num_epoch}"
Appel recommandé pour éviter les doublons :
Sorties : _metrics.csv, _dimensions.csv, _metrics_distribution.png dans le dossier visé.

Visualisation du latent VAE
UMAP : python umap_latent_vae.py --weights trained_weights/vae/xxx.pth --data data/inference/ --out results/umap/
t-SNE : python tsne_latent_vae.py --weights trained_weights/vae/xxx.pth --data data/inference/ --out results/tsne/

CONFIGURATION

config/config_train_16g_cond.json : hyperparamètres LDM (réseau, pas, epochs, batch size, sorties).

config/environment_tif.json : chemins datasets/poids, options I/O TIF, normalisations, etc.
Si un script n’a pas d’arguments CLI, régler les variables en tête de fichier (chemins/epochs…).

REQUIREMENTS

albumentations==2.0.8
matplotlib==3.10.6
monai==1.5.1
numpy==2.3.3
opencv_python==4.10.0.84
opencv_python_headless==4.10.0.84
pandas==2.3.2
Pillow==11.3.0
PyYAML==6.0
PyYAML==6.0.3
scikit_learn==1.7.2
scipy==1.16.2
skimage==0.0
tensorboard==2.18.0
tensorflow==2.20.0
tifffile==2024.9.20
torch==2.5.1
torchsummary==1.5.1
torchvision==0.20.1
tqdm==4.67.1
umap==0.1.1


