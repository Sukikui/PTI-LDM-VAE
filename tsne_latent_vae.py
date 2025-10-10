import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from monai.config import print_config
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, EnsureType
from utils_tif_no_augment import LocalNormalizeByMask, define_instance
import json
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === MODES DE VISU
couleur_par_examen = False  # True = couleurs par examen, False = bleu/orange

# === CONFIGURATION ===
vae_weights = "12_06_2025_vae_pour_ldm_edente_edente/trained_weights/autoencoder_epoch73.pth"
config_file = "config/config_train_16g_cond.json"
folder_edentee = "data_cs_1_dm_encastre_tif_32_bits_22_05_2025/edente"
folder_dentee = "data_cs_1_dm_encastre_tif_32_bits_22_05_2025/dente"
folder_output = "tsne_comparatif_edentee_dentee_fake"
save_path = f"{folder_output}/tsne_latent_projection_comparatif.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_images = 1000
patch_size = (256, 256)

os.makedirs(folder_output, exist_ok=True)

# === Chargement du modèle
print_config()
args = type('args', (object,), {})()
with open(config_file) as f:
    config_dict = json.load(f)
for k, v in config_dict.items():
    setattr(args, k, v)
vae = define_instance(args, "autoencoder_def").to(device)
vae.load_state_dict(torch.load(vae_weights, map_location=device))
vae.eval()

# === Prétraitement
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize(patch_size),
    LocalNormalizeByMask(),
    EnsureType(dtype=torch.float32)
])

# === Chargement édentées
paths_edentee = sorted(glob(os.path.join(folder_edentee, "*.tif")))[:max_images]
exams_edentee = [os.path.basename(p).split("_", 1)[1] for p in paths_edentee]
unique_exams = sorted(set(exams_edentee))
colormap = cm.get_cmap('tab20', len(unique_exams))
exam_to_color = {exam: colormap(i) for i, exam in enumerate(unique_exams)}

if couleur_par_examen:
    with open(os.path.join(folder_output, "legende_couleurs.txt"), "w") as f:
        for exam, color in exam_to_color.items():
            f.write(f"{exam} : {mcolors.to_hex(color)}\n")

# === Encodage édentées
Z_edentee, colors_edentee, ids_edentee = [], [], []
with torch.no_grad():
    for path in paths_edentee:
        img = transforms(path).unsqueeze(0).to(device)
        z = vae.encode_stage_2_inputs(img).cpu().flatten(start_dim=1)
        Z_edentee.append(z)
        exam = os.path.basename(path).split("_", 1)[1]
        color = exam_to_color[exam] if couleur_par_examen else 'tab:blue'
        colors_edentee.append(color)
        ids_edentee.append(exam)

Z_edentee = torch.cat(Z_edentee, dim=0).numpy()
print(f"🔹 {len(Z_edentee)} vecteurs édentées encodés.")

# === Chargement dentées
paths_dentee = sorted(glob(os.path.join(folder_dentee, "*.tif")))[:max_images]
Z_dentee, colors_dentee, ids_dentee = [], [], []
with torch.no_grad():
    for path in paths_dentee:
        img = transforms(path).unsqueeze(0).to(device)
        z = vae.encode_stage_2_inputs(img).cpu().flatten(start_dim=1)
        Z_dentee.append(z)
        exam = os.path.basename(path).split("_", 1)[1]
        color = exam_to_color.get(exam, "#cccccc") if couleur_par_examen else 'tab:red'
        colors_dentee.append(color)
        ids_dentee.append(exam)

Z_dentee = torch.cat(Z_dentee, dim=0).numpy()
print(f"🔹 {len(Z_dentee)} vecteurs dentées encodés.")



# === PCA + t-SNE combiné
print("⏳ PCA + t-SNE...")
pca = PCA(n_components=50).fit(Z_edentee)
Z_edentee_pca = pca.transform(Z_edentee)
Z_dentee_pca = pca.transform(Z_dentee)
Z_combined = [Z_edentee_pca, Z_dentee_pca]

Z_tsne_input = np.concatenate(Z_combined)
Z_tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(Z_tsne_input)

# === Découpage
Z_tsne_edentee = Z_tsne[:len(Z_edentee)]
Z_tsne_dentee = Z_tsne[len(Z_edentee):len(Z_edentee)+len(Z_dentee)]

# === Affichage
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(Z_tsne_edentee):
    plt.scatter(x, y, s=30, color=colors_edentee[i], marker='o', alpha=0.6)
for i, (x, y) in enumerate(Z_tsne_dentee):
    plt.scatter(x, y, s=30, color=colors_dentee[i], marker='^', alpha=0.6)

# === Titre
titre = "t-SNE : édentées (o), dentées (^)"
titre += " — couleurs par examen" if couleur_par_examen else " — couleurs globales"
plt.title(titre)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Graphe enregistré dans : {save_path}")
