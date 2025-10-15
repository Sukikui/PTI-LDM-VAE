import json
import os
import random
from glob import glob

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import tifffile
from monai.config import print_config
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, Resize
from sklearn.decomposition import PCA
from utils_tif_no_augment import LocalNormalizeByMask, define_instance

# Custom TIF reader transform
class TifReader:
    """Custom transform to read TIF files using the tifffile library."""
    def __call__(self, path: str) -> np.ndarray:
        """Load a TIF file and return it as a numpy array."""
        img = tifffile.imread(path)
        return img.astype(np.float32)

# === SEED POUR REPRODUCTIBILITÉ ===
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# === MODES DE VISU
couleur_par_examen = True  # True = couleurs par examen, False = bleu/orange

# === CONFIGURATION ===
script_dir = os.path.dirname(os.path.abspath(__file__))
vae_weights = os.path.join(script_dir, "../data/weights/autoencoder_epoch73.pth")
config_file = os.path.join(script_dir, "../data/config/config_train_16g_cond.json")
folder_edentee = os.path.join(script_dir, "../data/edente/")
folder_dentee = os.path.join(script_dir, "../data/dente/")
folder_output = os.path.join(script_dir, "../results/umap_old_script_output")
save_path = f"{folder_output}/umap_latent_projection_comparatif.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_images = 2000
patch_size = (256, 256)

os.makedirs(folder_output, exist_ok=True)

# === Chargement du modèle
print("⏳ Chargement du modèle VAE...")
print_config()
args = type("args", (object,), {})()

# Load and merge both config files
env_file = os.path.join(script_dir, "../data/config/environment_tif.json")
with open(config_file) as f:
    config_dict = json.load(f)
with open(env_file) as f:
    env_dict = json.load(f)["vae"] # Use the 'vae' section

for k, v in config_dict.items():
    setattr(args, k, v)
for k, v in env_dict.items():
    setattr(args, k, v)

vae = define_instance(args, "autoencoder_def").to(device)
vae.load_state_dict(torch.load(vae_weights, map_location=device))
vae.eval()
print("✅ Modèle VAE chargé.")

# === Prétraitement
transforms = Compose(
    [
        TifReader(),
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(patch_size),
        LocalNormalizeByMask(),
        EnsureType(dtype=torch.float32),
    ]
)

# === Chargement édentées
paths_edentee = sorted(glob(os.path.join(folder_edentee, "*.tif")))[:max_images]
exams_edentee = [os.path.basename(p).split("_", 1)[1] for p in paths_edentee]
unique_exams = sorted(set(exams_edentee))
exam_to_id = {exam: i for i, exam in enumerate(unique_exams)}

# Corrected cmap call
if len(unique_exams) > 0:
    colormap = cm.get_cmap("tab20", len(unique_exams))
    exam_to_color = {exam: colormap(i) for i, exam in enumerate(unique_exams)}
else:
    exam_to_color = {}

if couleur_par_examen:
    with open(os.path.join(folder_output, "legende_couleurs.txt"), "w") as f:
        for exam in unique_exams:
            color_hex = mcolors.to_hex(exam_to_color[exam])
            idx = exam_to_id[exam]
            f.write(f"{idx} : {exam} — {color_hex}\n")

# === Encodage édentées
print(f"⏳ Encodage de {len(paths_edentee)} images édentées...")
Z_edentee, colors_edentee, ids_edentee = [], [], []
with torch.no_grad():
    for i, path in enumerate(paths_edentee):
        if (i + 1) % 50 == 0:
            print(f"  - Image édentée {i+1}/{len(paths_edentee)}")
        img = transforms(path).unsqueeze(0).to(device)
        z = vae.encode_stage_2_inputs(img).cpu().flatten(start_dim=1)
        Z_edentee.append(z)
        exam = os.path.basename(path).split("_", 1)[1]
        color = exam_to_color.get(exam, "#cccccc") if couleur_par_examen else "tab:blue"
        colors_edentee.append(color)
        ids_edentee.append(exam)

if Z_edentee:
    Z_edentee = torch.cat(Z_edentee, dim=0).numpy()
    print(f"🔹 {len(Z_edentee)} vecteurs édentées encodés.")
else:
    Z_edentee = np.array([])
    print("🔹 No edentee vectors encoded.")


# === Chargement dentées
paths_dentee = sorted(glob(os.path.join(folder_dentee, "*.tif")))[:max_images]
print(f"⏳ Encodage de {len(paths_dentee)} images dentées...")
Z_dentee, colors_dentee, ids_dentee_group2 = [], [], []
with torch.no_grad():
    for i, path in enumerate(paths_dentee):
        if (i + 1) % 50 == 0:
            print(f"  - Image dentée {i+1}/{len(paths_dentee)}")
        img = transforms(path).unsqueeze(0).to(device)
        z = vae.encode_stage_2_inputs(img).cpu().flatten(start_dim=1)
        Z_dentee.append(z)
        exam = os.path.basename(path).split("_", 1)[1]
        color = exam_to_color.get(exam, "#cccccc") if couleur_par_examen else "tab:red"
        colors_dentee.append(color)
        ids_dentee_group2.append(exam)

if Z_dentee:
    Z_dentee = torch.cat(Z_dentee, dim=0).numpy()
    print(f"🔹 {len(Z_dentee)} vecteurs dentées encodés.")
else:
    Z_dentee = np.array([])
    print("🔹 No dentee vectors encoded.")


if len(Z_edentee) > 0:
    # === PCA sur édentées uniquement
    print("⏳ Calcul de la PCA...")
    pca = PCA(n_components=50).fit(Z_edentee)
    Z_edentee_pca = pca.transform(Z_edentee)
    if len(Z_dentee) > 0:
        Z_dentee_pca = pca.transform(Z_dentee)
    print("✅ PCA calculée.")

    # === UMAP : fit sur édentées, transform sur dentées
    print("⏳ Projection UMAP...")
    umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(40, len(Z_edentee_pca) - 1), min_dist=0.5).fit(Z_edentee_pca)
    Z_umap_edentee = umap_model.embedding_
    if len(Z_dentee) > 0:
        Z_umap_dentee = umap_model.transform(Z_dentee_pca)
    else:
        Z_umap_dentee = np.array([])
    print("✅ Projection UMAP terminée.")

    # === Affichage
    print("⏳ Génération du graphique...")
    plt.figure(figsize=(10, 8))

    for i, (x, y) in enumerate(Z_umap_edentee):
        color = colors_edentee[i]
        plt.scatter(x, y, s=30, color=color, marker="o", alpha=0.6)

    # Marqueurs dentées
    if len(Z_umap_dentee) > 0:
        for i, (x, y) in enumerate(Z_umap_dentee):
            exam = ids_dentee_group2[i]
            color = colors_dentee[i]
            plt.scatter(x, y, s=30, color=color, marker="^", alpha=0.6)

    # === Ajouter un seul label centré par examen, en noir
    for exam in unique_exams:
        idx = exam_to_id[exam]

        # Moyenne des édentées (rond '◯' car marker='o')
        ed_pts = [Z_umap_edentee[i] for i in range(len(ids_edentee)) if ids_edentee[i] == exam]
        if ed_pts:
            ed_mean = np.mean(ed_pts, axis=0)
            plt.text(
                ed_mean[0],
                ed_mean[1],
                f"◯{idx}",
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
                color="black",
                alpha=0.9,
            )

        # Moyenne des dentées (triangle '△' car marker='^')
        if len(Z_umap_dentee) > 0:
            de_pts = [Z_umap_dentee[i] for i in range(len(ids_dentee_group2)) if ids_dentee_group2[i] == exam]
            if de_pts:
                de_mean = np.mean(de_pts, axis=0)
                plt.text(
                    de_mean[0],
                    de_mean[1],
                    f"△{idx}",
                    fontsize=9,
                    weight="bold",
                    ha="center",
                    va="center",
                    color="black",
                    alpha=0.9,
                )


    # === Titre
    titre = "UMAP : édentées (o), dentées (^)"
    titre += " — couleurs par examen" if couleur_par_examen else " — couleurs globales"
    plt.title(titre)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    # plt.show()

    print(f"✅ Graphe enregistré dans : {save_path}")


    from collections import defaultdict

    from scipy.spatial.distance import cdist

    # === Regroupement des données par examen
    exam_data = defaultdict(lambda: {"edentee": [], "dentee": []})
    exam_data_umap = defaultdict(lambda: {"edentee": [], "dentee": []})

    for i, exam in enumerate(ids_edentee):
        exam_data[exam]["edentee"].append(Z_edentee[i])
        exam_data_umap[exam]["edentee"].append(Z_umap_edentee[i])

    if len(Z_dentee) > 0:
        for i, exam in enumerate(ids_dentee_group2):
            exam_data[exam]["dentee"].append(Z_dentee[i])
            exam_data_umap[exam]["dentee"].append(Z_umap_dentee[i])


    # === Fonction de calcul des métriques
    def compute_metrics(points1, points2):
        if len(points1) == 0 or len(points2) == 0:
            return None
        points1 = np.array(points1)
        points2 = np.array(points2)
        mean1 = np.mean(points1, axis=0)
        mean2 = np.mean(points2, axis=0)
        dist_centres = np.linalg.norm(mean1 - mean2)
        std1 = np.mean(np.std(points1, axis=0)) if len(points1) > 1 else 0.0
        std2 = np.mean(np.std(points2, axis=0)) if len(points2) > 1 else 0.0
        all_dists = cdist(points1, points2)
        cross_dist_mean = np.mean(all_dists)
        return dist_centres, std1, std2, cross_dist_mean


    # === Calcul des métriques et sauvegarde
    print("⏳ Calcul des métriques...")
    path_resultats = os.path.join(folder_output, "distances_par_examen_umap.txt")
    with open(path_resultats, "w") as f:
        f.write("Analyse par examen : distances dans l’espace latent et UMAP\n")
        f.write("=" * 60 + "\n\n")

        for exam in sorted(exam_data.keys()):
            lat_ed = exam_data[exam]["edentee"]
            lat_de = exam_data[exam]["dentee"]
            umap_ed = exam_data_umap[exam]["edentee"]
            umap_de = exam_data_umap[exam]["dentee"]

            if len(lat_ed) == 0 or len(lat_de) == 0:
                continue

            metrics_lat = compute_metrics(lat_ed, lat_de)
            metrics_umap = compute_metrics(umap_ed, umap_de)

            if not metrics_lat or not metrics_umap:
                continue

            idx = exam_to_id[exam]
            f.write(f"[{idx}] {exam}\n")
            f.write(f"  - nb_edentee   : {len(lat_ed)}\n")
            f.write(f"  - nb_dentee    : {len(lat_de)}\n")
            f.write(f"  - [Latent] distance_centres           : {metrics_lat[0]:.3f}\n")
            f.write(f"  - [Latent] std_edentee                : {metrics_lat[1]:.3f}\n")
            f.write(f"  - [Latent] std_dentee                 : {metrics_lat[2]:.3f}\n")
            f.write(f"  - [Latent] moyenne_distances_croisées : {metrics_lat[3]:.3f}\n")
            f.write(f"  - [UMAP  ] distance_centres           : {metrics_umap[0]:.3f}\n")
            f.write(f"  - [UMAP  ] std_edentee                : {metrics_umap[1]:.3f}\n")
            f.write(f"  - [UMAP  ] std_dentee                 : {metrics_umap[2]:.3f}\n")
            f.write(f"  - [UMAP  ] moyenne_distances_croisées : {metrics_umap[3]:.3f}\n\n")

    print(f"✅ Fichier '{path_resultats}' généré.")

    # === Tri des examens selon la distance dans l’espace latent
    print("⏳ Tri des examens par distance...")
    summary = []
    for exam in sorted(exam_data.keys()):
        lat_ed = exam_data[exam]["edentee"]
        lat_de = exam_data[exam]["dentee"]
        if len(lat_ed) == 0 or len(lat_de) == 0:
            continue
        metrics_lat = compute_metrics(lat_ed, lat_de)
        if not metrics_lat:
            continue
        summary.append((exam, metrics_lat[0]))

    summary.sort(key=lambda x: x[1])
    path_tri = os.path.join(folder_output, "examen_tries_par_distance_latente_umap.txt")
    with open(path_tri, "w") as f:
        f.write("Examens triés par distance centre dentée/édentée (espace latent)\n")
        f.write("=" * 60 + "\n\n")
        for exam, dist in summary:
            idx = exam_to_id[exam]
            f.write(f"[{idx}] {exam} : {dist:.3f}\n")


    print(f"📄 Fichier '{path_tri}' généré.")
    print("✅ Script terminé.")

else:
    print("No data found for group 1. Aborting.")
