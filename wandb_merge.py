import wandb
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# ========== USER SETTINGS ==========
ENTITY = "luca24ever_unibo"
PROJECT = "ML4CV_Assignment"
RUN1_ID = "xppyrixn"  # triplet_metric_fixed
RUN2_ID = "dqwnszi5"  # triplet_metric_fixed
MERGED_NAME = "triplet_metric_fixed_merged_4"
DOWNLOAD_DIR = "downloads"  # where to cache media
MAX_WORKERS = 8  # number of parallel downloads
HISTORY_SAMPLES = 1000  # adjust as needed
# ===================================

api = wandb.Api()

# Fetch runs
run1: wandb.Run = api.run(f"{ENTITY}/{PROJECT}/{RUN1_ID}")
run2: wandb.Run = api.run(f"{ENTITY}/{PROJECT}/{RUN2_ID}")

# ===== Download numeric histories =====
hist1 = run1.history(samples=HISTORY_SAMPLES)
hist2 = run2.history(samples=HISTORY_SAMPLES)

# Offset run2 steps
offset = hist1["_step"].max() + 1
hist2["_step"] += offset

# Merge histories
merged_hist = pd.concat([hist1, hist2], ignore_index=True)

# ===== Copy metadata & config from run1 =====
config = dict(run1.config)
metadata = {
    "name": MERGED_NAME,
    "notes": run1.notes,
    "tags": run1.tags,
    "group": run1.group,
}


# ===== Helper functions for parallel media download =====
def parse_filename(path):
    filename = os.path.basename(path)
    parts = filename.split("_")
    try:
        step = int(parts[1])
        col = "_".join(parts[:1])
    except Exception:
        step = None
        col = filename
    return step, col


def download_file(f):
    if f.name.startswith("media/") and not f.name.endswith("_thumb.png"):
        # print(f.name)
        print(os.path.join(DOWNLOAD_DIR, "media", f.name.replace("/", "\\")))
        if os.path.exists(
            os.path.join(DOWNLOAD_DIR, "media", f.name.replace("/", "\\"))
        ):
            print(f"File {f.name} already exists, skipping download.")
            local_path = os.path.join(DOWNLOAD_DIR, "media", f.name)
            step, col = parse_filename(local_path)
            return step, col, local_path
        local_path = f.download(root=DOWNLOAD_DIR, exist_ok=True).name
        step, col = parse_filename(local_path)
        return step, col, local_path
    return None


def download_media_parallel(run):
    media_map = {}
    files = list(run.files())
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_file, f) for f in files]
        with tqdm(total=len(futures), desc=f"Downloading media from {run.id}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    step, col, path = result
                    media_map.setdefault(step, []).append((col, path))
                pbar.update(1)
    return media_map


# ===== Download media from both runs =====
media1 = download_media_parallel(run1)
media2 = download_media_parallel(run2)

# Offset run2 steps in media2
media2 = {
    step + offset if step is not None else None: files for step, files in media2.items()
}

# Merge media maps
merged_media = {**media1}
for step, files in media2.items():
    merged_media.setdefault(step, []).extend(files)

# ===== Start new merged run =====
wandb_run = wandb.init(
    entity=ENTITY,
    project=PROJECT,
    name=metadata["name"],
    config=config,
    notes=metadata["notes"],
    tags=metadata["tags"],
    group=metadata["group"],
)

# ===== Re-log metrics + images =====
for _, row in tqdm(
    merged_hist.iterrows(), total=len(merged_hist), desc="Logging metrics"
):
    metrics = {}
    step = int(row["_step"])

    # numeric metrics
    for col, val in row.items():
        if pd.isna(val):
            continue
        if isinstance(val, (int, float, str)):
            metrics[col] = val

    # images/plots (if any at this step)
    if step in merged_media:
        for col, path in merged_media[step]:
            metrics[col] = wandb.Image(path)

    if metrics:
        wandb.log(metrics, step=step)


# ===== Attach artifacts from both runs =====
def attach_artifacts(src_run, target_run):
    for artifact in src_run.logged_artifacts():
        print(f"Attaching artifact {artifact.name}:{artifact.version}")
        target_run.use_artifact(artifact)


attach_artifacts(run1, wandb_run)
attach_artifacts(run2, wandb_run)

# ===== Finish merged run =====
wandb_run.finish()

print("✅ Merged run created successfully with metrics + images!")
