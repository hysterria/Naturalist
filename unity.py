import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm

csv_files = [
    "observations-660868.csv",
    "observations-660877.csv",
    "observations-660903.csv",
    "observations-660882.csv",
    "observations-660885.csv",
    "observations-660887.csv"
]

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    df['label'] = df['common_name']
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

full_df.to_csv("full_dataset.csv", index=False)


base_folder = Path("data/images")
base_folder.mkdir(parents=True, exist_ok=True)

for _, row in tqdm(full_df.iterrows(), total=len(full_df)):
    url = row['image_url']
    label = row['label']

    if pd.isna(url) or url.strip() == "":
        continue 

    folder = base_folder / label
    folder.mkdir(parents=True, exist_ok=True)

    img_path = folder / f"{row['id']}.jpg"

    try:
        img_data = requests.get(url, timeout=10).content
        with open(img_path, 'wb') as f:
            f.write(img_data)
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")

print("Скачивание завершено")