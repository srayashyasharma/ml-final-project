import os, pandas as pd, numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

def get_transforms(split="train"):
    norm = transforms.Normalize([0.5],[0.5])
    if split == "train":
        return transforms.Compose([
            transforms.Resize((244,244)), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2,contrast=0.2),
            transforms.ToTensor(), norm])
    return transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm])

_IDX = {}
def build_image_index(data_dir):
    print(f"Scanning {data_dir}...", flush=True)
    idx = {}
    for root,_,fnames in os.walk(data_dir):
        for f in fnames:
            if f.lower().endswith((".png",".jpg")) and f not in idx:
                idx[f] = os.path.join(root,f)
    print(f"Found {len(idx):,} images.", flush=True)
    return idx

def get_image_index(data_dir):
    if data_dir not in _IDX:
        _IDX[data_dir] = build_image_index(data_dir)
    return _IDX[data_dir]

class NIHChestXRay(Dataset):
    def __init__(self, data_dir, labels_csv, split="train", transform=None, max_per_class=None):
        self.transform = transform or get_transforms(split)
        self.img_index = get_image_index(data_dir)
        df = pd.read_csv(labels_csv)
        df["binary_label"] = (~df["Finding Labels"].str.contains("No Finding")).astype(int)
        patients = df["Patient ID"].unique()
        rng = np.random.RandomState(42); rng.shuffle(patients)
        n = len(patients)
        cuts = {"train":set(patients[:int(.70*n)]),
                "val"  :set(patients[int(.70*n):int(.85*n)]),
                "test" :set(patients[int(.85*n):])}
        df = df[df["Patient ID"].isin(cuts[split])].reset_index(drop=True)
        df = df[df["Image Index"].isin(self.img_index)].reset_index(drop=True)
        if max_per_class:
            h = df[df.binary_label==0].sample(min(max_per_class,(df.binary_label==0).sum()),random_state=42)
            d = df[df.binary_label==1].sample(min(max_per_class,(df.binary_label==1).sum()),random_state=42)
            df = pd.concat([h,d]).sample(frac=1,random_state=42).reset_index(drop=True)
        self.df = df
        c = df["binary_label"].value_counts().to_dict()
        print(f"  {split:5s}: {len(df):,}  healthy={c.get(0,0):,}  diseased={c.get(1,0):,}", flush=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_index[row["Image Index"]]).convert("L")
        return self.transform(img), int(row["binary_label"])
    def balanced_sampler(self):
        c = self.df["binary_label"].value_counts().to_dict()
        w = [1.0/c[l] for l in self.df["binary_label"].values]
        return WeightedRandomSampler(w, num_samples=len(w), replacement=True)

def get_dataloaders(data_dir, labels_csv, batch_size=32, num_workers=2, max_per_class=None):
    pin = torch.cuda.is_available()
    get_image_index(data_dir)
    train_ds = NIHChestXRay(data_dir,labels_csv,"train",max_per_class=max_per_class)
    val_ds   = NIHChestXRay(data_dir,labels_csv,"val",  max_per_class=max_per_class)
    test_ds  = NIHChestXRay(data_dir,labels_csv,"test", max_per_class=max_per_class)
    return (
        DataLoader(train_ds,batch_size,sampler=train_ds.balanced_sampler(),num_workers=num_workers,pin_memory=pin),
        DataLoader(val_ds,  batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin),
        DataLoader(test_ds, batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin))
