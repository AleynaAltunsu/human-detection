# Gerekli kütüphanelerin içe aktarılması
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from glob import glob
from PIL import Image
import random
import cv2
from matplotlib import pyplot as plt

# Özel bir veri kümesi sınıfı tanımlanması
class CustomDataset(Dataset):
    def __init__(self, root, data, masks, transform=None):
        # Veri ve maske dosya yollarının ayarlanması
        self.transform = transform
        self.im_paths = sorted(glob(f"{root}/{data}/*"))
        self.im_masks = sorted(glob(f"{root}/{masks}/*"))

        # Veri ve maske dosya sayılarının eşit olup olmadığını kontrol etme
        assert len(self.im_paths) == len(self.im_masks)
        
    def __len__(self):
        # Veri kümesinin uzunluğunu döndürme
        return len(self.im_paths)
    
    def __getitem__(self, idx):
        # Belirtilen indeksteki görüntü ve maske dosya yollarını alma
        im_path = self.im_paths[idx]
        mask_path = self.im_masks[idx]

        # Görüntüyü RGB formatında açma
        im = Image.open(im_path).convert("RGB")
        # Maskeyi grayscale formatında açma
        gt = np.array(Image.open(mask_path).convert("L"))

        # Maskedeki benzersiz nesne indekslerini bulma (arka plan hariç)
        obj_indexes = np.unique(gt)[1:]
        # Maskeleri tek tek boolean dizilere çevirme
        gts = gt == obj_indexes[:, None, None]
        obj_numbers = len(obj_indexes)

        # Bounding box'ları depolamak için bir liste oluşturma
        boxes = []
        for i, box in enumerate(gts):
            # Her bir nesne için bounding box koordinatlarını hesaplama
            pos = np.where(box)
            x1 = np.min(pos[1])
            x2 = np.max(pos[1])
            y1 = np.min(pos[0])
            y2 = np.max(pos[0])
            boxes.append([x1, y1, x2, y2])

        # Bounding box'ları ve diğer hedef bilgileri torch tensorlerine çevirme
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((obj_numbers), dtype=torch.int64)
        gts = torch.as_tensor(gts, dtype=torch.uint8)
        im_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((obj_numbers), dtype=torch.int64)

        # Hedef sözlüğünü oluşturma
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = gts
        target["image_id"] = im_id
        target["area"] = area
        target["iscrowd"] = is_crowd

        # Eğer bir transformasyon varsa, görüntüyü ona göre dönüştürme
        if self.transform is not None:
            im = self.transform(im)
        
        # Görüntüyü ve hedefi döndürme
        return im, target

# Veri yükleyicileri oluşturmak için bir fonksiyon tanımlanması
def get_dls(root, transformations, bs, split=[0.9, 0.1]):
    # Tam veri kümesini eğitim ve doğrulama veri kümelerine ayırma
    df = CustomDataset(root=root, data="train_images", masks="train_masks", transform=transformations)
    total_size = len(df)
    train_size = int(split[0] * total_size)
    val_size = total_size - train_size
    tr_ds, vl_ds = random_split(df, [train_size, val_size])

    # Eğitim ve doğrulama veri yükleyicilerini oluşturma
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=4)
    val_dl = DataLoader(vl_ds, batch_size=bs, shuffle=False, num_workers=4)
    
    return tr_dl, val_dl

# Görüntü dönüşümlerini tanımlama
tfs = T.Compose([T.ToTensor()])
root = "/kaggle/input/human-detection-dataset/PennFudanPed/"

# Eğitim ve doğrulama veri yükleyicilerini oluşturma
tr_dl, vl_dl = get_dls(root=root, transformations=tfs, bs=8)
print(len(tr_dl), len(vl_dl))

# Veri kümesini görselleştirmek için bir fonksiyon tanımlanması
def visualize(dataset, n_ims, rows, cmap=None):
    plt.figure(figsize=(20, 20))
    indices = [random.randint(0, len(dataset) - 1) for _ in range(n_ims)]

    for idx, index in enumerate(indices):
        # Belirtilen indeksteki görüntüyü ve hedefi alma
        img, target = dataset[index]
        orig_img = img.permute(1, 2, 0).numpy()
        orig_img = orig_img.copy()
        obj_count = 0

        # Bounding box'ları çizme
        for i, bbox in enumerate(target["boxes"]):
            x, y, w, h = [int(t.item()) for t in bbox]
            obj_count += 1
            cv2.rectangle(img=orig_img, pt1=(x, y), pt2=(w, h), color=(0, 255, 0), thickness=3)

        # Görüntüyü gösterme
        plt.subplot(rows, n_ims // rows, idx + 1)
        plt.imshow(orig_img)
        plt.title(f"Image has {obj_count} people.")
        plt.axis("off")

# Veri kümesini başlatma
dataset = CustomDataset(root=root, data="train_images", masks="train_masks", transform=tfs)
visualize(dataset=dataset, n_ims=16, rows=4)
