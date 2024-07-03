"""Model Mimarisi:
Faster R-CNN: Bu proje, torchvision.models.detection modülünden Faster R-CNN mimarisini kullanmaktadır. Faster R-CNN şu bileşenlerden oluşur:
Arka Plan Ağı: Tipik olarak giriş görüntüsünden özellik haritaları çıkarmak için önceden eğitilmiş bir CNN (örneğin ResNet).
Bölge Öneri Ağı (RPN): Görüntü içinde aday nesne bölgeleri (sınırlayıcı kutular) önerir.
RoI (İlgi Bölgesi) Havuzu / Hizalaması: Her aday bölge için sabit boyutlu özellik haritaları çıkarır.
Baş Ağı: Her RoI'yi sınıflandırır ve sınırlayıcı kutusunu iyileştirir."""


""" Bu projede, kod örneği üzerinden Faster R-CNN modeli özelleştirilmiş bir şekilde uygulanmaktadır. Projede kullanılan katmanlar ve yapılan özelleştirmeler şu şekildedir:

1. **Backbone Ağı:**
   - **Kullanımı:** ResNet gibi önceden eğitilmiş bir CNN kullanılmaktadır (`torchvision.models.detection.FasterRCNN` içinde tanımlanabilir).
   - **Görevi:** Giriş görüntüsünden özellik haritaları çıkararak nesne tespitinin temelini oluşturur.

2. **Bölge Öneri Ağı (RPN):**
   - **Kullanımı:** Faster R-CNN modeli içinde otomatik olarak yer alır ve aday nesne bölgeleri (sınırlayıcı kutular) önerir.
   - **Görevi:** Görüntüde potansiyel nesne bölgelerini tespit etmek ve bu bölgeler üzerinde daha detaylı analiz yapılabilmesini sağlamak.

3. **RoI Hizalama ve Havuzu:**
   - **Kullanımı:** RoI pooling veya RoI align yöntemleriyle aday nesne bölgelerinden sabit boyutlu özellik haritaları çıkarılır.
   - **Görevi:** Her aday bölgeyi özellik vektörlerine dönüştürerek sınıflandırma ve sınırlayıcı kutu iyileştirme işlemleri için hazırlar.

4. **Baş Ağı (Head Network):**
   - **Kullanımı:** Her RoI için sınıflandırma ve sınırlayıcı kutu regresyonu yapmak için kullanılır.
   - **Görevi:** Her bir aday bölge için nesne sınıfını tahmin etmek ve sınırlayıcı kutunun konumunu daha doğru hale getirmek.

5. **Özelleştirilmiş Veri Yükleyici (Custom Dataset ve DataLoader):**
   - **Kullanımı:** `CustomDataset` sınıfı ve `get_dls` fonksiyonu ile özelleştirilmiş veri kümesi yüklenir.
   - **Görevi:** Giriş görüntüleri ve bunların maske verileri (nesne bölgeleri) ile çalışarak, eğitim ve doğrulama için uygun veri yükleyicilerini sağlar.

6. **Görselleştirme ve Veri İşleme:**
   - **Kullanımı:** `visualize` fonksiyonu ile eğitim ve doğrulama veri kümelerinden rastgele örnekler görselleştirilir.
   - **Görevi:** Modelin eğitim veri kümesinden öğrendiği nesne tespitini görsel olarak doğrulamak ve modelin performansını değerlendirmek.

Bu yapı, Faster R-CNN'nin temel bileşenlerini ve bu projede nasıl kullanıldığını gösterir. Her bir bileşen, nesne tespit ve bölümleme görevlerinde belirli işlevleri yerine getirerek bir araya gelir ve sonuç olarak görüntü içindeki nesneleri tanımlamak için kullanılır.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from glob import glob
from PIL import Image
import random
import cv2
from matplotlib import pyplot as plt

# R-CNN için Özel Veri Kümesi Sınıfı
class CustomDataset(Dataset):
    def __init__(self, root, data, masks, transform=None):
        self.transform = transform
        self.im_paths = sorted(glob(f"{root}/{data}/*"))
        self.im_masks = sorted(glob(f"{root}/{masks}/*"))

        assert len(self.im_paths) == len(self.im_masks)
        
    def __len__(self):
        return len(self.im_paths)
    
    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        mask_path = self.im_masks[idx]

        im = Image.open(im_path).convert("RGB")
        gt = np.array(Image.open(mask_path).convert("L"))

        obj_indexes = np.unique(gt)[1:]
        gts = gt == obj_indexes[:, None, None]
        obj_numbers = len(obj_indexes)

        # Hedef için sözlük listesine dönüştürme
        target = []
        for i, box in enumerate(gts):
            pos = np.where(box)
            x1 = np.min(pos[1])
            x2 = np.max(pos[1])
            y1 = np.min(pos[0])
            y2 = np.max(pos[0])
            target.append({
                "boxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),  # Tek sınıf varsayımı (örn. insan)
                "masks": torch.tensor(box, dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.tensor((y2 - y1) * (x2 - x1), dtype=torch.float32),
                "iscrowd": torch.tensor(0, dtype=torch.int64),
            })

        if self.transform is not None:
            im, target = self.transform(im, target)

        return im, target

# R-CNN için Dönüşüm Fonksiyonu
def transform_image_target(image, target):
    image = T.ToTensor()(image)
    return image, target

# Veri yükleyicileri oluşturmak için fonksiyon
def get_dls(root, transformations, bs, split=[0.9, 0.1]):
    df = CustomDataset(root=root, data="train_images", masks="train_masks", transform=transformations)
    total_size = len(df)
    train_size = int(split[0] * total_size)
    val_size = total_size - train_size
    tr_ds, vl_ds = random_split(df, [train_size, val_size])

    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dl = DataLoader(vl_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    return tr_dl, val_dl

# Özel birleştirme işlevi
def collate_fn(batch):
    return tuple(zip(*batch))

# Veri kümesini başlatma ve görselleştirme
root = "/kaggle/input/human-detection-dataset/PennFudanPed/"
tr_dl, vl_dl = get_dls(root=root, transformations=transform_image_target, bs=8)

print(len(tr_dl), len(vl_dl))

def visualize(dataset, n_ims, rows, cmap=None):
    plt.figure(figsize=(20, 20))
    indices = [random.randint(0, len(dataset) - 1) for _ in range(n_ims)]

    for idx, index in enumerate(indices):
        img, targets = dataset[index]
        img = img.permute(1, 2, 0).numpy()
        img = img.copy()
        obj_count = 0

        for target in targets:
            for box in target["boxes"]:
                x, y, w, h = box.numpy().astype(int)
                obj_count += 1
                cv2.rectangle(img=img, pt1=(x, y), pt2=(w, h), color=(0, 255, 0), thickness=3)

        plt.subplot(rows, n_ims // rows, idx + 1)
        plt.imshow(img)
        plt.title(f"Görüntüde {obj_count} kişi var.")
        plt.axis("off")

root = "/kaggle/input/human-detection-dataset/PennFudanPed/"
dataset = CustomDataset(root=root, data="train_images", masks="train_masks", transform=transform_image_target)
visualize(dataset=dataset, n_ims=16, rows=4)

