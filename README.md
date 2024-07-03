# human-detection
computer vision project to detect and count humans
### Gerekli Kütüphanelerin İçe Aktarılması

İlk olarak, projemizde kullanacağımız gerekli kütüphaneleri içe aktarıyoruz:
- **torch**: PyTorch, derin öğrenme modellerini oluşturmak ve eğitmek için kullanılan bir kütüphane.
- **numpy**: Bilimsel hesaplamalar için kullanılan bir kütüphane.
- **torch.utils.data.Dataset, DataLoader, random_split**: PyTorch'un veri yükleme ve ayırma araçları.
- **torchvision.transforms**: Görüntü verilerini dönüştürmek için kullanılan araçlar.
- **glob**: Dosya yollarını bulmak için kullanılır.
- **PIL.Image**: Görüntü işleme için kullanılır.
- **random**: Rastgele seçimler yapmak için kullanılır.
- **cv2**: OpenCV kütüphanesi, görüntü işleme ve bilgisayarla görme görevleri için kullanılır.
- **matplotlib.pyplot**: Grafik ve görselleştirme için kullanılır.

### CustomDataset Sınıfının Tanımlanması

Bu sınıf, verileri yüklemek ve işlem yapmak için kullanılır.

#### __init__ Metodu

```python
class CustomDataset(Dataset):
    def __init__(self, root, data, masks, transform=None):
        self.transform = transform
        self.im_paths = sorted(glob(f"{root}/{data}/*"))
        self.im_masks = sorted(glob(f"{root}/{masks}/*"))
        
        assert len(self.im_paths) == len(self.im_masks)
```

- **root**: Veri dosyalarının bulunduğu ana klasör.
- **data**: Görüntülerin bulunduğu alt klasör.
- **masks**: Maskelerin bulunduğu alt klasör.
- **transform**: Görüntülere uygulanacak dönüşümler.
- **glob** ve **sorted** kullanılarak görüntü ve maske dosya yolları toplanır ve sıralanır.
- **assert**: Görüntü ve maske sayılarının eşit olup olmadığını kontrol eder.

#### __len__ Metodu

```python
    def __len__(self):
        return len(self.im_paths)
```

- Bu metod, veri kümesindeki toplam görüntü sayısını döner.

#### __getitem__ Metodu

```python
    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        mask_path = self.im_masks[idx]
        
        im = Image.open(im_path).convert("RGB")
        gt = np.array(Image.open(mask_path).convert("L"))
        
        obj_indexes = np.unique(gt)[1:]
        gts = gt == obj_indexes[:, None, None]
        obj_numbers = len(obj_indexes)
        
        boxes = []
        for i, box in enumerate(gts):
            pos = np.where(box)
            x1 = np.min(pos[1])
            x2 = np.max(pos[1])
            y1 = np.min(pos[0])
            y2 = np.max(pos[0])
            boxes.append([x1, y1, x2, y2])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((obj_numbers), dtype=torch.int64)
        gts = torch.as_tensor(gts, dtype=torch.uint8)
        im_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((obj_numbers), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = gts
        target["image_id"] = im_id
        target["area"] = area
        target["iscrowd"] = is_crowd
        
        if self.transform is not None:
            im = self.transform(im)
        
        return im, target
```

- **idx**: İstenen verinin indeksidir.
- Görüntü ve maske dosyaları yüklenir.
- Maskedeki her nesne için bounding box'lar hesaplanır.
- Veriler PyTorch tensörlerine dönüştürülür ve dönüşümler uygulanırsa, görüntü dönüştürülür.

### Veri Yükleyicilerin Oluşturulması

Bu bölümde veri yükleyicileri oluşturmak için bir fonksiyon tanımlanır.

#### get_dls Fonksiyonu

```python
def get_dls(root, transformations, bs, split=[0.9, 0.1]):
    df = CustomDataset(root=root, data="train_images", masks="train_masks", transform=transformations)
    total_size = len(df)
    train_size = int(split[0] * total_size)
    val_size = total_size - train_size
    tr_ds, vl_ds = random_split(df, [train_size, val_size])
    
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=4)
    val_dl = DataLoader(vl_ds, batch_size=bs, shuffle=False, num_workers=4)
    
    return tr_dl, val_dl
```

- **root**: Veri dosyalarının bulunduğu ana klasör.
- **transformations**: Görüntülere uygulanacak dönüşümler.
- **bs**: Batch size (her bir yükleyiciye kaç veri noktası verileceği).
- **split**: Eğitim ve doğrulama veri setlerinin oranı.
- Veri kümesi bölünür ve veri yükleyicileri oluşturulur.

### Veri Kümesini Görselleştirme

Görüntüleri ve bounding box'ları görselleştirmek için bir fonksiyon tanımlanır.

#### visualize Fonksiyonu

```python
def visualize(dataset, n_ims, rows, cmap=None):
    plt.figure(figsize=(20, 20))
    indices = [random.randint(0, len(dataset) - 1) for _ in range(n_ims)]
    
    for idx, index in enumerate(indices):
        img, target = dataset[index]
        orig_img = img.permute(1, 2, 0).numpy()
        orig_img = orig_img.copy()
        obj_count = 0
        
        for i, bbox in enumerate(target["boxes"]):
            x, y, w, h = [int(t.item()) for t in bbox]
            obj_count += 1
            cv2.rectangle(img=orig_img, pt1=(x, y), pt2=(w, h), color=(0, 255, 0), thickness=3)
        
        plt.subplot(rows, n_ims // rows, idx + 1)
        plt.imshow(orig_img)
        plt.title(f"Image has {obj_count} people.")
        plt.axis("off")
```

- **dataset**: Görselleştirilecek veri kümesi.
- **n_ims**: Görselleştirilecek görüntü sayısı.
- **rows**: Görselleştirme için satır sayısı.
- Her görüntü için bounding box'ları çizerek görselleştirir.

### Veri Kümesini Başlatma ve Görselleştirme

Bu adımda veri kümesi başlatılır ve visualize fonksiyonu çağrılarak görüntüler görselleştirilir.

```python
dataset = CustomDataset(root=root, data="train_images", masks="train_masks", transform=tfs)
visualize(dataset=dataset, n_ims=16, rows=4)
```

- **dataset**: Özel veri kümesi başlatılır.
- **visualize**: Görüntüleri ve bounding box'ları görselleştirmek için çağrılır.

### Teorik Arkaplan

Bu kod, bir veri kümesiyle çalışmak ve bu veriyi derin öğrenme modelleri için uygun bir formata dönüştürmek amacıyla kullanılır. Temel adımlar şunlardır:

1. **Veri Yükleme**: Görüntüler ve maskeler dosyalardan yüklenir.
2. **Veri İşleme**: Maskelerden bounding box'lar ve diğer hedef veriler hesaplanır.
3. **Veri Dönüştürme**: Görüntülere dönüşümler uygulanır (örneğin, tensörlere dönüştürme).
4. **Veri Yükleyicileri Oluşturma**: Eğitim ve doğrulama veri yükleyicileri oluşturulur.
5. **Görselleştirme**: Görüntüler ve bounding box'lar görselleştirilir.

Bu işlemler, derin öğrenme modellerinin görüntü verileri üzerinde eğitilmesi ve değerlendirilmesi için gereklidir. Veri kümesi, modelin ihtiyaç duyduğu biçime getirilir ve modelin eğitim sürecine hazırlanır.
