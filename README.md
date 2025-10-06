# Custom-Data-Classification-With-ResNet
# [TR]
## Projenin Amacı
Bu projeyi, PyTorch ile Transfer Learning tekniğini uygulayarak geliştirdim. Amaç, ResNet-18 modelinin önceden eğitilmiş modelini kullanarak, veri setimize özgü niş envanter sınıflarına (Cup, Adapter, Mouse, Toy Car) adapte etmektir.
ImageNet'in genel sınıflarında bu nesnelerin genel karşılıkları bulunsa dahi, modelin doğruluk ve fine-tuning seviyesini artırmak amacıyla son katmanını yeniden eğittim. Süreç boyunca bu dört nesne için kapsamlı bir veri seti derleyip, modelin eğitimden sonraki yetenklerini ekran görüntüleriyle kanıtladım.

## 📸 Veri Seti Oluşturma Süreci
- Toplamda 2100’den fazla görüntü topladım ve bu görüntülerle 4 ayrı nesne için (Mouse, Custom Cup, Adapter, Toy Car) veri seti oluşturdum.
- Modelin genelleme yeteneğini arttırmak ve nesneleri daha iyi algılayabilmesi için tüm nesneleri, 3 farklı arka plan, 3 farklı aydınlatma koşulu ve çeşitli açılarla çektim.
- Bu sayede modelin, farklı ortam ve açılarda da doğru tahmin yapabilmesi hedefledim.

## 💻 Kullanılan Teknolojiler
- Python 3.11.8
- PyTorch
- Torchvision
- ResNet18

## ⚙️ Kurulum
GEREKLİ KÜTÜPHANELERİ KURUN
```bash
pip install torch torchvision
```

## 🚀 Çalıştırma
```
└── /dataset
    ├── /train
    │   ├── /adapter      # Adatör eğitim görselleri
    │   └── /cup          # Kupa eğitim görselleri
    │   ├── /mouse        # Mouse eğitim görselleri
    │   └── /toy_car      # Oyuncak araba eğitim görselleri
    │
    └── /val
    │   ├── /adapter      # Adatör test görselleri
    │   └── /cup          # Kupa test görselleri
    │   ├── /mouse        # Mouse test görselleri
    │   └── /toy_car      # Oyuncak araba test görselleri  
```

1. Dataset'inizi yerleştirirken bu dosya düzenine uymaya özen gösterin.
2. **Model.py** dosyasına kullanmak istediğiniz ResNet modelini yazabilirsiniz.
3. **Train.py** dosyasını çalıştırıp modelinizi eğitin. (Bu dosyayıda sistem gereksinimlerinize ve datasetinize göre güncelleyin)
4. Model eğitildikten sonra **Prediction.py** modelinizin en iyi ağırlıklarını vererek tahmin yapmasını sağlayın.

## 📸 Ekran Görüntüleri 
| 1 | 2 | 
| :---------------------------------: | :------------------------: |
|<img width="540" height="540" alt="5" src="https://github.com/user-attachments/assets/f07dfe21-b1df-429f-871e-09e3331e319a" />| <img width="540" height="540" alt="6" src="https://github.com/user-attachments/assets/df35a514-9312-45b5-924c-8852cab5757a" />
| 3 | 4 | 
|<img width="540" height="540" alt="7" src="https://github.com/user-attachments/assets/376c98cf-59bc-411e-ab0b-683c72bc9708" />| <img width="540" height="540" alt="8" src="https://github.com/user-attachments/assets/597ef2ac-168f-4928-a970-7254394da749" />


## BU PROJE HİÇBİR ŞEKİLDE TİCARİ AMAÇ İÇERMEMEKTEDİR.

# [EN]
## Project Objective
I developed this project using the Transfer Learning technique with PyTorch. The goal is to adapt the pre-trained ResNet-18 model to the niche inventory classes in our dataset (Cup, Adapter, Mouse, Toy Car).
Even though there may be general equivalents of these objects in ImageNet's classes, I retrained the last layer to increase the model's accuracy and fine-tuning level. Throughout the process, I compiled a comprehensive dataset for these four objects and demonstrated the model's capabilities after training with screenshots.

## 📸 Dataset Creation Process
- I collected over 2,100 images in total and created a dataset for 4 separate objects (Mouse, Custom Cup, Adapter, Toy Car).
- To improve the model's generalization ability and help it recognize objects better, I captured all objects under 3 different backgrounds, 3 different lighting conditions, and various angles.
- This way, the model is expected to make accurate predictions even in different environments and angles.

## 💻 Technologies Used
- Python 3.11.8
- PyTorch
- Torchvision
- ResNet18

## ⚙️ Installation
INSTALL REQUIRED LIBRARIES
```bash
pip install torch torchvision
```
```
└── /dataset
    ├── /train
    │   ├── /adapter      # Adapter Training Images
    │   └── /cup          # Cup Training Images
    │   ├── /mouse        # Mouse Training Images
    │   └── /toy_car      # Toy car Training Images
    │
    └── /val
    │   ├── /adapter      # Adapter Test Images
    │   └── /cup          # Cup Test Images
    │   ├── /mouse        # Mouse Test Images
    │   └── /toy_car      # Toy car Test Images  
```

1. Make sure to follow this folder structure when placing your dataset.
2. You can specify the ResNet model you want to use in the **Model.py** file.
3. Run the **Train.py** file to train your model. (Update this file according to your system requirements and dataset)
4. After the model is trained, use **Prediction.py** to make predictions by providing the best weights of your model.

## 📸 Screenshots
| 1 | 2 | 
| :---------------------------------: | :------------------------: |
|<img width="540" height="540" alt="5" src="https://github.com/user-attachments/assets/f07dfe21-b1df-429f-871e-09e3331e319a" />| <img width="540" height="540" alt="6" src="https://github.com/user-attachments/assets/df35a514-9312-45b5-924c-8852cab5757a" />
| 3 | 4 | 
|<img width="540" height="540" alt="7" src="https://github.com/user-attachments/assets/376c98cf-59bc-411e-ab0b-683c72bc9708" />| <img width="540" height="540" alt="8" src="https://github.com/user-attachments/assets/597ef2ac-168f-4928-a970-7254394da749" />

## THIS PROJECT DOES NOT CONTAIN ANY COMMERCIAL PURPOSES IN ANY WAY.
