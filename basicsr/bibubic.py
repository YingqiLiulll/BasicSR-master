from PIL import Image
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
f = os.listdir('F:/classical_SR_datasets/OST_dataset/OutdoorSceneTest300/OutdoorSceneTest300')
scale = 4

for i in f:
    pf = os.path.join('F:/classical_SR_datasets/OST_dataset/OutdoorSceneTest300/OutdoorSceneTest300', i)
    image = Image.open(pf).convert('RGB')
    hr = image
    lr = hr.resize((hr.width // scale, hr.height // scale), resample=Image.BICUBIC)
    bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)
    pf1 = os.path.join('F:/classical_SR_datasets/OST_dataset/OutdoorSceneTest300/x4HR', i)
    pf2 = os.path.join('F:/classical_SR_datasets/OST_dataset/OutdoorSceneTest300/x4LR', i)
    pf3=os.path.join('F:/classical_SR_datasets/OST_dataset/OutdoorSceneTest300/x4BIC',i)
    hr.save(pf1)
    lr.save(pf2)
    bicubic.save(pf3)
