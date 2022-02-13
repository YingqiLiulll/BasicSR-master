from PIL import Image
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
f = os.listdir('C:/Users/13637/Desktop/Set5BIL/GTmod12')
scale = 4

for i in f:
    pf = os.path.join('C:/Users/13637/Desktop/Set5BIL/GTmod12', i)
    image = Image.open(pf).convert('RGB')
    hr = image
    lr = hr.resize((hr.width // scale, hr.height // scale), resample=Image.BILINEAR)
    bilinear = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BILINEAR)
    # pf1 = os.path.join('F:/classical_SR_datasets/OST_dataset/OutdoorSceneTest300/x4HR', i)
    pf2 = os.path.join('C:/Users/13637/Desktop/Set5BIL/x4LR', i)
    pf3=os.path.join('C:/Users/13637/Desktop/Set5BIL/x4BIL',i)
    # hr.save(pf1)
    lr.save(pf2)
    bilinear.save(pf3)
