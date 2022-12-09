import os
from PIL import Image

dir = 'F:/dehaze/nyuhaze500/gt/'
new_dir = 'F:/dehaze/nyuhaze500_square/gt/'
files = os.listdir(dir)
files.sort()
a = 0
a1 = 0
for each_bmp in files:  # 遍历，进行批量转换
    first_name, second_name = os.path.splitext(each_bmp)
    print("first_name, second_name:", first_name, second_name)
    each_bmp = os.path.join(dir, each_bmp)
    image = Image.open(each_bmp)
    img = image.convert('RGB')
    if img.size[0] > img.size[1]:
        x = abs((img.size[0] - img.size[1]) / 2)
        y = 0
        w = img.size[1]
    else:
        x = 0
        y = abs((img.size[0] - img.size[1]) / 2)
        w = img.size[0]
       # 第一个参数左上x距离，第二参数左上y距离，第三个参数x+w，第四个参数y+h
    img_c = img.crop([x, y, x + w, y + w])
    image_data = img_c.resize((512, 512))  # 缩放
    print(image_data)
    image_data.save(new_dir + first_name + '.png')  # 保存图片 参数一保存图片的格式 2为路径