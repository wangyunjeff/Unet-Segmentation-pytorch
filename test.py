


from PIL import Image
import numpy as np
def get_train(image,shape):
    # image = cvtColor(image)
    h, w = shape

    iw, ih = image.size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    # new_image = Image.new('L', [w, h], (128))
    # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return image

def preprocess_input(image):
    image /= 7800.0
    image[image < 0] = 0
    image = np.array([image, image, image])
    return image

jpg = Image.open(r'G:\Code\tf2_torch\pytorch\2_Projects\Terrain\RIDGE\pspnet_ex1\pspnet-pytorch-master\VOCdevkit\VOC2007\JPEGImages\400_2400.tif')
# -------------------------------#
#   数据增强
# -------------------------------#
jpg = get_train(jpg, (512,512))

jpg = preprocess_input(np.array(jpg, np.float64))
pass