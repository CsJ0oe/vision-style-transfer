import time

from tensorflow.python.keras.utils.data_utils import get_file

from src.StyleTransferModel import TransferStyle
from src.utils import *

style_layers = [-2, -3, -4]
content_layers = [1, 2, 4, 5, 7]

model = TransferStyle(style_layers, content_layers)


content = 'https://images.adsttc.com/media/images/5d66/f567/284d/d161/f000/02c9/newsletter/2.jpg?1567028571'
style = 'https://medias.gazette-drouot.com/prod/medias/mediatheque/25336.jpg'

base_image_path = get_file(f"{int(time.time()+4)}", content)
img = load_img(base_image_path)
img_content = tf.keras.preprocessing.image.img_to_array(img)
height, width, _ = img_content.shape

style_reference_image_path = get_file(f"{int(time.time()+3)}", style)
img = load_img(style_reference_image_path)
img_style = tf.keras.preprocessing.image.img_to_array(img)

gen_height = 400
gen_width = int(width * gen_height / height)

content = preprocess_image(img_content, gen_height, gen_width)
style = preprocess_image(img_style, gen_height, gen_width)

model.fit(1, content, style, 'transfertest')
