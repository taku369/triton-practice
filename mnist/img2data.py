import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

im = Image.open(args.filename)
im = im.convert('L')
im_arr = np.array(im)
print(", ".join(list(map(lambda x : f"{x:.3f}", (im_arr / 255.0).flatten()))))