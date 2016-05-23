from skimage.feature import hog
from skimage import color
from skimage.io import imread

import pandas as pd

df = pd.read_csv("./local_path.csv")

#PREFIX = "/data/damoncrockett/"

#df.local_path = df.local_path.map(lambda x: PREFIX + x[33:])

import numpy as np

vector_list = []

for i in range(1000):
    img = imread(df.local_path.loc[i])
    
    if img.shape[0] < 200 or img.shape[1] < 200:
        df.drop(i)
    else:
		img_gray = color.rgb2gray(img)
		fd = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(4, 4))
		vector_list.append(fd)
		print i, len(fd), df.local_path.loc[i]
    
X = np.vstack(vector_list)

from sklearn.manifold import TSNE as tsne

tsne = tsne(n_components=2)
tsne.fit(X)
subspace_tsne = pd.DataFrame(tsne.fit_transform(X),columns=["x","y"])

num_bins = 64

subspace_tsne['grid_x'] = pd.cut(subspace_tsne['x'],num_bins,labels=False)
subspace_tsne['grid_y'] = pd.cut(subspace_tsne['y'],num_bins,labels=False)

subspace_tsne['local_path'] = df.local_path[:len(subspace_tsne)]

# I should save the dataframe here, so later maybe I can use full images

thumb_side = 128

from PIL import Image

n = len(subspace_tsne)

grid_side = num_bins

px_w = thumb_side * grid_side
px_h = thumb_side * grid_side

canvas = Image.new('RGB',(px_w,px_h),(50,50,50))

for i in range(n):
    im = Image.open(subspace_tsne.local_path.loc[i])
    im.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)
    x = subspace_tsne.grid_x.loc[i] * thumb_side
    y = subspace_tsne.grid_y.loc[i] * thumb_side
    canvas.paste(im,(x,y))
    
canvas.save("./hog.png")