from skimage.feature import hog
from skimage import color
from skimage.io import imread

import pandas as pd

df = pd.read_csv("./local_path.csv")

#PREFIX = "/data/damoncrockett/"

#df.local_path = df.local_path.map(lambda x: PREFIX + x[33:])

import numpy as np

vector_list = []

smp = 4000

for i in range(smp):
    img = imread(df.local_path.loc[i])
    
    if img.shape[0] < 200 or img.shape[1] < 200:
        df.drop(i)
    else:
		img_gray = color.rgb2gray(img)
		fd = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(4, 4))
		vector_list.append(fd)
		print i, len(fd), df.local_path.loc[i]
    
X = np.vstack(vector_list)

from sklearn.cluster import KMeans

k = 32

kmeans = KMeans(n_clusters = k)
kmeans.fit(X)

df = df[:len(vector_list)]

df['clusters'] = kmeans.labels_

centroids = pd.DataFrame(kmeans.cluster_centers_)

from sklearn.decomposition import RandomizedPCA as pca
pca = pca(n_components=2)
Y = centroids.as_matrix()
pca.fit(Y)
subspace = pd.DataFrame(pca.transform(Y),columns=["x","y"])

euclidean_distance = []

hog_points = pd.DataFrame(X)

for i in range(len(df)):
    tmp = hog_points.loc[i].as_matrix()
    cluster_integer = int(df.clusters.loc[i])
    
    euclidean_distance_i = np.linalg.norm(tmp - centroids.loc[cluster_integer].as_matrix())
    euclidean_distance.append(euclidean_distance_i)
    
df['euclidean_distance'] = euclidean_distance


# Now plotting...

num_bins = int(round(np.sqrt(len(df)))) * 4

# adding in some extremes to push the edges out
xmin = subspace.x.min()
xmax = subspace.x.max()
ymin = subspace.y.min()
ymax = subspace.y.max()

x = [xmin - 1, xmax + 1]
y = [ymin - 1, ymax + 1]

tmp = pd.DataFrame(x,columns=["x"])
tmp["y"] = y
subspace = subspace.append(tmp)

subspace['x_bin'] = pd.cut(subspace['x'],num_bins,labels=False)
subspace['y_bin'] = pd.cut(subspace['y'],num_bins,labels=False)

# now we can remove the extreme points we used as grid expanders
subspace = subspace[:k]

# now to expand the grid by simple multiplication
factor = 2
subspace["x_grid"] = subspace.x_bin * factor
subspace["y_grid"] = subspace.y_bin * factor

from shapely.geometry import Point

centroid_point = []
for i in range(len(subspace)):
    centroid_point.append(Point(subspace.x_grid.loc[i],subspace.y_grid.loc[i]))
    
subspace['centroid_point'] = centroid_point


#GRID LIST
grid_side = num_bins * factor

x,y = range(grid_side) * grid_side, np.repeat(range(grid_side),grid_side)
grid_list = pd.DataFrame(x,columns=['x'])
grid_list['y'] = y

point = []
for i in range(len(grid_list)):
    point.append(Point(grid_list.x.loc[i],grid_list.y.loc[i]))

grid_list['point'] = point

open_grid = list(grid_list.point)

centroids = list(subspace.centroid_point)


#REMOVAL OF CENTROIDS FROM OPEN_GRID LIST
open_grid = [item for item in open_grid if item not in centroids]


#PLOT FUNCTION
from PIL import Image

thumb_side = 128

px_w = thumb_side * grid_side
px_h = thumb_side * grid_side

canvas = Image.new('RGB',(px_w,px_h),(50,50,50))

def plot():
    for i in range(len(subspace)):
        centroid = subspace.centroid_point.loc[i]
        try:
            candidates = df[df.clusters==i]
            candidates.sort("euclidean_distance",inplace=True)
            best = candidates.iloc[0]
            im = Image.open(best.local_path)
            im.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)
            closest_open = min(open_grid,key=lambda x: centroid.distance(x))
            x = int(closest_open.x) * thumb_side
            y = int(closest_open.y) * thumb_side
            canvas.paste(im,(x,y))
            idx = df[df.local_path==best.local_path].index
            df.drop(idx,inplace=True)
            open_grid.remove(closest_open)
            print i
        except:
            print "cluster empty"
            
iterations = list(df.clusters.value_counts())[0]

for i in range(iterations):
    print "plot_",i
    plot()
    
canvas.save("./tmp.png")