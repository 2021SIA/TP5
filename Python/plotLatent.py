import model, imageUtils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import argparse

def scatter3d(x, y, z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=z)
    scalarMap.set_array(cs)
    plt.show()

def scatter2d(x, y):
    plt.scatter(x, y, marker='x')
    plt.show()

def scatter(*datas):
    if len(datas) == 2:
        scatter2d(*datas)
    if len(datas) == 3:
        z = datas[2]
        scatter3d(*datas, (min(z), max(z)))

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('images')
args = parser.parse_args()
modeldir = args.model
imagesdir = args.images

images = imageUtils.loadImages(imagesdir)
# images = imageUtils.loadImages('Images/Microsoft')
vae = model.VAE.load(modeldir)
# vae = model.VAE.load('modelBasic2')

allLatent = []

for image in images:
    latent = vae.getLatent(image)
    allLatent.append(latent)

scatter(*zip(*allLatent))
