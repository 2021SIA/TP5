import model, imageUtils
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('images')
args = parser.parse_args()
modeldir = args.model
imagesdir = args.images

images = imageUtils.loadImages(imagesdir)
vae = model.VAE.load(modeldir)

min = vae.getLatent(images[0])
max = min

for image in images:
    latent = vae.getLatent(image)
    min = np.minimum(min, latent)
    max = np.maximum(max, latent)

print(min, max)