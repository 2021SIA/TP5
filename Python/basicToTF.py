import model
import imageUtils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('weights')
parser.add_argument('model')
parser.add_argument('dims', type=int)
args = parser.parse_args()
modeldir = args.model
weights = args.weights
dims = args.dims

vae = model.VAE.loadFromBasic(weights, (36, 36, 3), dims)
# vae = model.VAE.loadFromBasic('basicWeights/weights 4.csv', (36, 36, 3), 4)


vae.save(modeldir)

# images = imageUtils.loadImages('Images/Microsoft')
# for i in range(20):
#     imageUtils.showImage(vae.predict(vae.getLatent(images[i])))