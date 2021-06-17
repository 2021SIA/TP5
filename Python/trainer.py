import model
import imageUtils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('images')
args = parser.parse_args()
modelsDir = args.model
imagesDir = args.images

# size = (36, 36, 3)

images = imageUtils.loadImages(imagesDir)
size = images[0].shape
# showImage(images[0])

# imageUtils.showImage(images[0])
# exit(0)

def loadOrCreate(forceCreate: bool = False):
    vae = None
    if not forceCreate:
        vae = model.VAE.load(modelsDir)
    if vae is None:
        vae = model.buildModel(size, 1)
    return vae

vae = loadOrCreate(False)
vae.train(images, epochs=2000, batch_size=16)

vae.save(modelsDir)

# print(vae.getLatent(images[0]))
imageUtils.showImage(vae.predict(vae.getLatent(images[0])))