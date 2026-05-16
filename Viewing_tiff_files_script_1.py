import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np

img = tiff.imread("mask.tif")

print("Shape:", img.shape)
print("Datatype:", img.dtype)
print("Min:", img.min())
print("Max:", img.max())
print("Unique values:", np.unique(img)[:20])

plt.imshow(img, cmap="gray", vmin=img.min(), vmax=img.max())
plt.colorbar()
plt.title("TIFF Image")
plt.show()