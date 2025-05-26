import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder

# This is to fix deprecated alias in numpy, but still used in slgbuilder
# You need to add it to any script using (this version of) slgbuilder
np.bool = bool
np.int = int

I = skimage.io.imread('data/peaks_image.png').astype(np.int32)

fig, ax = plt.subplots(1,2)
ax[0].imshow(I, cmap='gray')


layers = [slgbuilder.GraphObject(0*I), slgbuilder.GraphObject(0*I)] # no on-surface cost
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)

# Adding regional costs, 
# the region in the middle is bright compared to two darker regions.
helper.add_layered_region_cost(layers[0], I, 255-I)
helper.add_layered_region_cost(layers[1], 255-I, I)

# Adding geometric constrains
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=1, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=1)

# Cut
helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

# Visualization
ax[1].imshow(I, cmap='gray')
for line in segmentation_lines:
    ax[1].plot(line, 'r')

plt.show()



