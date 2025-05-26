import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder

# This is to fix deprecated alias in numpy, but still used in slgbuilder
np.bool = bool
np.int = int

#%% input
# I = skimage.io.imread('data/layers_A.png').astype(np.int32)
I = skimage.io.imread('data/rammed-earth-layers-limestone.jpg').astype(np.int32)

fig, ax = plt.subplots(1,4)
ax[0].imshow(I, cmap='gray')
ax[0].set_title('input image')

#%% one line
delta = 3

layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

ax[1].imshow(I, cmap='gray')
ax[1].plot(segmentation_line, 'r')
ax[1].set_title(f'delta = {delta}')


#%% a smoother line
delta = 1

layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

ax[2].imshow(I, cmap='gray')
ax[2].plot(segmentation_line, 'r')
ax[2].set_title(f'delta = {delta}')


#%% tow lines
layers = [slgbuilder.GraphObject(I), slgbuilder.GraphObject(I) ]
delta = 3

helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=30)

helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

ax[3].imshow(I, cmap='gray')
for line in segmentation_lines:
    ax[3].plot(line, 'r')
ax[3].set_title('two dark lines')

plt.show()