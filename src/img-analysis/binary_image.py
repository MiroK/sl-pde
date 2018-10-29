from scipy.ndimage.measurements import label
from skimage.draw import circle as Circle, line as Line
from scipy.spatial import distance
from collections import namedtuple
import numpy as np
import types


# Stress test connected components
def get_random_binary_image(size, nconv=0):
    '''Image of size having binary valued pixels'''
    assert nconv >= 0

    img = np.random.randint(2, size=size)
    k = 0
    while k < nconv:
        img *= np.random.randint(2, size=size)
        k += 1

    return img

# Visual test of connected components
def get_circle_binary_image(size, ncircles, max_radius):
    '''Random 1 circle of max radius on image of size'''
    assert ncircles > 0 and max_radius > 0
    # Avoid collisions with boundaries
    x_idx = np.arange(max_radius, size[0]-max_radius)
    y_idx = np.arange(max_radius, size[1]-max_radius)
    # Canvas
    img = np.zeros(size, dtype=int)
    
    # Get centers
    cx_idx = x_idx[np.random.randint(len(x_idx)-1, size=ncircles)]
    cy_idx = y_idx[np.random.randint(len(y_idx)-1, size=ncircles)]
    # Smallest is radius is 1 pixel
    radii = np.random.randint(1, max_radius, size=ncircles)
    
    for c in zip(cx_idx, cy_idx, radii):
        circle = Circle(*c)
        img[circle] = 1

    return img


def get_bubble_binary_image(size, ncircles, max_radius, dist):
    '''Random 1 2-circle of max radius on image of size'''
    assert ncircles > 0 and max_radius > 0
    # Avoid collisions with boundaries
    x_idx = np.arange(max_radius, size[0]-max_radius)
    y_idx = np.arange(max_radius, size[1]-max_radius)
    # Canvas
    img = np.zeros(size, dtype=int)
    
    # Get centers
    cx_idx = x_idx[np.random.randint(len(x_idx)-1, size=ncircles)]
    cy_idx = y_idx[np.random.randint(len(y_idx)-1, size=ncircles)]
    # Smallest is radius is 1 pixel
    radii = np.random.randint(1, max_radius, size=ncircles)
    
    for c in zip(cx_idx, cy_idx, radii):
        circle0 = Circle(*c)
        try:
            img[circle0] = 1
            inserted0 = True
        except IndexError:
            inserted0 = False
            
        if inserted0:
            # The friend circle
            c0, c1, radius = c
            c1 = (radius + np.random.randint(c0, c0 + dist), c1, radius)

            circle1 = Circle(*c1)
            print 'xxx', circle1
            try:
                img[circle1] = 1
            except IndexError:
                img[circle0] = 0
    return img

# --

def pixel_collection(indices):
    '''Init a connected collection of pixels'''
    assert len(indices) == 2
    _, = set(map(len, indices)) 

    center = np.array(map(np.mean, indices))
    area = len(indices[0])  # In pixels
    # Fitting a rectangle in direction of axis
    # label algorithm rotates the first axis slowest
    axis_len = lambda idx: (np.max(idx) - np.min(idx)) + 1  # Assume contig

    len0 = axis_len(indices[0])
    # Longetst contig slice in 0 axis
    len1 = max(axis_len(indices[1][np.where(indices[0] == axis)[0]]) for axis in set(indices[0]))

    PixelCollection = namedtuple('pixel_collection',
                                 ('center', 'area', 'len0', 'len1', 'indices'))

    return PixelCollection(center, area, len0, len1, indices)

    
def find_connected_components(img, strict=True, structure='all'):
    '''Find islands of ones in the binary image'''
    assert not strict or set(img.flat) == set((0, 1))
    # For each pixel we consider ALL its neigbors i.e
    # x x x
    # x o x
    # x x x
    if structure == 'all':
        structure = np.ones((3, 3), dtype=np.int) 
    # # 0 x 0
    # # x o x
    # # 0 x 0
    elif structure == 'no_diag':
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.int)
    # User given. NOTE: not necessary to be 3x3. Then stencil can be
    # less localized
    else:
        assert isinstance(structure, np.ndarray)
   
    # Look for the component. In the out this is sort of a mask of the
    # original image
    connected_components, ncomponents = label(img, structure)
    
    component_indices = [np.where(connected_components == index)
                         for index in range(1, ncomponents+1)]
    assert len(component_indices) == ncomponents
    
    # NOTE: a single pixel can be an island (connected component). As we
    # don't filter by size such 1 pixel island can be present in the results.
    return map(pixel_collection, component_indices)


def find_objects(pixel_collection, bubble_area, noise_area=1, pair_constraints=None):
    '''
    A connected component with area smaller than `noise_area` is classified 
    as noise. Area larger than `bubble_area`(expected) means a large bubble.
    The remaining connected components are paired to form bubbles.
    '''
    noise, bubbles = [], []
    bubble_pairs = []
    for pc in pixel_collection:
        print pc.area
        if pc.area <= noise_area:
            noise.append(pc)
        elif pc.area >= bubble_area:
            bubbles.append(pc)
        else:
            bubble_pairs.append(pc)

    print len(bubble_pairs), '<<<<'
    bubble_pairs = make_pairs(bubble_pairs, pair_constraints)
    
    return {'noise': noise, 'bubbles': bubbles, 'bubble_pairs': bubble_pairs}


def make_pairs(pixel_collection, constraints=None):
    '''
    The idea exploited here is that due to scene illumination the connected 
    components closest to each other are a signature of one bubble
    '''
    # if constraints is None:
    #     constraints = [lambda x, y: True]
    # assert all(isinstance(f, types.FunctionType) for f in constraints)
    # # NOTE: at this point it makes more sense that 1sized pixel collections
    # # are not present
    # assert all(pc.area > 1 for pc in pixel_collection)
    
    # centers = [pc.center for pc in pixel_collection]
    # G = distance.cdist(centers, centers, 'euclidean')
    return []


# -------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = get_bubble_binary_image((256, 512), max_radius=10, ncircles=40, dist=13)

    plt.figure()
    plt.imshow(img)
    plt.colorbar()

    pixel_collection = find_connected_components(img)

    objects = find_objects(pixel_collection, bubble_area=400, noise_area=1, pair_constraints=None)

    for key in objects:
        print key, len(objects[key])

    bimg = np.zeros_like(img)
    for b in objects['bubbles']:
        bimg[b.indices] = 1
    # print len(pairs)
    # for value, (p, q) in enumerate(pairs, 2):
    #     pc_p = pixel_collection[p]
    #     pc_q = pixel_collection[q]
        
    #     img[Line(int(pc_p.center[0]), int(pc_p.center[1]),
    #              int(pc_q.center[0]), int(pc_q.center[1]))] = 1
    
    plt.figure()
    plt.imshow(img - bimg)
    plt.colorbar()

    plt.show()
