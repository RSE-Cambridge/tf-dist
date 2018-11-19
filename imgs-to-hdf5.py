import PIL
import PIL.Image
import numpy as np
import h5py
import glob
import sys, os

output = sys.argv[1]
basepath = sys.argv[2]

img_size = (64,64)

def h5_append(dataset, item):
    dataset.resize(dataset.shape[0]+1, axis=0)
    dataset[dataset.shape[0]-1] = item

with h5py.File(output, "w") as f:
    categories = f.create_dataset("categories", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=bytes))

    for s in ["test", "train", "val"]:
        s_group = f.create_group(s)

        data = s_group.create_dataset("data", (0,)+img_size+(3,), maxshape=(None,)+img_size+(3,), dtype='u4', chunks=True)
        labels = s_group.create_dataset("labels", (0,), maxshape=(None,), dtype='u2', chunks=True)

        for category_index, category_path in enumerate(glob.iglob(os.path.join(basepath, s, '*'))):
            h5_append(categories, os.path.basename(category_path))

            for ext in ['jpg', 'jpeg', 'JPG', 'JPEG']:
                for file_path in glob.iglob(os.path.join(category_path, '*.%s'%ext)):
                    image = PIL.Image.open(file_path)
                    image.thumbnail(img_size)

                    h5_append(data, np.asarray(image))
                    h5_append(labels, category_index)

