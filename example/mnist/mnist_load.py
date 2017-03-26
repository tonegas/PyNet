import numpy, os, struct

def to_one_hot_vect(vect, num_classes):
    on_hot_vect = []
    for i,target in enumerate(vect):
        on_hot_vect.append(numpy.zeros(num_classes))
        on_hot_vect[i][target] = 1
    return on_hot_vect

def load_mnist_dataset(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = numpy.fromfile(flbl, dtype=numpy.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = numpy.fromfile(fimg, dtype=numpy.uint8).reshape(len(lbl), rows*cols).astype(numpy.float)

    return zip(img, numpy.array(to_one_hot_vect(lbl,10)))

