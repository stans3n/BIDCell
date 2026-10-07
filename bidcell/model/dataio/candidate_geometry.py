"""Full-nucleus pixel index and exact nearest-pixel distances (1 um grid)."""
from collections import OrderedDict
import numpy as np
from scipy.spatial import cKDTree


def transform(array, flip, rotation):
    # imgaug Rot90(k) is clockwise; numpy's positive k is counterclockwise.
    value = np.flip(array, 0) if flip else array
    return np.ascontiguousarray(np.rot90(value, -int(rotation), axes=(0, 1)))


class CandidateGeometry:
    def __init__(self, labels, cache_size=512):
        yy, xx = np.nonzero(labels)
        ids = labels[yy, xx]
        order = np.argsort(ids, kind='stable')
        self.points = np.column_stack((yy[order], xx[order])).astype(np.float64)
        unique, first, counts = np.unique(ids[order], return_index=True, return_counts=True)
        self.index = {int(i): (int(s), int(n)) for i, s, n in zip(unique, first, counts)}
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def tree(self, cell_id):
        cell_id = int(cell_id)
        if cell_id not in self.index:
            raise ValueError(f'missing full nucleus {cell_id}')
        if cell_id not in self.cache:
            first, count = self.index[cell_id]
            self.cache[cell_id] = cKDTree(self.points[first:first+count])
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(cell_id)
        return self.cache[cell_id]

    def maps(self, ids, row, col, size, flip, rotation, nuclei, support):
        yy, xx = np.mgrid[row:row+size, col:col+size]
        query = np.column_stack((yy.ravel(), xx.ravel()))
        result = np.empty((len(ids), 4, size, size), np.float32)
        areas = []
        for k, cell_id in enumerate(ids):
            tree = self.tree(cell_id)
            area = self.index[int(cell_id)][1]
            radius = np.sqrt(area / np.pi)
            distance = tree.query(query, workers=1)[0].reshape(size, size)
            result[k, 0] = nuclei[:, :, k]
            result[k, 1] = transform(np.minimum(distance/(radius+1e-6), 4)/4, flip, rotation)
            result[k, 2] = np.log1p(radius/10)
            result[k, 3] = support[:, :, k]
            areas.append(area)
        return result, np.asarray(areas, dtype=np.int64)
