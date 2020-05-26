import h5py
import numpy as np
from typing import Tuple, Callable, TypeVar
import random


TypeSet = Tuple[np.ndarray, np.ndarray]
TypeCallableSet = Callable[[int, int], TypeSet]
TypeCTrackSet = Callable[[], Tuple[np.ndarray]]
TypeHolderSet = Tuple[TypeCallableSet, TypeCallableSet, TypeCTrackSet, np.ndarray]

def holder(dataset, real, idset, m) -> TypeHolderSet:
    thetas = np.zeros([dataset[0].size])

    def trainset(f=0, t=m)-> TypeSet:
        return dataset[f:t], real[f:t]
    def testset(f=m, t = real.size)->TypeSet:
        return dataset[f:t], real[f:t]
    def trackset()->Tuple[np.ndarray]:
        return idset
    return trainset, testset, trackset, thetas


def load(m=2950, filename="/mnt/d/antoniospace/schoolspace/ai_projects/xseed-regression/data/easy.h5") -> TypeHolderSet:
    original = h5py.File(filename, 'r')
    dataset = np.array(original['attributes'])
    real = np.array(original['hotttnesss'])
    idset = np.array(original['tracks_id'])
    original.close()
    return holder(dataset, real, idset, m)

def randomTracks(tracks, times=1):
    for j in range(times):
        random_tracks=""
        for i in range(3):
            index = random.randint(0, tracks.size - 2)
            random_tracks += tracks[index].decode() + " "
        print("Random tracks", random_tracks)

def randomInterval(dataset, window, fstr, tstr, initial=0):
    f = random.randint(initial, dataset.size - window - 2)
    t = f + window
    message = fstr + " " + str(f) + " " + tstr + " " + str(t)
    
    return f, t, message