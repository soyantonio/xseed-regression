import h5py
import numpy as np

GMETADATA = 0
GANALYSIS = 1
GMUSICBRZ = 2

print("Welcome")
filename= "/mnt/d/tmp_ml/msd/MillionSongSubset/AdditionalFiles/subset_msd_summary_file.h5"
original = h5py.File(filename, 'r')

metadata = original['metadata']['songs']       #V5320
analysis = original['analysis']['songs']       #V220
musicbrz = original['musicbrainz']['songs'] #V8

metadata_dt = metadata.dtype
analysis_dt = analysis.dtype
musicbrz_dt = musicbrz.dtype

print("Dataset loaded")

def np_filter(fn):
    def wrapper(*args, **kwargs):
        rlambda, rarr = fn(*args, **kwargs)
        return np.array(list(filter(rlambda, rarr)))
    return wrapper

@np_filter
def greater_filter(items, f=5, threshold=0, group=GMETADATA):
    return lambda x: x[group][f] > threshold, items

@np_filter
def nan_filter(items, f=16, threshold=False, group=GMETADATA):
    return lambda x: np.isnan(x[group][f]) == threshold, items

# Join three groups into a one, by each record
dset = np.array(list(zip(metadata, analysis, musicbrz)))
print("Dataset converted")

# Metada valid_songhotttnesss
dset = nan_filter(dset, group=GMETADATA, f=16)        #5648
dset = greater_filter(dset, group=GMETADATA, f=16)        #4214

# Metada artist_familiarity
dset = nan_filter(dset, group=GMETADATA, f=2)       #4214
# dset = greater_filter(dset, group=GMETADATA, f=2)   #4205
# Metada artist_hotttnesss
dset = nan_filter(dset, group=GMETADATA, f=3)       #4205

# Analysis duration
dset = nan_filter(dset, group=GANALYSIS, f=3)       #4205
# dset = greater_filter(dset, group=GANALYSIS, f=3)   #4205

# Analysis end_of_fade_in
dset = nan_filter(dset, group=GANALYSIS, f=4)       #4205

# Analysis year
# dset = greater_filter(dset, group=GMUSICBRZ, f=1)  #2712

# xdset = dset_h
xdset = dset
# Analysis key to time_signature_confidence
for i in range(21, 30):
    xdset = nan_filter(xdset, group=GANALYSIS, f=i)     #2710

original.close()

# Return two three major groups
sdataset=np.array(list(zip(*xdset)))
print("Ssize", xdset.shape), 

# New dataset
print("Creating secure dataset")

secure_filename="./data/data.h5"
shf = h5py.File(secure_filename, 'w')

shf.create_dataset('metadata', data=sdataset[0].astype(metadata_dt))
shf.create_dataset('analysis', data=sdataset[1].astype(analysis_dt))
shf.create_dataset('musicbrainz', data=sdataset[2].astype(musicbrz_dt))

shf.close()
print("Done")