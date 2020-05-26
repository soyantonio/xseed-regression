import numpy as np
import h5py

filename= "./data/data.h5"
original = h5py.File(filename, 'r')
dataset_a = np.array(original['analysis'])
dataset_m = np.array(original['metadata'])
dataset_z = np.array(original['musicbrainz'])
original.close()


dset = np.empty([dataset_a.size, 9])
# dset = np.empty([dataset_a.size, 8])
yset = np.empty([dataset_a.size])
iset = np.empty([dataset_a.size], dtype=dataset_a[0]['track_id'].dtype)


for i, v in enumerate(dataset_a):
    j = 0
    dset[i][j] = 1
    # Can be 0.000219
    # j +=1
    # dset[i][j] = dataset_a[i]['duration']
    
    # Can be 1.487-5
    # j +=1 
    # dset[i][j] = dataset_a[i]['end_of_fade_in']
    
    # Can be 2.235-5
    # j +=1
    # dset[i][j] = dataset_a[i]['key']

    # Can be 0.00159403
    j +=1
    dset[i][j] = int(dataset_a[i]['audio_md5'], 16)

    # Can be 0.104254
    j +=1
    dset[i][j] = dataset_z[i]['year']

    # Can be 0.000941
    # j +=1
    # dset[i][j] = dataset_a[i]['key_confidence']
    
    # Can be 0.04023 - 0.2669557
    j +=1
    dset[i][j] = dataset_a[i]['loudness']
    
    # Can be 0.001159
    j +=1
    dset[i][j] = dataset_a[i]['mode']
    
    # Can be 0.0002866
    # j +=1
    # dset[i][j] = dataset_a[i]['mode_confidence']

    # Can be 0.00020294
    # j +=1
    # dset[i][j] = dataset_a[i]['start_of_fade_out']

    # Can be 0.0041717
    j +=1
    dset[i][j] = dataset_a[i]['tempo']

    # Can be 0.001147
    j +=1
    dset[i][j] = dataset_a[i]['time_signature']

    # Can be 0.000242
    # j +=1
    # dset[i][j] = dataset_a[i]['time_signature_confidence']
    
    # Can be 0.259
    j +=1
    dset[i][j] = dataset_m[i]['artist_familiarity']

    # Can be 0.23637 - 0.27669
    j +=1
    dset[i][j] = dataset_m[i]['artist_hotttnesss']

    yset[i] = dataset_m[i]['song_hotttnesss']
    iset[i] = dataset_a[i]['track_id']

helper = dset.T
print(dset.shape)
print(helper.shape)


# Fit a range
for i, attribute in enumerate(helper):    
    minv = np.min(attribute)
    maxv = np.max(attribute)
    rangev = maxv - minv
    
    if rangev != 0:
        for j, x in enumerate(attribute):
            helper[i][j] = (x)/(rangev)


# Normalize distribution
for i, attribute in enumerate(helper):
    mean = np.mean(attribute)
    std = np.std(attribute)

    if std != 0:
        for j, x in enumerate(attribute):
            helper[i][j] = (x - mean)/std

dset = helper.T



secure_filename="./data/easy.h5"
shf = h5py.File(secure_filename, 'w')
shf.create_dataset('attributes', data=dset)
shf.create_dataset('hotttnesss', data=yset)
shf.create_dataset('tracks_id', data=iset)
shf.close()
print("Done")







# tracks = np.empty([dataset.size], dtype=dict)
# tracks_path=b"/mnt/d/tmp_ml/msd/MillionSongSubset/tracks/"

# def get_track(tid, tpath=tracks_path):
#     track_filename=tpath + tid + b'.h5'
#     track_h5 = h5py.File(track_filename, 'r')
#     track = dict()
#     for k in track_h5['analysis'].keys():
#         track[k] = np.array(track_h5['analysis'][k])
#     track_h5.close()
#     return track

# for i, v in enumerate(dataset):
#     tracks[i] = get_track(dataset[i]['track_id'])