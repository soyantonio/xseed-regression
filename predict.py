import h5py
import numpy as np
import sys

nargs = len(sys.argv)

# params = np.array([0.4573910008766597, 0.01478932283616461, 0.0041640300572309205, 0.05431737498443852, 0.03555889458119846])

# params = np.array([0.4573630101189306, 0.1292828312173423, 0.39530074217900296, 0.30444812263952703])

# params = np.array([0.4558686860185034, 0.0025828053816397503, 0.03664176149069534, 0.010665929995266167, 0.049812482151862944, 0.03114144055146752])

# 25000 epochs, 0.001 learning_rate
params = np.array([4.5689989205777737e-01, 5.4572800635885434e-03, 3.4212943936733034e-02, 1.3203919317690549e-02, -4.3672212169919176e-03, 2.6600315619077974e-03, 4.1248291193673103e-04, 4.6245504699725826e-02 , 3.4646232543542800e-02])


# TRBHLIH128F1473B78 TRAKBHG12903CDA83B TRASOHG128F4287A68

tracks = ['TRAHTZN128F421A41F'] if nargs < 2 else sys.argv[1:]

def predictor(track):
    print("\nAnalyzing track", track)
    track_id = track.encode() 
    filename=b"/mnt/d/antoniospace/schoolspace/ai_projects/xseed-regression/data/easy.h5"

    original = h5py.File(filename, 'r')
    dataset = np.array(original['attributes'])
    real = np.array(original['hotttnesss'])
    idset = np.array(original['tracks_id'])
    original.close()

    result, = np.where(idset == track_id)
    if(result.size < 1):
        print("Can not found", track_id)
        return 1

    index = result[0]
    y = real[index]
    hyp = np.dot(params, dataset[index])
    print("Located at", index)
    print("Song hotttest", y)
    print("Prediction", hyp)
    print("Error", abs(y-hyp)*100.0/y, "%")

for track_id in tracks:
    predictor(track_id)
