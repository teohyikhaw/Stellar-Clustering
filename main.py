import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('star_classification.csv')
features = ['obj_ID','alpha','delta','u','g','r','i','z','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','redshift','plate','MJD','fiber_ID']

x = np.array([df[name] for name in features])
#normalize x here
#x = StandardScaler().fit_transform(x)

xbar = [np.mean(temp) for temp in x]

for i in range(17):
    x[i]=x[i]-xbar[i]

U, S, V = np.linalg.svd(x, full_matrices=True)
Y = U*S
print(U)