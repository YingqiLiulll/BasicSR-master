import numpy as np
x = np.random.random((4,5))
print(x)
print("\n")
def Dropout(x,drop_proba):
    return x*np.random.choice(
        [0,1],
        x.shape,
        p = [drop_proba,1-drop_proba]
    ) / (1 - drop_proba)
print(Dropout(x,0.5))