# Symbolic Mutual Information and Transfer Entropy (smite)

## installation
```bash
git clone https://github.com/NeoNeuron/smite.git
pip install -e .
```

## Simple Example:
```python
import smite
import numpy as np

X = np.random.randint(10, size=3000)
Y = np.roll(X,-1)

symX = smite.symbolize(X,3)
symY = smite.symbolize(Y,3)

MI = smite.symbolic_mutual_information(symX, symY)

TXY = smite.symbolic_transfer_entropy(symX, symY)
TYX = smite.symbolic_transfer_entropy(symY, symX)
TE = TYX - TXY

print("Mutual Information = " + str(MI))
print("T(Y->X) = " + str(TXY))
print("T(X->Y) = " + str(TYX))
print("Transfer of Entropy = " + str(TE))

# compute the STE matrix of a set of time series
data = np.random.randn(3000,3)
STE = smite.symbolic_transfer_entropy_matrix(data, 3, n_jobs=1)
# if njobs > 1, the computation will be parallelized
print(STE)

```
