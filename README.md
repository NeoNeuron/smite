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

```
