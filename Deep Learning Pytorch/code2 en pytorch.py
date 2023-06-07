import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

#import torchmetrics
#note personnelle : il faut utiliser la commande python et non python3.8


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #définition des fonctions utilisées dans le forward
        self.hidden = torch.nn.Linear(100, 32)
        self.output = torch.nn.Linear(32, 100)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.sigmoid(self.hidden(x)) #calcul première couche (entrée)
        output = self.sigmoid(self.output(hidden)) #calcul seconde couche (sortie)
        return output




# Effectuer une prédiction sur une entrée de dimension (batch_size, input_size)
X = torch.Tensor([
    [-0.94957536, 0.09138459, 0.93214973, 0.20734231, -0.36750218, 0.2532073,
-0.40867925, -0.27450909, 0.24224991, 0.158872, 0.17984262, 0.19490678,
-0.18640787, 0.12157863, 0.49287043, -0.81607272, -0.88256159, -0.90136274,
-0.44207091, 0.63318667, 0.74738819, 0.07947606, 0.83076047, -0.28699936,
0.91209859, 1.05235514, 0.62194713, 0.35809164, -0.03880452, -0.16849662,
0.18452538, -0.33911143, 0.10167817, -0.24980872, 0.3198604, -0.97315797,
-0.73097704, 0.25473331, -0.21290872, -0.27834832, 1.16614005, 0.19060984,
0.54409779, 0.35966887, 0.89607094, -0.1892848, -0.46440441, 0.46542725,
-0.02494581, 0.75084969, -0.02786858, -0.05339409, 0.90718408, -0.40622936,
-0.74854102, 0.36512093, 0.10964665, -0.17507169, -0.92614192, 0.25215638,
0.2540514, -0.71813367, -0.21920932, 0.20066063, -0.45295043, -0.26890899,
-0.53262577, -0.12304334, -0.14706275, 0.18244661, -0.76217167, -1.09814276,
1.03134551, 0.28932654, 0.33985089, -0.31844734, -0.78830944, 0.14875111,
-0.34621142, -0.24788535, -1.03522483, -0.01059175, -0.25693921, 0.91169518,
0.14499169, -0.22702077, 0.18360624, 0.06782759, -0.00733005, -0.31198216,
-0.03575288, -0.23718862, -0.84991065, -0.20797388, 0.53654176, 0.16885665,
-0.26702753, 0.1359203, 0.82087304, 0.61304799],
 [-0.12813035, -0.23000817, 0.18081215, 0.10226083, -0.01327843, -0.05203713, 
  0.73225992, -0.01129157, 0.19336875, -0.1382192, 0.21222537, -0.378387,
-0.20091598, 1.15057745, 0.8082293, -0.52736928, 0.45011628, 0.06675095, 
-0.86978873, 0.8650562, -0.67652354, 0.2995841, 0.55536396, 0.14986059, 
-0.71548415, -0.39770103, 0.66363424, -0.21190579, -0.04479644, 0.09246966, 
0.26943691, -0.37541346, -0.27383157, 0.26103118, 0.08111827, -0.24370005, 
0.79925201, 0.24617894, 0.44389159, -0.00448651, 0.08892456, 0.21286408, 
-0.97343377, -1.02284615, 0.52314367, 1.1393919, 1.07276126, -0.88994615, 
-1.16013548, 0.53400186, -0.2768946, 0.20833059, 0.14907219, -0.95198749, 
-0.5975383, -0.11304964, 1.0938811, -1.12779988, 0.35762079, 0.13782711, 
1.00316909, -0.65838026, 0.35727189, 0.08256359, -0.25828402, -0.31417928, 
-0.74551749, -0.37467242, 0.91388133, -0.49524183, 0.65493707, -0.50010396, 
-0.10116975, -0.19339448, -0.06559569, 0.12536031, 0.64013156, -0.11399355, 
0.12376982, -0.07589191, 0.08194666, -0.87830074, -0.13619794, -0.37699514, 
-1.33376406, 0.42786757, 0.227532, 0.86000564, -0.31734599, 0.29011826, 
-0.37579939, -1.01857636, 0.41878948, -0.40463448, 0.92245093, -0.05156667, 
0.16465775, 0.24870812, -0.26971282, -0.72820082]])
y=torch.Tensor([
    0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1,
 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0
])


# Initialisation du modèle
model = Net()

# Fonction de coût
criterion = nn.CrossEntropyLoss() #fonction de coût : logloss

# Optimiseur (descente de gradient stochastique avec un taux d'apprentissage de 0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Boucle d'apprentissage
train_acc=[]
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
   
    # Calcul de la perte
    loss = criterion(y_pred[1], y)
    #on prend la secondes coordonnée de y_pred car cela correspond au résultat de la deuxième fonction d'activation : la sortie
    #(il me semble)
    
    Ny=np.array([int(i) for i in y]) #car tensor.numpy() ne fonctionne pas pour une raison inconnue
    NyPred=np.array([int(i >= .5) for i in y_pred[1]])
    train_acc.append(accuracy_score(Ny, NyPred))

    # Backward pass et mise à jour des poids
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

aX=np.array([[-0.94957536, 0.09138459, 0.93214973, 0.20734231, -0.36750218, 0.2532073,
-0.40867925, -0.27450909, 0.24224991, 0.158872, 0.17984262, 0.19490678,
-0.18640787, 0.12157863, 0.49287043, -0.81607272, -0.88256159, -0.90136274,
-0.44207091, 0.63318667, 0.74738819, 0.07947606, 0.83076047, -0.28699936,
0.91209859, 1.05235514, 0.62194713, 0.35809164, -0.03880452, -0.16849662,
0.18452538, -0.33911143, 0.10167817, -0.24980872, 0.3198604, -0.97315797,
-0.73097704, 0.25473331, -0.21290872, -0.27834832, 1.16614005, 0.19060984,
0.54409779, 0.35966887, 0.89607094, -0.1892848, -0.46440441, 0.46542725,
-0.02494581, 0.75084969, -0.02786858, -0.05339409, 0.90718408, -0.40622936,
-0.74854102, 0.36512093, 0.10964665, -0.17507169, -0.92614192, 0.25215638,
0.2540514, -0.71813367, -0.21920932, 0.20066063, -0.45295043, -0.26890899,
-0.53262577, -0.12304334, -0.14706275, 0.18244661, -0.76217167, -1.09814276,
1.03134551, 0.28932654, 0.33985089, -0.31844734, -0.78830944, 0.14875111,
-0.34621142, -0.24788535, -1.03522483, -0.01059175, -0.25693921, 0.91169518,
0.14499169, -0.22702077, 0.18360624, 0.06782759, -0.00733005, -0.31198216,
-0.03575288, -0.23718862, -0.84991065, -0.20797388, 0.53654176, 0.16885665,
-0.26702753, 0.1359203, 0.82087304, 0.61304799],
 [-0.12813035, -0.23000817, 0.18081215, 0.10226083, -0.01327843, -0.05203713, 
  0.73225992, -0.01129157, 0.19336875, -0.1382192, 0.21222537, -0.378387,
-0.20091598, 1.15057745, 0.8082293, -0.52736928, 0.45011628, 0.06675095, 
-0.86978873, 0.8650562, -0.67652354, 0.2995841, 0.55536396, 0.14986059, 
-0.71548415, -0.39770103, 0.66363424, -0.21190579, -0.04479644, 0.09246966, 
0.26943691, -0.37541346, -0.27383157, 0.26103118, 0.08111827, -0.24370005, 
0.79925201, 0.24617894, 0.44389159, -0.00448651, 0.08892456, 0.21286408, 
-0.97343377, -1.02284615, 0.52314367, 1.1393919, 1.07276126, -0.88994615, 
-1.16013548, 0.53400186, -0.2768946, 0.20833059, 0.14907219, -0.95198749, 
-0.5975383, -0.11304964, 1.0938811, -1.12779988, 0.35762079, 0.13782711, 
1.00316909, -0.65838026, 0.35727189, 0.08256359, -0.25828402, -0.31417928, 
-0.74551749, -0.37467242, 0.91388133, -0.49524183, 0.65493707, -0.50010396, 
-0.10116975, -0.19339448, -0.06559569, 0.12536031, 0.64013156, -0.11399355, 
0.12376982, -0.07589191, 0.08194666, -0.87830074, -0.13619794, -0.37699514, 
-1.33376406, 0.42786757, 0.227532, 0.86000564, -0.31734599, 0.29011826, 
-0.37579939, -1.01857636, 0.41878948, -0.40463448, 0.92245093, -0.05156667, 
0.16465775, 0.24870812, -0.26971282, -0.72820082]])

ay=np.array([int(i >= .5) for i in y_pred[1]])
plt.scatter(aX[0, :], aX[1, :], c=ay, cmap='summer')
plt.show()

plt.plot(train_acc, label='train acc')
plt.legend()
plt.show()