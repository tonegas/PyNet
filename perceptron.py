import numpy as np
import matplotlib.pyplot as plt

# x input R^n
inputDim = 2
# y input R
outDim = 1

w = np.zeros([outDim,inputDim+1]) #np.random.rand(outDim,inputDim+1)
alfa = 0.1

dataVett = [[0,1],[0,0],[1,1],[1,0],[5,5],[-5,5]]
targetVett = [1,0,1,1,1,1]
cVett=lambda t: 'r' if t>0 else 'b'
dVett=lambda t: 20 if t>0 else 5

epochs = 5000
errors = []
outputs = []
wH = []

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def siggrad(x):
    return sigmoid(x)*(1.0-sigmoid(x))
    # return w.dot(np.exp(w.dot(x)))/np.power(np.exp(w.dot(x))+1,2)


for epoch in xrange(epochs):
    outputs = []
    for x,t in zip(dataVett,targetVett):
        x1 = np.hstack([1,x])
        #step 1
        y1 = w.dot(x1)
        #step 2
        #y = np.sign(y)
        y2 = sigmoid(y1)
        outputs.append(y2)
        #step 3
        err = t-y2
        errors.append(err)
        grad = siggrad(y1)
        #grad = x
        #grad = (y2*(1-y2))
        aw = alfa * x1 * err * siggrad(y1)
        wH.append(w.tolist())
        w = w + aw

plt.subplot(131)
plt.scatter([x[0] for x in dataVett],[x[1] for x in dataVett],c=[cVett(t) for t in outputs],s=[dVett(t) for t in targetVett])
plt.subplot(132)
plt.plot(range(len(errors)),errors)
plt.subplot(133)
plt.plot(range(len(wH)),[x[0] for x in wH])
plt.show()










