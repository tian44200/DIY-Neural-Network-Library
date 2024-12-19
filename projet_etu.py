import numpy as np
import copy
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm, trange
import time
from IPython import display


class Loss(object):
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)       
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass



# ! Partie 1: Classe Lineaire 

class Linear(Module):
    """
    La classe Linear est une implémentation d'un module linéaire pour un réseau de neurones.
    """
    def __init__(self, n, d , bias=True,type="he_normal"):
        self._n = n
        self._d = d

        if type == "random":
            self._parameters = np.random.random((n, d))
        elif type == "normal":
            self._parameters = np.random.normal(0, 1, (n, d))
        elif type == "he_normal":
            self._parameters = np.random.normal(0, np.sqrt(2 / n ), (n, d))
        elif type=="xavier":
            std_dev = np.sqrt(2 / (n + d))
            self._parameters = np.random.normal(
                0, std_dev, (n, d)
            )

        self._gradient = np.zeros_like(self._parameters)

        if bias:
            if type == "random":
                self._bias = np.random.random((1, d))
            elif type == "normal":
                self._bias = np.random.normal(0, 1, (1, d))
            elif type == "he_normal":  
                self._bias =  np.random.normal(0, np.sqrt(2 / n ), (1, d))
            elif type=="xavier":
                std_dev = np.sqrt(2 / (n + d))
                self._bias = np.random.normal(
                    0, std_dev, (1, d)
                )

            self._gradient_bias = np.zeros_like(self._bias)
        else:
            self._bias = None
            self._gradient_bias = None
    
    def zero_grad(self):
        self._gradient = np.zeros((self._n,self._d))
        if self._bias is not None:
            self._gradient_bias = np.zeros((1,self._d))

    def forward(self, X):
        assert X.shape[1] == self._n
        if self._bias is not None:
            return np.dot(X,self._parameters) + self._bias
        else:
            return np.dot(X,self._parameters)
    
    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step*self._gradient
        if self._bias is not None:
            self._bias -= gradient_step*self._gradient_bias
    
    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self._n
        assert delta.shape[1] == self._d
        assert input.shape[0] == delta.shape[0]

        self._gradient += np.dot(input.T,delta)
        if self._bias is not None:
            self._gradient_bias += np.sum(delta,axis=0)

    def backward_delta(self, input, delta):
        assert input.shape[1] == self._n
        assert delta.shape[1] == self._d
        return np.dot(delta,self._parameters.T)

class MSELoss(Loss):
    """
    La classe MSELoss est une implémentation de la fonction de perte de l'erreur quadratique moyenne (MSE).
    """
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return np.sum((y-yhat)**2, axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -2*(y-yhat)

###################################
# ! Partie 2: Classe non lineaire #
###################################

class TanH(Module):
    """
    La classe TanH est une implémentation d'un module d'activation de la fonction tangente hyperbolique.
    """
    def forward(self, X):
        return np.tanh(X)
    
    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input)**2)

class Sigmoid(Module):
    """
    La classe Sigmoid est une implémentation d'un module d'activation de la fonction sigmoïde.    
    """
    def forward(self, X):
        return 1/(1+np.exp(-X))

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        return delta*self.forward(input)*(1-self.forward(input))


###################################
#     ! Partie 3 : Encapsulage    #
###################################

class Sequential:
    """
    La classe Sequential est une implémentation d'un réseau de neurones séquentiel.
    """
    def __init__(self, modules, classes_type="0/1"):
        self.modules = modules
        self.inputs_modules = []
        self.classes_type = classes_type

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, X):
        self.inputs_modules = []

        for module in self.modules:
            self.inputs_modules.append(X)
            X = module.forward(X)
        return X

    def backward(self, delta):
        for i in range(len(self.modules) - 1, -1, -1):
            X = self.inputs_modules[i]
            self.modules[i].backward_update_gradient(X, delta)
            delta = self.modules[i].backward_delta(X, delta)

    def update_parameters(self, eps=1e-3):
        for module in self.modules:
            module.update_parameters(gradient_step=eps)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def predict(self, X):
        ypred = self(X)

        if self.classes_type == "multi":
            return np.argmax(ypred, axis=1, keepdims=True)

        neg_class = -1
        th = 0
        if self.classes_type == "0/1":
            neg_class = 0
            th = 0.5

        return np.where(ypred < th, neg_class, 1)

    def score(self, X, y):
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if y.shape[1] > 1:
            y = np.argmax(y, axis=1, keepdims=True)

        yhat = self.predict(X)
        return (yhat == y).mean()
    
class Optim:
    """
    La classe Optim est une implémentation d'un optimiseur pour un réseau de neurones.
    """
    def __init__(self, net, loss=MSELoss(), eps=1e-5):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.train_loss = []
        self.test_loss = []

        self.train_score = []
        self.test_score = []

    def step(self, X, y):
        yhat = self.net(X)

        delta = self.loss.backward(y, yhat)
        self.net.zero_grad()
        self.net.backward(delta)
        self.net.update_parameters(self.eps)

    def SGD(self, X, y, batch_size, epochs, test_train_split=False, verbose=False):
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if test_train_split:
            X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

        if batch_size > len(X):
            batch_size = len(X)

        self.train_loss = []
        self.test_loss = []

        self.train_score = []
        self.test_score = [] 

        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        for epoch in tqdm(range(epochs)):
            loss_epoch = 0

            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            X_batchs = np.array_split(X, n_batches)
            y_batchs = np.array_split(y, n_batches)

            for X_batch, y_batch in zip(X_batchs, y_batchs):
                self.step(X_batch, y_batch)

            loss_epoch /= n_batches

            yhat = self.net(X)
            loss_epoch = self.loss(y, yhat).mean()
            self.train_loss.append(loss_epoch)

            score_epoch = self.net.score(X, y)
            self.train_score.append(score_epoch)

            if test_train_split:
                y_test_hat = self.net(X_test)
                loss_value = self.loss(y_test, y_test_hat).mean()
                self.test_loss.append(loss_value)

                score_epoch = self.net.score(X_test, y_test)
                self.test_score.append(score_epoch)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - Loss: {loss_epoch} - Score: {score_epoch}"
                )

        print("Training completed.")


class CELogSoftmax(Loss):
    """
    La classe CELogSoftmax est une implémentation de la fonction de perte de l'entropie croisée logistique.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return (-y * yhat).sum(axis=1) + np.log(np.exp(yhat).sum(axis=1))

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return -y + np.exp(yhat) / np.exp(yhat).sum(axis=1).reshape(-1, 1)

###################################
# ! Partie 4: Softmax             #
###################################

# Softmax
class Softmax(Module):
    """
    La classe Softmax est une implémentation d'un module d'activation de la fonction softmax.
    """

    def forward(self, X):
        #pass forward 
        e = np.exp(X)
        return e / np.sum(e, axis=1).reshape((-1, 1))

    def backward_delta(self, input, delta):
        #backward, pour la propagation
        e = np.exp(input)
        outh = e/ np.sum(e, axis=1).reshape((-1, 1))
        return delta * (outh * (1 - outh))

    def update_parameters(self, gradient_step=1e-3):
        pass


###################################
#        ! Partie 5               #
###################################

class BCELoss(Loss):
    """
    La classe BCELoss est une implémentation de la fonction de perte de l'entropie croisée binaire.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert(y.shape == yhat.shape)
        return - (y*np.log(yhat + 1e-4) + (1-y)*np.log(1-yhat+ 1e-4))
    
    def backward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert(y.shape == yhat.shape)
        return ((1-y)/(1-yhat+ 1e-4)) - (y/yhat+ 1e-4)
    

###################################
#        ! Partie 6               #
###################################

# Conv1D

class Conv1D(Module):
    """
    La classe Conv1D est une implémentation d'un module de convolution 1D pour un réseau de neurones.
    """

    def __init__(self, k_size, chan_in, chan_out, stride=1, bias=True):
        """
        k_size : kernel size
        chan_in : number of channels (in)
        chan_out : number of channels (in)

        """
        super(Conv1D, self).__init__()
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        b=1 / np.sqrt(chan_in*k_size)
        self._parameters = np.random.uniform(-b, b, (k_size,chan_in,chan_out))
        self._gradient=np.zeros(self._parameters.shape)
        self.bias = bias
        if(self.bias):
            self._bias=np.random.uniform(-b, b, chan_out)
            self._gradBias = np.zeros((chan_out))

    def zero_grad(self):
       
        self._gradient=np.zeros(self._gradient.shape)
        if (self.bias):
            self._gradBias = np.zeros(self._gradBias.shape)

    def forward(self, X):
        """
       
        X: (batch,input,chan_in)
        out: (batch, (input-k_size)/stride +1,chan_out)
        """
        size = ((X.shape[1] - self.k_size) // self.stride) + 1

        out=np.array([(X[:, i: i + self.k_size, :].reshape(X.shape[0], -1)) @ (self._parameters.reshape(-1, self.chan_out))
                         for i in range(0,size,self.stride)])
        if (self.bias):
            out+=self._bias
        self._forward=out.transpose(1,0,2)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
       
        self._parameters -= gradient_step * self._gradient
        if self.bias:
            self._bias -= gradient_step * self._gradBias

    def backward_update_gradient(self, input, delta):
        """
        input: (batch,input,chan_in)
        delta: (batch, (input-k_size)/stride +1,chan_out)
       
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        out = np.array([ (delta[:,i,:].T) @ (input[:, i: i + self.k_size, :].reshape(input.shape[0], -1))  \
                           for i in range(0, size, self.stride)])
        self._gradient=np.sum(out,axis=0).T.reshape(self._gradient.shape)/delta.shape[0]

        if self.bias:
            self._gradBias=delta.mean((0,1))

    def backward_delta(self, input, delta):
        """
        input: (batch,input,chan_in)
        delta: (batch, (input-k_size)/stride +1,chan_out)
        out: (batch,input,chan_in)
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        out = np.zeros(input.shape)
        for i in range(0, size, self.stride):
            out[:,i:i+self.k_size,:] += ((delta[:, i, :]) @ (self._parameters.reshape(-1,self.chan_out).T)).reshape(input.shape[0],self.k_size,self.chan_in)
        self._delta= out
        return self._delta

class MaxPool1D(Module):
    """
    La classe MaxPool1D est une implémentation d'un module de pooling 1D pour un réseau de neurones.
    """

    def __init__(self, k_size=3, stride=1):
        super(MaxPool1D, self).__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        """
        X: (batch,input,chan_in)
        out:  (batch,(input-k_size)/stride +1,chan_in)
        """

        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        out = np.zeros((X.shape[0], size, X.shape[2]))
        for i in range(0, size, self.stride):
            out[:,i,:]=np.max(X[:,i:i+self.k_size,:],axis=1)
        self._forward=out
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        """
         input: (batch,input,chan_in)
         delta: (batch,(input-k_size)/stride +1,chan_in)
         out: (batch,input,chan_in)
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        out=np.zeros(input.shape)
        batch=input.shape[0]
        chan_in=input.shape[2]
        for i in range(0,size,self.stride):
            indexes_argmax = np.argmax(input[:, i:i+self.k_size,:], axis=1) + i
            out[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,:].reshape(-1)
        self._delta=out
        return self._delta
    
class Flatten(Module):
    """
    La classe Flatten est une implémentation d'un module de flattening pour un réseau de neurones.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        """
        X:(batch,input,chan_in)
        out:(batch,input*chan_in)
        """
        self._forward = X.reshape(X.shape[0], -1)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        """
         input: (batch,input,chan_in)
         delta: (batch, input * chan_in)
         out: (batch,input,chan_in)
        """
        self._delta = delta.reshape(input.shape)
        return self._delta
    

class Conv2D(Module):
    """
    La classe Conv2D est une implémentation d'un module de convolution 2D pour un réseau de neurones.
    """
    def __init__(self, k_size, chan_in, chan_out, stride=1, bias=True):
        Module.__init__(self)
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        b = 1 / np.sqrt(chan_in * k_size * k_size)
        self._parameters = np.random.uniform(-b, b, (k_size, k_size, chan_in, chan_out))
        self._gradient = np.zeros(self._parameters.shape)
        self.bias = bias
        if self.bias:
            self._bias = np.random.uniform(-b, b, chan_out)
            self._gradBias = np.zeros((chan_out))

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
        if self.bias:
            self._gradBias = np.zeros(self._gradBias.shape)

    def forward(self, X):
        """
        X: (batch, H, W, chan_in)
        out: (batch, size_H, size_W, chan_out)
        """
        batch_size, H, W, _ = X.shape
        size_H = (H - self.k_size) // self.stride + 1
        size_W = (W - self.k_size) // self.stride + 1
        out = np.zeros((batch_size, size_H, size_W, self.chan_out))

        for i in range(size_H):
            for j in range(size_W):
                out[:, i, j, :] = np.tensordot(
                    X[:, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, :],
                    self._parameters, axes=([1, 2, 3], [0, 1, 2])
                )
        if self.bias:
            out += self._bias
        self._forward = out
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient
        if self.bias:
            self._bias -= gradient_step * self._gradBias

    def backward_update_gradient(self, input, delta):
        batch_size, H, W, _ = input.shape
        size_H = (H - self.k_size) // self.stride + 1
        size_W = (W - self.k_size) // self.stride + 1

        for i in range(size_H):
            for j in range(size_W):
                self._gradient += np.tensordot(
                    delta[:, i, j, :], 
                    input[:, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, :],
                    axes=([0], [0])
                ).transpose(1, 2, 3, 0)
        self._gradient /= batch_size

        if self.bias:
            self._gradBias += delta.sum(axis=(0, 1, 2)) / batch_size

    def backward_delta(self, input, delta):
        batch_size, H, W, _ = input.shape
        size_H = (H - self.k_size) // self.stride + 1
        size_W = (W - self.k_size) // self.stride + 1
        dX = np.zeros_like(input)

        for i in range(size_H):
            for j in range(size_W):
                grad = np.tensordot(
                    delta[:, i, j, :], self._parameters, axes=([1], [3])
                )
                dX[:, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, :] += grad.transpose(0, 2, 1, 3)
        return dX

class MaxPool2D(Module):
    """
    La classe MaxPool2D est une implémentation d'un module de pooling 2D pour un réseau de neurones.
    """
    def __init__(self, k_size=2, stride=2):
        Module.__init__(self)
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        """
        X: (batch, H, W, chan_in)
        out: (batch, t_h, t_w, chan_in)
        """
        batch_size, H, W, C = X.shape
        t_h = (H - self.k_size) // self.stride + 1
        t_w = (W - self.k_size) // self.stride + 1
        out = np.zeros((batch_size, t_h, t_w, C))
        for i in range(t_h):
            for j in range(t_w):
                out[:, i, j, :] = np.max(
                    X[:, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, :],
                    axis=(1, 2)
                )
        self._forward = out
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        batch_size, H, W, C = input.shape
        t_h = (H - self.k_size) // self.stride + 1
        t_w = (W - self.k_size) // self.stride + 1
        dX = np.zeros_like(input)
        for i in range(t_h):
            for j in range(t_w):
                patch = input[:, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, :]
                max_val = np.max(patch, axis=(1, 2), keepdims=True)
                mask = (patch == max_val)
                dX[:, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, :] += mask * delta[:, i, j, :][:, None, None, :]
        return dX


class ReLU(Module):
    """
    La classe ReLU est une implémentation d'un module d'activation de la fonction ReLU.
    """
    def __init__(self,threshold=0.):
     
        self._threshold=threshold

    def update_parameters(self, gradient_step=1e-3):
        pass

    def forward(self, X):
        self._forward=self.threshold(X)
        return self._forward

    def threshold(self,input):
        return np.where(input>self._threshold,input,0.)


    def derivative_Threshold(self,input):
        #Batch x out
        #np.where(self.threshold(input)<=self._threshold,0.,1.)
        return (input > self._threshold).astype(float)

    def backward_delta(self, input, delta):
        self._delta=np.multiply(delta,self.derivative_Threshold(input))
        return self._delta
    

class CELoss(Loss):
    """
    La classe CELoss est une implémentation de la fonction de perte de l'entropie croisée.
    """
    
    def forward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert(y.shape == yhat.shape)
        return 1 - np.sum(yhat * y, axis = 1)
    
    def backward(self, y, yhat):
        #params passé en entrée sont de la bonne taille 
        assert(y.shape == yhat.shape)  
        return yhat-y
