import numpy as np

# gradient function = 1 / 1 + e^(-x)
# gradient derivative = x * (1 - x)
def gradient(z, deriv=False):
    if (deriv==True):
        return z*(1-z)
    return 1/(1+np.exp(-z))

def loss(t, o, deriv=False):
    if (deriv==True):
        return -(t-o)
    return 0.5*np.square(t-o)
    
x = np.array([[0,0],[0,1],[1,0],[1,1]])

y = np.array([[0],[1],[1],[0]])

# Seed random numbers for weights 
np.random.seed(4)

# Initialize weights randomly with mean 0
# The first matrix is 2 x 4 - 2 weights for each activation with 4 total results
syn0 = 2 * np.random.random((2,4)) - 1

# The second is 4 x 1 - 4 weights for each activation with 1 result (the guess)
syn1 = 2 * np.random.random((4,1)) - 1

# Learning rate
alpha = 1

# Initial results
hn = np.dot(x,syn0)
ho = gradient(hn)
    
on = np.dot(ho,syn1)
oo = gradient(on)

print "Output before training"
print oo

# Try 10000 times for learning
for i in xrange(60000):
    
    hn = np.dot(x,syn0)
    ho = gradient(hn)
    
    on = np.dot(ho,syn1)
    oo = gradient(on)
    
    # Calculate the error between guesses and expected result
    E = loss(y,oo)
    E_total = np.sum(E)
    
    if i % 10000 == 0:
        #print "Weights at " + str(i) + ":\n" + str(syn0) + "\n" + str(syn1) + "\n"
        print "Total error at " + str(i) + ": " + str(E_total)
    
    # Next calculate how much a change in syn1 affects the
    # total Error, ie the partial derivatives of E with
    # respect to the weights in syn1. We can use the chain
    # rule to determine this (hence our "gradienting" function)
    # must have a derivative.
    
    # First the derivatives of E with respect
    # to oo (how much does the Error change with
    # respect to the final output of the output layer?)
    dE_doo = loss(y,oo,True)
    
    # Next, the derivatives of oo with respect to 
    # on (how much does the output of the output layer change   
    # with respect to its total net (ie raw) input?)
    doo_don = gradient(oo, True)
    
    # Last, the derivatives of on with respect to
    # syn1 (how much does the total net input of the output
    # layer change with respect to the weights of syn1?)
    don_syn1 = ho
    
    # Putting all the derivatives together ie the chain rule
    dE_dsyn1 = np.dot(np.transpose(don_syn1), (np.multiply(dE_doo, doo_don)))
    
    # Now we need to calculate how much a change in syn0 affects
    # the total Error, ie the partial derivatives of E with respect
    # to the weights in syn0. Again we cna use the chain rule.
    
    # First, the derivatives of E with respect to ho
    # (how much does the Error change with respect 
    # to the output of the hidden layer?)
    dE_dho = np.dot(np.multiply(dE_doo, doo_don), np.transpose(syn1))
    
    # Next, the derivatives of ho with respect to hn
    # (how much does the output of the hidden layer change
    # with respect to the input of the hidden layer?)
    dho_dhn = gradient(ho, True)
    
    # Last, the derivatives of hn with respect to syn0
    # (how much does the input of the hidden layer 
    # change with respect to the weights of syn0?)
    dhn_dsyn0 = x 
    
    # Putting all the derivatives together ie the chain rule
    dE_dsyn0 = np.dot(np.transpose(dhn_dsyn0), np.multiply(dE_dho, dho_dhn))
    
    # Now update the weights
    syn1 -= alpha * dE_dsyn1
    syn0 -= alpha * dE_dsyn0
    
print "Ouput after training"    
print oo

    
    