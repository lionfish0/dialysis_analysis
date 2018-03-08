import matplotlib.pyplot as plt
import numpy as np
def plotmodel(m,inputdim,region=None,Nsteps=100,bounds=None,prophet=None):
    """
    Plots a GPy model, the size of the dots are scaled by the distance from the plane that is being plotted
    
    Set region if you want plots for a given region.
    """
    meanpoint = np.mean(m.X,0)
    if region is not None:
        meanpoint[-1]=region
    testX = np.repeat(meanpoint[None,:],Nsteps,0)
    leftX = m.X.copy()
    Y = m.Y.copy()
    np.delete(leftX,inputdim,1)
    #invsqrdist = 1/(np.sum((leftX - np.mean(leftX))**2,1))
    leftX /= (np.std(leftX,0)+0.01)
    invsqrdist = 1+10/(1+np.sum((leftX - np.mean(leftX))**2,1)) #dist=0 -> 10, dist=1 -> 5, dist=9 -> 1.
    #print("Centring around:")
    #print(testX[0,:])
    if bounds is not None:
        minv = bounds[0]
        maxv = bounds[1]
    else:
        minv = np.min(m.X[:,inputdim])
        maxv = np.max(m.X[:,inputdim])
        diff = maxv-minv
        minv -= diff / 10
        maxv += diff / 10

    testvals = np.linspace(minv,maxv,Nsteps)
    testX[:,inputdim] = testvals
    predmeans,predvars = m.predict(testX)

    if prophet is not None:
        if region is not None:
            predmeans = prophet.unnormalise_specific_mean(predmeans,region)
            predvars = prophet.unnormalise_specific_variance(predvars,region)
        else:
            print("Error, can't apply normalisation without specifying region")
            return
    plt.plot(testvals,predmeans)
    plt.plot(testvals,predmeans-1.96*np.sqrt(predvars))
    plt.plot(testvals,predmeans+1.96*np.sqrt(predvars))

    alpha = 10.0/np.sqrt(len(Y))  #e.g. len(Y)=1000 ---> alpha = 0.15
    if alpha>1: alpha=1
    if alpha<0.02: alpha=0.02
    
    if region is not None:
        keep = m.X[:,-1]==region
        if prophet is not None:
            Y = prophet.unnormalise_specific_mean(Y,region)
        plt.scatter(m.X[keep,inputdim],Y[keep],invsqrdist,alpha=alpha)
    else:
        if prophet is not None:
            Y = prophet.unnormalise_specific_mean(Y,region)
        plt.scatter(m.X[:,inputdim],Y,invsqrdist,alpha=alpha)
    return minv, maxv
