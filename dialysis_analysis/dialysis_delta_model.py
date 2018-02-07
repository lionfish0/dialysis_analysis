import numpy as np
from datetime import datetime,timedelta
import pandas as pd
import GPy
np.set_printoptions(precision=2,suppress=True)
import matplotlib.pyplot as plt

veryverbose = False
verbose = True

class DeltaModel():
    def __init__(self,prophets,boxstds = 0.1,kerntype='RBF'):
        regions = prophets[0].regions
        deltaregions = prophets[0].deltaregions
        deltamodels = []
        for region in range(regions):
            deltaX = []
            deltaY = []
            for pro in prophets:
                if not hasattr(pro,'res'):
                    if veryverbose: print("Skipping prophet as it has not had its results computed")
                    continue
                error = pro.get_actual() - pro.res['mean'][:,0]
                delta_value = np.array(pro.res['delta_values'])
                if ~np.isnan(error[region]):
                    deltaX.append(delta_value)
                    deltaY.append(error[region])


            middle = np.mean(deltaX,0)*0 #*0 to move middle to axis origin
            keep = np.any((deltaX<middle-boxstds*np.std(deltaX,0)) | (deltaX>middle+boxstds*np.std(deltaX,0)),1)
            deltaX = np.array(deltaX)
            deltaY = np.array(deltaY)[:,None]

            deltaX = deltaX[keep,:]
            deltaY = deltaY[keep,:]
            #TODO NORMALISE deltaX so we can make ARD=False??!?
            if kerntype=='RBF':
                k = GPy.kern.RBF(deltaregions,ARD=True)
            if kerntype=='linear':
                k = GPy.kern.Linear(deltaregions)


            m = GPy.models.GPRegression(deltaX,deltaY,k)
            m.optimize()
            deltamodels.append(m)
        self.deltamodels = deltamodels
        self.prophets = prophets
        
    def plotmodel(self,m,inputdim,Nsteps = 100,bounds=None):
        """
        Plots a GPy model, the size of the dots are scaled by the distance from the plane that is being plotted
        """
        testX = np.repeat(np.mean(m.X,0)[None,:],Nsteps,0)
        leftX = m.X.copy()
        np.delete(leftX,inputdim,1)
        #invsqrdist = 1/(np.sum((leftX - np.mean(leftX))**2,1))
        leftX /= np.std(leftX,0)
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

        plt.plot(testvals,predmeans)
        plt.plot(testvals,predmeans-1.96*np.sqrt(predvars))
        plt.plot(testvals,predmeans+1.96*np.sqrt(predvars))
        plt.scatter(m.X[:,inputdim],m.Y,invsqrdist)
        
        
    def plot(self):
        mins = []
        maxs = []
        for m in self.deltamodels:
            mins.append(np.min(m.X,0))
            maxs.append(np.max(m.X,0))
        mins = np.min(np.array(mins),0)
        maxs = np.max(np.array(maxs),0)
        diffs = maxs-mins
        mins-=diffs/10
        maxs+=diffs/10
        p = self.prophets[0]
        regnames = p.outputdialysis.copy()
        regnames.extend(p.outputlab)
        figi = 0
        l = [dd[0]+'('+dd[1]+')' for dd in p.delta_dialysis]
        l.extend([dl[0]+'('+dl[1]+')' for dl in p.delta_lab])    
        for dm,name in zip(self.deltamodels,regnames):
            for i,reg in enumerate(l):
                figi+=1
                plt.subplot(len(self.deltamodels),len(l),figi)
                self.plotmodel(dm,i,bounds=[mins[i],maxs[i]])
                plt.title("Error in %s" % name)
                plt.xlabel(reg)   
                
    def add_delta_to_prophets(self,prophets):
        """
        Alter the mean and variance 'res' predictions for all the prophets in the passed list.
        The old values are saved in the uncorrected_mean and uncorrected_var values in res, for each prophet.
        A flag 'corrected' is also added, so you know the mean and var have been updated with the deltamodel.
        This DeltaModel instance is pointed to by the corrected field, for future reference.
        """
        suppressduplicatemsgs = False
        
        assert prophets[0].delta_dialysis == self.prophets[0].delta_dialysis, "The dialysis delta parameters in the prophets being updated differ from those used to create the model"
        assert prophets[0].delta_lab == self.prophets[0].delta_lab, "The lab delta parameters in the prophets being updated differ from those used to create the model"
        
        for prophet in prophets:
            if 'corrected' in prophet.res:
                if not suppressduplicatemsgs: print("Warning: Attempting to delta-correct already corrected prophet(s), skipping.")
                suppressduplicatemsgs = True
                continue
            prophet.res['corrected'] = self
            prophet.res['uncorrected_mean'] = prophet.res['mean'].copy()
            prophet.res['uncorrected_var'] = prophet.res['var'].copy()

            
            for i,dm in enumerate(self.deltamodels):
                assert len(prophet.res['delta_values'])==dm.X.shape[1], "The delta_values you..." 
                deltapred, deltavar = dm.predict_noiseless(np.array(prophet.res['delta_values'])[None,:])

                prophet.res['mean'][i] += deltapred[0,0]
                #we assume that the predicted variance and delta variance are independent and are both normally distributed.
                #so we can add the variances together
                prophet.res['var'][i] += deltavar[0,0]                