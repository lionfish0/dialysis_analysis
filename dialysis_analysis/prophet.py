import numpy as np
import pickle
import csv
from datetime import datetime,timedelta
import pandas as pd
pd.options.mode.chained_assignment = None
import GPy
import matplotlib.pyplot as plt
from dask import compute, delayed
from dask.distributed import Client
np.set_printoptions(precision=2,suppress=True)
import dask_dp4gp
import percache
from dialysis_analysis.plotting import plotmodel
from dialysis_analysis import *
cache = percache.Cache("cache") #some of the methods to load patient data are cached
verbose = False
veryverbose = False

def get_params(p,labelstring=""):
    """
    Get a dictionary of the hyperparameters that describe the model.
    
    E.g. get_params(model)
    """
    if len(labelstring)>0:
        labelstring = labelstring + "." + p.name
    else:
        labelstring = p.name
    params = {}
    for param in p.parameters:
        params = {**params, **get_params(param,labelstring)}
    if hasattr(p,'values'):
        params[labelstring] = p.values
    return params

class ProphetException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs) 
        
class Prophet(object):
    def compute_population_priors(self, ms):
        """Given a list of population prior models (that use the same
        demographics vector), compute the priors for this prophet.
        
        Set elements to None for those regions we do not have a prior
        model for (e.g. due to lack of data)"""
       
        if self.frompatient is not None:
            assert ms not in self.frompatient.usedforpopmodel, "The patient this prophet is from has been used to build the population model you are now using to build prophets. This is peeking."
            
        preds = []
        for m in ms:
            if m is not None:
                pred, _ = m.predict(np.array(self.demographics_vector())[None,:])
                preds.append(pred[0,0])
            else:
                preds.append(None)
        return preds

    def compute_normalisation(self,Y,region_indexes,prior_means=None):
        """
        Normalise the data in Y, using the regions in regions.
        Y = data to compute normalisation parameters for
        region_indexes = indicies of the rows of Y
        
        prior_means = use these instead of the means in Y (set to None otherwise)
            pass a list of the means. Set them to None to use the mean of the region's data;
            e.g. prior_means = [10,None,2,0,None]
        
        Example:
            compute_normalisation(Y,X[:,-1])
        the normalisation parameters are stored as instance variables in the object
        
        We require self.regions = number of regions to be set
        """
        
        self.normparams = []
        for region in range(self.regions):
            in_reg = (region_indexes==region)
            if prior_means is None:
                mean=np.mean(Y[in_reg,:])
            else:
                if prior_means[region] is None: #individual elements of prior_means can be none
                    mean=np.mean(Y[in_reg,:])
                else:
                    mean=prior_means[region]

            if len(Y[in_reg,:])>=2:
                std = np.std(Y[in_reg,:])
            else:
                std = np.NaN
            self.normparams.append({'mean':mean,'std':std})
            
            
    def normalise(self,Y,region_indexes):    
        """
        Normalise the values in Y, using the parameters calculated in 'compute_normalisation'
        Example:
            normed_Y = normalise(Y,X[:,-1])
        """
        normalisedY = Y.copy()
        for region in np.unique(region_indexes):
            in_reg = (region_indexes==region)
            if ~np.isnan(self.normparams[int(region)]['std']):
                normalisedY[in_reg,:]-=self.normparams[int(region)]['mean']
                normalisedY[in_reg,:]/=self.normparams[int(region)]['std']
        return normalisedY

    def unnormalise_specific_mean(self,mean,region):
        """
        Unnormalise a single mean value, specified by region.
        Example:
            predictions = unnormalise_specific_mean(regiontwomean,2)
        """
        if ~np.isnan(self.normparams[region]['std']):
            mean*=self.normparams[region]['std']
            mean+=self.normparams[region]['mean']
        return mean
    
    def unnormalise_specific_variance(self,variance,region):
        """
        Unnormalise a single variance value, specified by region.
        Example:
            predictions = unnormalise_specific_mean(regiontwomean,2)
        """
        if ~np.isnan(self.normparams[region]['std']):
            variance*=self.normparams[region]['std']**2
        return variance    
    
    def unnormalise_means(self,normalised_means):
        """
        Unnormalise an array of mean values
        Array should be one for each region.
        Example:
            predictions = unnormalise_means(normalised_predictedmeans)
        """
        means = normalised_means.copy()
        for region in range(self.regions):
            if ~np.isnan(self.normparams[region]['std']):
                means[region]*=self.normparams[region]['std']
                means[region]+=self.normparams[region]['mean']
        return means

    def unnormalise_variances(self,normalised_vars):
        """
        Unnormalise an array of variances.
        Array should be one for each region.
        Example:
            prediction_variances = unnormalise_variances(normalised_predictedvariances)
        
        """
        unnormalised_vars = normalised_vars.copy()
        for region in range(self.regions):
            if ~np.isnan(self.normparams[region]['std']):
                unnormalised_vars[region]*=self.normparams[region]['std']**2
        return unnormalised_vars
    
    
    def __init__(self,X,Y,testX,testY,regions,deltaX=None, deltaY=None, deltaOption=None,prior_means=None,demographics=None,prior_models=None, frompatient=None,inputdialysis=None,outputdialysis=None,outputlab=None,delta_dialysis=None,delta_lab=None):
        """
        A prediction of the dialysis variables.
        X and testX should contain in the first two columns
         vintage and days-since-dialysis, ...
        deltaX and deltaY are the arrays for the delta model
        
        prior_means = a list of the means of the outputs
        """
        
        self.demographics = demographics
        
        self.frompatient = frompatient #lets us track which patients created which prophets
        
        #remove NaNs from Y
        keep = ~np.isnan(Y)[:,0]
        X = X[keep,:]
        Y = Y[keep,:]
        #remove NaNs from X
        keep = np.all(~np.isnan(X),1)
        X = X[keep,:]
        Y = Y[keep,:]
        if len(X)<2:
            raise ProphetException('Fewer than two data points in training data (N=%d)' % len(X))
            
        self.X = X.copy()
        self.regions = regions
        
        if prior_models is not None:
            if prior_means is not None:
                raise ProphetException("Predictions from prior_models will overwrite prior_means. Either set prior_means to None or prior_models to None.")
            prior_means = self.compute_population_priors(prior_models)
            
            
        self.compute_normalisation(Y,X[:,-1],prior_means=prior_means)
        self.Y = self.normalise(Y,X[:,-1])
        self.testX = testX
        self.testY = self.normalise(testY,testX[:,-1])
        
        
        
        #TODO Check all regions are in X
        
        #TODO NORMALISE deltaX and deltaY...
        #TODO Add checks for NaN, range, etc?
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.deltaOption = deltaOption #what should happen with these variables.
        self.deltaregions = len(deltaOption)
        #TODO not sure what to do if these assertions aren't met...
        #assert len(np.unique(self.deltaX[:,-1]))==self.deltaregions
        #assert np.all(np.arange(len(np.unique(self.deltaX[:,-1])))==np.unique(self.deltaX[:,-1])), "not all the regions are represented in deltaX"
        
        #this is stored for future plotting and reporting.
        self.inputdialysis = inputdialysis
        self.outputdialysis = outputdialysis
        self.outputlab = outputlab
        self.delta_dialysis=delta_dialysis
        self.delta_lab=delta_lab
        
        self.baselinerank = 1 #TODO Make this flexible
              
    def print(self):
        print("Demographics of source patient")
        for item in self.demographics:
            print("%10s %6.2f" % (item,self.demographics[item]))

        print("Patient")
        if self.frompatient is None:
            print(" (parameter not set)")
        else:
            print("Start date: %d" % self.frompatient.startdate)
            #todo add more patient info here? or just call a print statement for the patient?

        print("Computation")
        print("X")
        print(self.X)
        print("Y (normalised)")
        print(self.Y)
        print("Expected number of regions: %d" % self.regions)
        if not hasattr(self,'prior_means'):
            print("Population prior not included ")
        else:
            print("Population prior") #todo
        print("Normalisation Parameters")
        print("Region    Mean      Std")
        for i,np in enumerate(self.normparams):
            print("%2d    %8.4f %8.4f" % (i,np['mean'],np['std']))
        print("Test point")
        print(self.testX)
        print("Test Y (normalised)")
        print(self.testY)

        print("Delta Model")
        print("expected number of delta regions: %d" % self.deltaregions)
        print("X")
        print(self.deltaX)
        print("Y")
        print(self.deltaY)
        print("Delta Option")
        print(self.deltaOption)
        print("")
        print("Requested Constructor Parameters")

        print("Input Dialysis: ")
        print(self.inputdialysis)
        print("Output Dialysis: ")
        print(self.outputdialysis)
        print("Output Lab: ")
        print(self.outputlab)
        print("Delta Dialysis: ")
        print(self.delta_dialysis)
        print("Delta Lab: ")
        print(self.delta_lab) 
        print("")
        print("Results")
        if hasattr(self,'res'):
            for item in self.res:
                print("%10s:"%item)
                print(self.res[item])
        
        
    def remove_outliers(self):
        """
        This method does several operations:
            1. Removes rows in which the days-since-dialysis > 4
            2. Removes rows in which Y is +/- 4std from the mean
            
        """
        assert not hasattr(self, 'outliers_removed'), "Outliers already removed!"
        self.outliers_removed = True
        keep = self.X[:,1]<=4
        if sum(~keep)>0:
            if verbose: print("Removing %d items that have X[:,1]>4" % sum(~keep))
        self.X = self.X[keep,:]
        self.Y = self.Y[keep,:]

        if len(self.X)<2:
            raise ProphetException("Removal of outliers has left this prophet with fewer than two training points.")
        keep = np.full(len(self.X),True)

        #print("iterating over %d regions" % self.regions)
        for region in range(self.regions):
            inreg = self.X[:,-1]==region #which rows are in region
            mean = np.mean(self.Y[inreg,:])
            std = np.std(self.Y[inreg,:])
            lowerbound = mean-std*4
            upperbound = mean+std*4
            keep[inreg & ((self.Y[:,0]<lowerbound) | (self.Y[:,0]>upperbound))] = False
            if sum(~keep)>0:
                if verbose: print("removing %d in region %d due to being 4std out" % (sum(~keep),region))
        self.X = self.X[keep,:]
        self.Y = self.Y[keep,:]
        
    def define_model(self):
        """
        Each type of prophet needs a model
        
        returns m
        """
        raise NotImplementedError
        
    

    
    def get_delta_diff(self,region):
        """
        Just gets the difference between the values of the last two training points in region 'region'"""
        inreg = self.deltaX[:,-1]==region
        y = self.deltaY[inreg,:]
        keep = ~np.isnan(y)[:,0]
        y = y[keep,:]
        
        if len(y)<2:
            return 0
        #delta isn't normalised! Don't need to unnormalise!
        #yA = self.unnormalise_specific_mean(y[-1,0],region)
        #yB = self.unnormalise_specific_mean(y[-2,0],region)
        yA = y[-1,0]
        yB = y[-2,0]
        return yA-yB
    
    def get_delta_abs(self,region):
        """
        Just gets the value of the last training point in region 'region'"""
        inreg = self.deltaX[:,-1]==region
        y = self.deltaY[inreg,:]
        keep = ~np.isnan(y)[:,0]
        y = y[keep,:]
        
        if len(y)<1:
            return 0
        #delta isn't normalised! Don't need to unnormalise!
        #yval = self.unnormalise_specific_mean(y[-1,0],region)
        yval = y[-1,0]
        
        return yval
    
    def get_delta_gradient(self,region, ls=None):
        """
        Make predictions using a simple gradient model.
        """
        if ls is None:
            ls = 3

        inreg = self.deltaX[:,-1]==region
        x = self.deltaX[inreg,0:1]
        y = self.deltaY[inreg,:]

        #remove NaNs from Y
        keep = ~np.isnan(y)[:,0]
        x = x[keep,:]
        y = y[keep,:]
        #remove NaNs from X
        keep = np.all(~np.isnan(x),1)
        x = x[keep,:]
        y = y[keep,:]
        if len(x)<3:
            raise ProphetException("Fewer than three training points (in gradient method).")

        kern = GPy.kern.RBF(1)
        y = y - np.mean(y) #we just remove the mean - we don't save this etc as we'll just be grabbing the gradient later
        m = GPy.models.GPRegression(x,y,kern)
        m.kern.lengthscale.fix(ls)
        m.kern.variance = 5*np.var(y) #no idea what to use for this!
        m.optimize()
         

        testpoint = self.testX[0:1,0:1]
        delta = 0.001
        predmean, _ = m.predict(testpoint)
        
        predmean_delta, _ = m.predict(testpoint+delta)
        
        #m.kern.lengthscale.fix(ls*10)
        #predmean_delta, _ = m.predict(testpoint)
        
        #still untested - unnormalise delta inputs
        #delta isn't normalised! Don't need to unnormalise!
        #predmean = self.unnormalise_specific_mean(predmean[0,0],region)
        #predmean_delta = self.unnormalise_specific_mean(predmean_delta[0,0],region)
        predmean = predmean[0,0]
        predmean_delta = predmean_delta[0,0]
        
        
        return (predmean_delta - predmean)/delta, m
    
    def predict(self):
        """
        Make predictions using the model.
        Specify the list of regions to test in.
        
        Returns:
        - unnormalised means
        - unnormalised variances
        - model
        """
        raise NotImplementedError

    def get_predictions(self,getmodel=False):
        predmean, predvar, m = self.predict()
       

        ##delta model
        delta_values = []
        ms = []
        for floatregion in range(self.deltaregions):            
            region = int(floatregion)
            if self.deltaOption[region][0]=='grad':                
                if len(self.deltaOption[region])>1:
                    ls = self.deltaOption[region][1] #lengthscale can be 2nd element of tuple
                else:
                    ls = None
                try:
                    grad, delta_m = self.get_delta_gradient(region,ls)
                except ProphetException as e:
                    if veryverbose: print("Failed to compute gradient, using 0 (error %s)" % e)
                    grad = 0.0
                    delta_m = None
                delta_values.append(grad)
            if self.deltaOption[region][0]=='diff':
                delta_m = None
                delta_values.append(self.get_delta_diff(region))
            if self.deltaOption[region][0]=='abs':                
                delta_m = None
                delta_values.append(self.get_delta_abs(region))
            ms.append(delta_m)
            
        if getmodel:
            returnedmodel = {'normalmodel':m,'gradientmodels':ms}
        else:
            returnedmodel = None

        return {'mean':predmean, 'var':predvar, 'delta_values':delta_values, 'model':returnedmodel, 'hyperparameters':get_params(m)}
    


    def get_actual(self):
        act = np.full(self.regions,np.NaN)
        act[[int(x) for x in self.testX[:,-1]]] = self.testY[:,0]
        return self.unnormalise_means(act)
        
        
    def demographics_vector(self):
        return [self.demographics[d] for d in self.demographics]
    
    def means_vector(self):
        return [nparam['mean'] for nparam in self.normparams]
        
    def plot(self):
        mod = None
        if not hasattr(self,'res'):
            print("Need results vector to plot")
            return
        if 'model' not in self.res:
            print("Need GP model to plot")
            return
        mod = self.res['model']['normalmodel']
        inputdims = mod.X.shape[1]-1
        i=0
        regnames = self.outputdialysis.copy()
        regnames.extend(self.outputlab)

        for dim in range(inputdims):
            for reg in range(self.regions):
                i+=1
                plt.subplot(1,self.regions*inputdims,i)
                minv,maxv = plotmodel(mod,dim,region=reg,prophet=self)#.plot(fixed_inputs=[(2,0),(1,0)])
                plt.xlabel(self.inputdialysis[dim])
                plt.ylabel(regnames[reg])
          
                #to do - we need to actually test this is the vintage input
                if dim==0:
                    plt.vlines(self.frompatient.hospital['hp_date_start_num'],0,10)
                plt.xlim([minv,maxv])
        
        
        
class ProphetSimple(Prophet):
    """
    Simple model returns the average of the last three observations
    """
    def predict(self):
        normed_means = []
        normed_vars = []
        for reg in range(self.regions):# self.testX[:,-1]:
            Xinreg = self.X[self.X[:,-1]==reg,:] #get those rows of X for region reg
            Yinreg = self.Y[self.X[:,-1]==reg,:]
            orderawayfromtesttime = np.argsort(np.abs(Xinreg[:,0]-self.testX[0,0]) )
            Xinreg = Xinreg[orderawayfromtesttime,:] #get in chronological order
            Yinreg = Yinreg[orderawayfromtesttime,:]
            normed_means.append(np.mean(Yinreg[0:3]))
            normed_vars.append(np.var(Yinreg[0:10])) #get the std of the data
        normed_means = np.array(normed_means)[:,None]
        normed_vars = np.array(normed_vars)[:,None]
        return self.unnormalise_means(normed_means), self.unnormalise_variances(normed_vars), None
    
class ProphetGaussianProcess(Prophet):
    """
    Abstract class of prophets that make predictions using a Gaussian Process.
    """
    def predict(self):
        m = self.define_model()
        try:
            m.optimize()
        except np.linalg.LinAlgError:
            return None, None, None
        testpoints = np.repeat(self.testX[0:1,0:-1],self.regions,0)    
        testpoints = (np.c_[testpoints,np.arange(0,self.regions)[:,None]])
        normalised_predmean, normalised_predvar = m.predict(testpoints)
        return self.unnormalise_means(normalised_predmean), self.unnormalise_variances(normalised_predvar), m
    
    
    
class ProphetCoregionalised(ProphetGaussianProcess):
    """
    Prophets that use a coregionalised representation to make predictions.
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.secondICM = False

    def define_model(self):
        """
        """
        debuginfo = []

        
        #this is an RBF over really just the vintage (coregionalised between all features)
        kern_baseline = (GPy.kern.RBF(self.X.shape[1]-1, ARD=True, name='baselinerbf') \
            **GPy.kern.Coregionalize(input_dim=1, output_dim=self.regions, rank=self.baselinerank, name='baselinecoreg'))
        if self.secondICM:
            kern_baseline+= (GPy.kern.RBF(self.X.shape[1]-1, ARD=True, name='baselinerbf2') \
                **GPy.kern.Coregionalize(input_dim=1, output_dim=self.regions, rank=self.baselinerank, name='baselinecoreg2'))

        kern_dsd = (GPy.kern.RBF(1, active_dims=[0], lengthscale=60, name='dsdrbf') * \
            GPy.kern.Linear(1,active_dims=[1],name='dsdlinear')) \
            **GPy.kern.Coregionalize(input_dim=1, output_dim=self.regions, rank=1, name='dsdcoreg')

        #this accounts for the different noise in the different regions.
        kern_white = GPy.kern.White(2, active_dims=[0,1]) \
            **GPy.kern.Coregionalize(input_dim=1,output_dim=self.regions,rank=1,name='whitenoise')

        kern = kern_baseline + kern_dsd + kern_white

        
        m = GPy.models.GPRegression(self.X,self.Y,kern)
        

        m['.*baselinerbf.variance'].fix(1,warning=False) #this is controlled now by kappa
        m['.*baselinerbf.lengthscale'][1:].fix(100000,warning=False)
        m['.*baselinerbf.lengthscale'][0:1].set_prior(GPy.priors.LogGaussian(np.log(50),0.3),warning=False)
        if self.secondICM:
            m['.*baselinerbf2.variance'].fix(1,warning=False) #this is controlled now by kappa
            m['.*baselinerbf2.lengthscale'][1:].fix(100000,warning=False)
            m['.*baselinerbf2.lengthscale'][0:1].set_prior(GPy.priors.LogGaussian(np.log(50),0.3),warning=False)

        m['.*dsdrbf.lengthscale'].set_prior(GPy.priors.LogGaussian(np.log(50),0.3),warning=False)
        m['.*dsdrbf.variance'].fix(1,warning=False) #this is controlled now by kappa
        #m['.*baseline.kappa']=10 #start off big to reduce temptation to over-coregionalise
        m['.*dsdlinear.variances'].fix(1,warning=False) #this is controlled by kappa

        #this makes the white noise non-coregionalised
        m['.*whitenoise.W'][:,:].constrain_fixed(0,warning=False)
        m['.*white.variance'].fix(1,warning=False) #controlled by kappa

        m['.*dsdcoreg.W'].fix(0,warning=False) #we don't coregionalise DSD
        m.Gaussian_noise.fix(0.01,warning=False)
        return m

class ProphetSimpleGaussian(ProphetGaussianProcess):
    """
    Prophets that use a non-coregionalised representation to make predictions.
    
    TODO Currently just using coreg with a diagonal coreg matrix
    """
    def define_model(self):
        """
        """
        debuginfo = []
        
        #TODO SORT THIS OUT!
        #raise NotImplementedError
        #this is an RBF over really just the vintage
        kern = (GPy.kern.RBF(self.X.shape[1]-1, ARD=True, name='baselinerbf') \
            **GPy.kern.Coregionalize(input_dim=1, output_dim=self.regions, rank=1, name='baselinecoreg'))

        m = GPy.models.GPRegression(self.X,self.Y,kern)

        m['.*baselinecoreg.W'][:,:].fix(0,warning=False)
        m['.*baselinerbf.variance'].fix(1,warning=False) #this is controlled now by kappa
        #m['.*baselinerbf.lengthscale'][1:].fix(100000,warning=False)
        m['.*baselinerbf.lengthscale'][0:1].set_prior(GPy.priors.LogGaussian(np.log(50),0.3),warning=False)
        #m.Gaussian_noise.fix(0.01,warning=False) 
        return m
