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
cache = percache.Cache("cache") #some of the methods to load patient data are cached
from dialysis_analysis.patient import Patient
from dialysis_analysis.prophet import Prophet

verbose = True
veryverbose = False

def build_population_prior_model(prophets):
    priorX = []
    priorYs = []
    for p in prophets:
        priorX.append(p.demographics_vector())
        priorYs.append(p.means_vector())
    priorX = np.array(priorX)
    priorYs = np.array(priorYs)
    ms = []
    for out in range(priorYs.shape[1]):
        k = GPy.kern.RBF(priorX.shape[1],ARD=True)
        keep = ~np.isnan(priorYs[:,out])
        keeppriorX = priorX[keep,:]
        keeppriorYs = priorYs[keep,:]
        if len(keeppriorYs)<3:
            m = None
        else:
            k.lengthscale = np.std(keeppriorX,0)
            m = GPy.models.GPRegression(keeppriorX,keeppriorYs[:,out:(out+1)],k)
            m.optimize()
        ms.append(m)
        
        #set usedforpopmodel to the list so we can check later we aren't predicting for this patient
        #from the same popmodel
        for p in prophets: 
            if p.frompatient is not None: 
                p.frompatient.usedforpopmodel.append(ms)
    return ms        
        
def compute_errors(prophets):
    """Get the RMSE and MAE for all prophets"""
    errs = []
    for i in range(prophets[0].regions): errs.append([])
    for p in prophets:
        if not hasattr(p,'res'):
            if veryverbose: print("Skipping prophet as it has not had its results computed")
            continue
        for reg,(pred,act) in enumerate(zip(p.res['mean'],p.get_actual())):
            if ~np.isnan(act):
                if np.isnan(pred):
                    print("Actual value known, but prediction is NaN.")
                    #raise ProphetException("Actual value known, but prediction is NaN.")
                    continue #not sure we should just let this happen?
                err = pred-act
                errs[reg].append(err)
    mae = []
    rmse = []
    for err in errs:
        mae.append(np.mean(np.abs(err)))
        rmse.append(np.sqrt(np.mean(np.array(err)**2)))
    return mae, rmse

def add_duration_shortfall(patients):
    """
    Adds the duration shortfall to the dialysis tables in all the patients in the list passed
    """
    for p in patients:
        p.dialysis['duration_shortfall'] = p.dialysis['dt_prescr_duration']-p.dialysis['dt_duration']
  
def compute_results(prophets,ip='local'):
    resultsget_predictions = []
    if ip=='local':
        for i, proph in enumerate(prophets):
            getmodel = True #can get all of them (local!)
            res = proph.get_predictions(getmodel)
            proph.res = res
    else:
        print("Launching remote execution")
        delayobjects = []
        if ip is None:
            client = Client(processes=False)
        else:
            print("Setting remote client")
            client = Client(ip+':8786')
        print("adding prophets.get_prediction functions to delayobject list")
        for i, proph in enumerate(prophets):
            getmodel = (i % 10)==0 #only get 1 in x models as these are too large to download
            delayobjects.append(delayed(proph.get_predictions)(getmodel=getmodel))
            #delayobjects.append(delayed(test)(getmodel=getmodel))
        print("Computation Initiated")
        results = compute(*delayobjects, get=client.get)
        print("Computation Complete")
        for proph,res in zip(prophets,results):
            proph.res = res

def loadpatientdata_fromfiles(datafiles):
    dial = pd.read_csv(datafiles['dial'],encoding='latin1')
    pat = pd.read_csv(datafiles['pat'])
    hosp = pd.read_csv(datafiles['hosp'],encoding = "ISO-8859-1")
    lab = pd.read_csv(datafiles['lab'])
    comorbidity = pd.read_csv(datafiles['comorbidity'])
    return dial,pat,hosp,lab,comorbidity

def createpatientobjects(datafiles,dial,pat,hosp,lab,comorbidity,getevery=1):
    patients = []
    patincludelist = None
    if datafiles['uselist'] is not None:
        with open(datafiles['uselist'], 'r') as f:
          reader = csv.reader(f)
          training = list(reader)
        patincludelist = np.array(training)[1:,1].astype(float)

    for pt_code in dial['pt_code'].unique()[::getevery]:
        if patincludelist is not None:
            if pt_code not in patincludelist:
                continue
        try:
            patient = Patient(pat[pat['proband']==pt_code], dial[dial['pt_code']==pt_code], lab[lab['pt_code']==pt_code], hosp[hosp['pt_code']==pt_code], comorbidity[comorbidity['pt_code']==pt_code]) 
            patients.append(patient)
        except TypeError:
            if verbose: print("Skipped patient %d due to invalid date time value" % pt_code)
        except PatientException as e:
            if verbose: print("Skipped patient %d due to an error creating the patient (%s)" % (pt_code,e))
    return patients

@cache
def loadpatientdata(datafiles, getevery = 1):
    """
    Load patient data from datafiles.
    """
    dial,pat,hosp,lab,comorbidity = loadpatientdata_fromfiles(datafiles)
    return createpatientobjects(datafiles,dial,pat,hosp,lab,comorbidity,getevery)
