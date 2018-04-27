import numpy as np
import pickle
import csv
from datetime import datetime,timedelta
import pandas as pd
pd.options.mode.chained_assignment = None
import GPy
from dask import compute, delayed
from dask.distributed import Client
np.set_printoptions(precision=2,suppress=True)
import percache
cache = percache.Cache("cache") #some of the methods to load patient data are cached
from dialysis_analysis.patient import Patient, PatientException
from dialysis_analysis.prophet import Prophet, ProphetException
import os
import importlib

verbose = True
veryverbose = False



def install_libraries_on_workers(url,runlist = None):
    """Install libraries if necessary on workers etc.
    e.g. if already on server...
    install_libraries_on_workers('127.0.0.1:8786')
    """
    
    client = Client(url)
    
    if runlist is None:
        runlist = ['sudo apt-get -y install build-essential','pip install -U pip','sudo apt install libgl1-mesa-glx -y','conda update scipy -y','pip install git+https://github.com/sods/paramz.git','pip install git+https://github.com/SheffieldML/GPy.git','pip install git+https://github.com/lionfish0/dp4gp.git','conda install dask-searchcv -c conda-forge -y', 'pip install git+https://github.com/lionfish0/dask_dp4gp.git', 'pip install numpy', 'conda remove argcomplete -y','pip install git+https://github.com/lionfish0/dialysis_analysis.git --upgrade']#, 'conda install python=3.6 -y']

    for item in runlist:
        print("Installing '%s' on workers..." % item)
        res = client.run(os.system,item)
        print(res)
        print("Installing '%s' on scheduler..." % item)
        res = client.run_on_scheduler(os.system,item)    
        print(res)
        #os.system(item) #if you need to install it locally too

def reload_modules_on_workers(url,modulelist = None):
    """Run reload(module) on the items in the modulelist"""
    client = Client(url)
    for mod in modulelist:
        print("reloading %s" % mod)
        client.run(importlib.reload,mod)
        client.run_on_scheduler(importlib.reload,mod)

def build_population_prior_model(prophets):
    priorX = []
    priorYs = []
    for p in prophets:
        priorX.append(p.demographics_vector())
        priorYs.append(p.means_vector())
    priorX = np.array(priorX)
    priorYs = np.array(priorYs)
    
    #remove NAN rows - this might be done a little below - TODO Remove duplicate code
    keep = np.all(~np.isnan(priorX),1)
    priorX = priorX[keep,:]
    priorYs = priorYs[keep,:]
    keep = np.all(~np.isnan(priorYs),1)
    priorX = priorX[keep,:]
    priorYs = priorYs[keep,:]

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


def compute_errors(prophets,meantype='mean'):
    """Get the RMSE and MAE for all prophets.
    
    set meantype to 'uncorrected_mean' to use the non-delta model version.
    
    Specifically the function returns arrays of:
    Mean Absolute Error (mae)
    Root Mean Squared Error (rmse)
    The standard deviation of the MAE estimates
    The standard deviation of the RMSE estimates
    The total number of results used to compute each estimate
    """
    errs = []
    for i in range(prophets[0].regions): errs.append([])
    for p in prophets:
        if not hasattr(p,'res'):
            if veryverbose: print("Skipping prophet as it has not had its results computed")
            continue
        if meantype not in p.res:
            if veryverbose: print("Skipping prophet as %s is not in the results structure" % meantype)
            continue
        if p.res[meantype] is None:
            if veryverbose: print("Skipping prophet as %s is None" % meantype)
            continue            
        for reg,(pred,act) in enumerate(zip(p.res[meantype],p.get_actual())):
            if ~np.isnan(act):
                if np.isnan(pred):
                    #print("Actual value known, but prediction is NaN.")
                    #raise ProphetException("Actual value known, but prediction is NaN.")
                    continue #not sure we should just let this happen?
                err = pred-act
                errs[reg].append(err)
    mae = []
    rmse = []
    count = []
    mae_std = []
    rmse_std = []
    for allerr in errs:
        mae.append(np.mean(np.abs(allerr)))
        listoferrs = []
        for it in range(500):
            err = np.random.choice(np.array(allerr)[:,0],len(allerr),replace=True)
            listoferrs.append(np.mean(np.abs(err)))
        mae_std.append(np.std(listoferrs))
        
        rmse.append(np.sqrt(np.mean(np.array(allerr)**2)))
        listoferrs = []
        for it in range(500):
            err = np.random.choice(np.array(allerr)[:,0],len(allerr),replace=True)
            listoferrs.append(np.sqrt(np.mean(np.array(err)**2)))
        rmse_std.append(np.std(listoferrs))
        count.append(len(err))        
    return np.array(mae), np.array(rmse), np.array(mae_std), np.array(rmse_std), np.array(count)



def get_days_til_hospitalisation(prophet,vintage):
    daystil = prophet.frompatient.hospital['hp_date_start_num']-vintage
    daystil = daystil[daystil>0]
    return np.min(daystil)
    
def compute_results_dataframe(prophets):
    data = []
    regnames = prophets[0].outputdialysis.copy()
    regnames.extend(prophets[0].outputlab)
    colnames = []
    colnames.append('Prophet number')
    colnames.append('Patient Id')
    colnames.append('Vintage')
    for r in regnames:
        colnames.append(r+" (actual)")
        colnames.append(r+" (predicted)")
    colnames.append('Days until hospitalisation')
    colnames.append('Date')
    for i,p in enumerate(prophets):
        row = []
        row.append(i)
        row.append(p.frompatient.pat['proband'].values[0])
        vintage = p.testX[0,0]
        row.append(vintage) #TODO Assumes first row is vintage
        
        if not hasattr(p,'res'):
            if veryverbose: print("Skipping prophet as it has not had its results computed")
            continue
        if p.res['mean'] is None:
            if veryverbose: print("Skipping prophet as it has None as prediction")
            continue
            
        for reg,(pred,act) in enumerate(zip(p.res['mean'],p.get_actual())):
            row.append(act)
            row.append(pred[0])
            
        row.append(get_days_til_hospitalisation(p,vintage))
        #get actual date
        row.append(p.frompatient.dialysis[p.frompatient.dialysis['num_date']==p.testX[0,0]]['dt_date'].values[0])
        data.append(row)
    df = pd.DataFrame(data,columns=colnames)
    return df

#TODO: Move these methods to inside the patient class?
def add_pulse_pressures(patients):
    """
    Adds the pre and post pulse pressures to the dialysis tables in all the patients in the list passed
    Also add the syspre-syspost
    """
    for p in patients:
        p.dialysis['dt_pulse_pressure_post']=p.dialysis['dt_systolic_post']-p.dialysis['dt_diastolic_post']
        p.dialysis['dt_pulse_pressure_pre']=p.dialysis['dt_systolic_pre']-p.dialysis['dt_diastolic_pre']
        p.dialysis['dt_systolic_drop']=p.dialysis['dt_systolic_pre']-p.dialysis['dt_systolic_post']
        
def add_duration_shortfall(patients):
    """
    Adds the duration shortfall to the dialysis tables in all the patients in the list passed
    """
    for p in patients:
        p.dialysis['duration_shortfall'] = p.dialysis['dt_prescr_duration']-p.dialysis['dt_duration']
        
def add_bloodflow_shortfall(patients):
    """
    Adds the bloodflow shortfall to the dialysis tables in all the patients in the list passed
    """
    for p in patients:
        p.dialysis['blood_flow_shortfall'] = p.dialysis['dt_prescr_blood_flow']-p.dialysis['dt_achiev_blood_flow']
        
        
def add_comorbidity_columns_to_df(data,main_df,date_column):
    """
    Adds maxnumcomorbidities comorbidity columns to the table
    """
    maxnumcomorbidities = 14
    if len(main_df)<1:
        return
    times_since = []
    for date in main_df[date_column]:
        timesince = np.full(maxnumcomorbidities,np.inf)
        for d in data:
            t = (date-d[0])
            if t<0: t = np.inf
            timesince[d[2]] = min(t,timesince[d[2]])
        times_since.append(timesince)
    times_since = np.array(times_since)
    comorb_effect = 10.0/(10.0+times_since) #never = 0, now = 1, 10 days ago = 0.5
    for i in range(comorb_effect.shape[1]):
        if i not in Patient.dict_of_comorbidity_flags: #skip comorbidities not in the dictionary
            continue
        main_df['comorbidity_factor_%s'%Patient.dict_of_comorbidity_flags[i]] = comorb_effect[:,i]
        main_df['has_comorbidity_%s'%Patient.dict_of_comorbidity_flags[i]] = times_since[:,i]!=np.inf #have they had this comorbidity any time in the past?
        
def add_comorbidity_columns(patients):
    """
    Add comorbidity columns to the dialysis and lab tables for all patients
    """
    for p in patients:
        df = p.comorbidity
        df = df[df['ComorbPost']==1]
        data = df[df['Flag'].isin(['ASHD','CHF','CVA','DYSRT','OthCard'])][['cm_date_new','ComorbPost','ComorbNum']].values.tolist()
        for i in range(len(data)):
            data[i][0] = (datetime.strptime(data[i][0],'%Y-%m-%d')-datetime(1970,1,1)).days-p.startdate

        add_comorbidity_columns_to_df(data,p.dialysis,'num_date')
        add_comorbidity_columns_to_df(data,p.lab,'lt_date')
  
def compute_results_chunk(prophets,ip):
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
            getmodel = (i % 100)==0 #only get 1 in x models as these are too large to download
            delayobjects.append(delayed(proph.get_predictions)(getmodel=getmodel))
            #delayobjects.append(delayed(test)(getmodel=getmodel))
        print("Computation Initiated")
        results = compute(*delayobjects, get=client.get)
        print("Computation Complete")
        for proph,res in zip(prophets,results):
            proph.res = res

def compute_results(prophets,chunksize=1000,ip='local'):
    for chunk in np.arange(0,len(prophets),chunksize):
        print("Computing %d of %d" % (chunk,len(prophets)))
        currentprophets = prophets[chunk:(chunk+chunksize)]
        try:
            compute_results_chunk(currentprophets,ip=ip)
        except:
            print("Exception occured in one of the predictions, running seperately")
            for minichunk in np.arange(chunk,chunk+chunksize,20): #run in batches of 100 instead of chunksize
                minicurrentprophets = prophets[minichunk:(minichunk+20)]
                try:
                    compute_results_chunk(minicurrentprophets,ip=ip)
                except:
                    print("failed to compute a prophet (skipping block of 20)")
                #    for mcp in minicurrentprophets: #run one by one
                #        try:
                #            compute_results_chunk([mcp],ip=ip)
                #        except:
                #            print("failed to compute one prophet")
                    

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

