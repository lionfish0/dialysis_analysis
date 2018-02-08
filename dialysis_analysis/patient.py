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
from dialysis_analysis import *
cache = percache.Cache("cache") #some of the methods to load patient data are cached

verbose = True
veryverbose = False

class PatientException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs) 


def strtodate(s):
    #helper function to convert string to datetime object
    return datetime.strptime(s,'%d%b%Y')

def strtodatenum(d):
    #get date string into days since epoch
    return (strtodate(d)-datetime(1970,1,1)).days

def patstrtodate(s):
    #helper function to convert from PATIENT table string to datetime object
    return datetime.strptime(s,'%d.%m.%Y')

def patstrtodatenum(d):
    #get PATIENT FORMATTED date string into days since epoch
    return (patstrtodate(d)-datetime(1970,1,1)).days

def datenumtopatstr(days):
    return (datetime(1970,1,1)+timedelta(days=days)).strftime("%d.%m.%Y")

assert datenumtopatstr(patstrtodatenum("23.07.2009"))=='23.07.2009'

def getweekdayfromnum(n):
    #As we've now got days since epoch we need to get day of week of a given day
    return ((datetime(1970,1,1).weekday())+n)%7

def strseriestodatenum(l):
    dates = np.zeros(len(l))
    for i,d in enumerate(l):
        dates[i]=strtodatenum(d)
    return dates
       
       
       
       
        
class Patient(object): 
    #class variables for the names of the lab variables and dialysis variables
    labvars = ['lt_value_albumin', 'lt_value_calcium', 'lt_value_cholesterol', 'lt_value_crp',
               'lt_value_ferritin', 'lt_value_hdl', 'lt_value_hb', 'lt_value_iron', 'lt_value_pth',
               'lt_value_transferrin', 'lt_value_wbc', 'lt_value_phosphate', 'lt_value_ureapre',
               'lt_value_ureapost', 'lt_value_triglycerides', 'lt_value_ldl', 'lt_value_tsat',
               'lt_value_ekt_v', 'lt_value_creatinine', 'lt_value_k', 'lt_value_na', 'lt_value_totprot',
               'lt_value_rbc', 'lt_value_hba1c', 'lt_value_b2m', 'lt_value_glycemia', 'lt_value_platelets',
               'lt_value_prealb', 'lt_value_folates', 'lt_value_uricac', 'lt_value_b12', 'lt_value_alp']
    
    dialvars = ['dt_duration','dt_achiev_duration',
                'dt_prescr_duration','dt_blood_volume','dt_prescr_blood_flow','dt_achiev_blood_flow',
                'dt_weight_pre','dt_weight_post','dt_systolic_pre','dt_systolic_post','dt_diastolic_pre',
                'dt_diastolic_post','dt_heart_rate_pre','dt_heart_rate_post']
    
    def __init__(self, pat, dialysis, lab, hospital, comorbidity):
        """
        Create a patient object.
        Pass dataframes for the:
            patient  [uses baseline_pt_pop_analysis_c, baseline_pt_date_fu_start baseline_pt_date_fu_stop, baseline_pt_dialysis_vintage]
            dialysis [uses dt_valid_session_d, dt_date]
            lab      [uses lt_date]
            hospital [uses hp_date_start, hp_date_stop]
            comorbidity [unused]
        """
        self.usedforpopmodel = [] #list of population models this patient's involved in
        
        #skip patient that aren't available for analysis
        if pat['baseline_pt_pop_analysis_c'].values[0]!=1:
            raise PatientException("Patient unavailable for analysis")
        
        #get the start and stop date of the followup study
        fu_start_date = patstrtodatenum(pat['baseline_pt_date_fu_start'].values[0])
        fu_stop_date = patstrtodatenum(pat['baseline_pt_date_fu_stop'].values[0])
        
        #First get the dates of dialysis and hospitalisation from database
        #into numerical lists (if it fails record as error)
        #can raise a TypeError
        dial_dates = []
        dialysis = dialysis[dialysis['dt_valid_session_d']=='Yes'] #exclude dates that weren't valid
        dial_dates = strseriestodatenum(dialysis.dt_date.values)            
        dialysis['num_date'] = dial_dates #add new column to dataframe with dates as numbers
        hospitalisation_dates_str = hospital['hp_date_start']
        hospitalisation_stop_dates_str = hospital['hp_date_stop']
        hosp_dates = np.zeros(len(hospitalisation_dates_str))
        for i,h in enumerate(hospitalisation_dates_str.values):
            hosp_dates[i]=strtodatenum(h)
        hosp_stop_dates = np.zeros(len(hospitalisation_stop_dates_str))
        for i,h in enumerate(hospitalisation_stop_dates_str.values):
            hosp_stop_dates[i]=strtodatenum(h)
        
        #included hospital dates
        inchospdates = (hosp_dates>=fu_start_date) & (hosp_dates<=fu_stop_date)
        hosp_dates = hosp_dates[inchospdates]
        hosp_stop_dates = hosp_stop_dates[inchospdates]
        
        #included dialysis sessions
        incdialdates = (dial_dates>=fu_start_date) & (dial_dates<=fu_stop_date)
        dialysis = dialysis[incdialdates] #only include dates within the followup study

        if len(dialysis)<1: #if the patient has no dates in that range abort
            raise PatientException("Patient has no data outside hospital")
            
        outpatient_ends = np.append(hosp_dates,dialysis['num_date'].values[-1]+1)
        outpatient_starts = np.append(0,hosp_stop_dates)
        
        dial_lists = []
        for start, stop in zip(outpatient_starts,outpatient_ends):
            dial_lists.append(dialysis[(dialysis['num_date']>start) & (dialysis['num_date']<stop)])

        ###Lab Test Data
        lab['lt_date'] = strseriestodatenum(lab['lt_date'].values)
        labdata = lab[Patient.labvars]
        
        #combine the periods between hospitalisations (remove the first of each, and add a column for days_since_dialysis)
        for l in dial_lists:
            if len(l)<1: #if this period has no data, skip...
                continue
            dayssincedial = np.diff(l['num_date'])
            weightchange = l['dt_weight_pre'][1:].values-l['dt_weight_post'][:-1].values #find the delta in the weight change
            weightchange = weightchange/dayssincedial
            
            
            #l.drop(l.index[0], inplace=True) #erase first entry as we can't guess this as we don't know how long it's been since the previous dialysis session
            l['days_since_dialysis'] = np.r_[np.NaN,dayssincedial]
            l['weight_change_rate'] = np.r_[np.NaN,weightchange]
        dialysis = pd.concat(dial_lists) #overwrite old dialysis variable
      
        if (len(dialysis)<1):
            raise PatientException("Patient has insufficient data outside hospital")
            
        #e.g. Xdial = 1000, p['vintage']=20, startdate = 1000-20 = 980.
        #what vitage is the first row of the dialysis dataset?
        startdate = dialysis['num_date'].values[0]-pat['baseline_pt_dialysis_vintage'].values[0]
        
        #replace the dates with the VINTAGE of the patient (so subtract the date they started on dialysis)
        dialysis['num_date'] -= startdate
        lab['lt_date'] -= startdate
        

        
        self.startdate = startdate
        self.dialysis = dialysis
        self.pat = pat
        self.lab = lab
        self.hospital = hospital
        self.comorbidity = comorbidity
        self.lab = lab
        
        
        hospdates = []
        for d in self.hospital['hp_date_start']:
            hospdates.append(strtodatenum(d)-self.startdate)
        hospdates = np.array(hospdates)
        self.hospital['hp_date_start_num'] = hospdates
        
                
        
        
    def build_model_matrices(self,inputdialysis,outputdialysis,outputlab,startvintage,endvintage):
        """
        Parameters:
        
        inputdialysis
            Which variables we want as inputs to the model. Pass as a list of tuples.
            The first value is the label of the variable, the second is either
            'current' or 'previous' to indicate if we should be modelling this
            with the current value or the last one observed. 
            
            Important note: All variables that are not vintage or days_since_dialysis
                            the algorithm will use the PREVIOUS SESSIONS value - as
                            it is assumed we don't have the next session's value!
                
        outputdialysis    
            Which variables we want as outputs of the model?
        inputlab
            All these variables will use the previous observation (this means they
            might be quite out of date!)
        outputlab
            Which variables we want as outputs of the model?
            
        startvintage and endvintage: time period to include in matrices INCLUSIVE
            
        Returns matrices X and Y
        """
        Xdial = np.zeros([len(self.dialysis),0])
        for dialvar in inputdialysis:
            data = self.dialysis[dialvar].values.copy()
            if dialvar not in ['num_date','days_since_dialysis']:
                #moving data back one time step, as it can't be observed at prediction time
                data[:-1] = data[1:]
                data[-1] = np.NaN
            Xdial = np.c_[Xdial,data]

        Xlab = self.lab['lt_date'].copy()
        dialysisinputs = np.zeros([len(self.lab),Xdial.shape[1]-1])
        for i, tdiff in enumerate(self.lab['lt_date'].values[:,None]-self.dialysis['num_date'].values[None,:]):#(self.lab['lt_date']-self.dialysis['num_date'])):
            if (len(np.where(tdiff>=0)[0])==0): #the lab sample was before any dialysis - so we don't know how many days since...
                continue

            loc = np.where(tdiff>=0)[0][-1]
            dialysisinputs[i,:] = Xdial[loc,1:]

            #we might have to overwrite the days-since-dialysis with 'tdiff'? Maybe don't if it's on the same day...
            if tdiff[loc] != 0: #TODO: Comment out this 'if' if we take labs AFTER dialysis
                if 'days_since_dialysis' in inputdialysis:
                    dialysisinputs[i,inputdialysis.index('days_since_dialysis')-1] = tdiff[loc]
        Xlab = np.c_[Xlab,dialysisinputs] #TODO: Confirm the labs are taken AFTER dialysis    


        Ylab = np.zeros([0,1])
        indx = 0
        Ydial = np.zeros([0,1])
        fullXdial = np.zeros([0,1+Xdial.shape[1]])
        for dialvar in outputdialysis:
            if dialvar not in self.dialysis:
                raise PatientException("Column %s not in dialysis dataframe" % dialvar)
            col = self.dialysis[dialvar]
            N = col.shape[0]
            Ydial = np.r_[Ydial,col[:,None]]
            fullXdial = np.r_[fullXdial,np.c_[Xdial,indx*np.ones([N,1])]]
            indx+=1
            
        fullXlab = np.zeros([0,1+Xlab.shape[1]])
        for labvar in outputlab:
            col = self.lab[labvar]
            N = col.shape[0]
            Ylab = np.r_[Ylab,col[:,None]]
            fullXlab = np.r_[fullXlab,np.c_[Xlab,indx*np.ones([N,1])]]
            indx+=1
        
        
        X = np.r_[fullXdial,fullXlab]
        Y = np.r_[Ydial,Ylab]
        included = (X[:,0]>=startvintage) & (X[:,0]<=endvintage)
        
        X = X[included,:]
        Y = Y[included,:]
        return X,Y

    def generate_prophet(self,prophetclass,date,traininglength,inputdialysis,outputdialysis,outputlab,delta_dialysis=None,delta_lab=None,prior_means=None,prior_models=None):
        """
        Produces a prophet prediction object for time point 'date' for the patient.
        
        traininglength = how far back in time to include from test day
        inputdialysis = variables to use as inputs
        outputdialysis, outputlab = variables to use to predict
        delta_dialysis and delta_lab = variables to use as inputs to the delta model.
           Pass as a list of tuples, e.g.
                                   delta_dialysis=[('dt_heart_rate_pre','grad'),
                                                   ('dt_duration','diff')],
                                   delta_lab=[('lt_value_calcium','grad',2)]
        the second element of each tuple describes how we should process that variable for the model.
        Possible values include:
            - abs = absolute (actual value)
            - grad = the gradient from the simple model (a third item in the tuple can be added to specify the lengthscale)
            - diff = difference (whether we will just be considering the difference since the last measurement)
        """
        
        X, Y = self.build_model_matrices(inputdialysis, outputdialysis, outputlab, date-traininglength, date-1)
        testX,testY = self.build_model_matrices(inputdialysis, outputdialysis, outputlab, date, date)
        if len(X)<3:
            raise PatientException("Fewer than three training points.")
            
        deltaOption = []
        if delta_dialysis is not None:
            for dd in delta_dialysis:
                assert type(dd) is tuple
            deltaX, deltaY = self.build_model_matrices(['num_date'],[dd[0] for dd in delta_dialysis],[dl[0] for dl in delta_lab],date-traininglength,date-1)
            deltaOption.extend([dopt[1:] for dopt in delta_dialysis])
            deltaOption.extend([dopt[1:] for dopt in delta_lab])
        else:
            deltaX = None
            deltaY = None
            
        #a summary of the patient's demographics (useful for producing a prior)
        
        age = self.pat['baseline_pt_age'].values[0] + (date/365)
        vintage = date
        weight = self.dialysis['dt_weight_post'].values[0]
        height = self.pat['baseline_pt_height'].values[0]
        gender = self.pat['baseline_pt_gender_c'].values[0]
        demographics = {'age':age,'vintage':vintage,'weight':weight,'height':height,'gender':gender}
        #frompatient might be causing memory problems! TODO
        return prophetclass(X,Y,testX,testY,len(inputdialysis),deltaX = deltaX, deltaY = deltaY, deltaOption=deltaOption, demographics=demographics, prior_means = prior_means, prior_models=prior_models, frompatient=self,  inputdialysis=inputdialysis,outputdialysis=outputdialysis,outputlab=outputlab,delta_dialysis=delta_dialysis,delta_lab=delta_lab)


    def generate_all_prophets(self,prophetclass,traininglength,inputdialysis,outputdialysis,outputlab,delta_dialysis=None,delta_lab=None,skipstep=1,stopearly=np.inf,prior_models=None):
        """
        Produces a prediction object for every time point in the dialysis of the patient.
        See generate_prophet for parameter details.
        
        e.g. generate_all_prophets(100,['num_date','days_since_dialysis'],['dt_heart_rate_pre'],['lt_value_calcium'])
        
        set skipstep to a value larger than one to skip some
        set stopearly to only use earlier data from patient (e.g. if interested in vintages <100 days, set to 100)
        """
        prophets = []
        for d in self.dialysis['num_date'][0::skipstep]:
            if d>=stopearly:
                print("Stopped early")
                break
            try:
                prophets.append(
                    self.generate_prophet(prophetclass,d,traininglength,
                                          inputdialysis,outputdialysis,
                                          outputlab,
                                          delta_dialysis,delta_lab,
                                          prior_models=prior_models))
            except PatientException as e:
                if verbose: print("skipping time point %d (%s)" % (d,e))
        return prophets
    
    def generate_prehospitalisation_prophets(self,prophetclass,traininglength,inputdialysis,outputdialysis,outputlab,delta_dialysis=None,delta_lab=None,skipstep=1,prehospitalperiod=np.inf,prior_models=None):
        """
        Produces a prediction object for every time point in the dialysis of the patient in the run up to their hospitalisations.
        See generate_prophet for parameter details.
        """
        
        hospdates = np.array(self.hospital['hp_date_start_num'])
    
        prophets = []
        for d in self.dialysis['num_date'][0::skipstep]:
            daystillhosp = hospdates-d
            if not np.any((daystillhosp>0) & (daystillhosp<prehospitalperiod)): #if we're not within prehospperiod of a hospitalisation
                continue
            try:
                prophets.append(
                    self.generate_prophet(prophetclass,d,traininglength,
                                          inputdialysis,outputdialysis,
                                          outputlab,
                                          delta_dialysis,delta_lab,
                                          prior_models=prior_models))
            except PatientException as e:
                if verbose: print("skipping time point %d (%s)" % (d,e))
        return prophets 
    
    def plot():
        d = self.dialysis
        plt.plot(d['num_date'],d['weight_change_rate'])
