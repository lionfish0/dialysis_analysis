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
from scipy.stats import pearsonr
from dialysis_analysis.prophet import ProphetException

verbose = True
veryverbose = False

import numpy as np
import matplotlib.pyplot as plt

def hinton(matrix, max_weight=None, ax=None):
    
    """
    Draw Hinton diagram for visualizing a weight matrix.
    
    From https://matplotlib.org/examples/specialty_plots/hinton_demo.html
    Demo of a function to create Hinton diagrams.

    Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
    a weight matrix): Positive and negative values are represented by white and
    black squares, respectively, and the size of each square represents the
    magnitude of each value.

    Initial idea from David Warde-Farley on the SciPy Cookbook
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    
def plot_correlations(correlationmatrix,correlationlabels):
    hinton(correlationmatrix,1.0)
    plt.xticks(np.arange(0,len(correlationlabels)),correlationlabels,rotation=90)
    plt.yticks(np.arange(0,len(correlationlabels)),correlationlabels);
    

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
       
       
def get_dict_of_comorbidity_flags(self,ps):
    """
    We need to know how the ComorbNum relates to the labels given in Flag.
    This method solves this.
    """
    l = []
    for p in ps:
        l.extend(list(zip(list(p.comorbidity['Flag']),list(p.comorbidity['ComorbNum']))))
    res = {}
    for a in set(l):
        res[a[1]] = a[0]
    return res       
       
        
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
    
    
    #I ran the get_dict_of_comorbidty_flags method and have this result
    dict_of_comorbidity_flags = {2: 'ASHD', 3: 'CAN', 4: 'CHF', 5: 'COPD', 6: 'CVA', 7: 'DEP', 8: 'DM', 9: 'DYSRT', 10: 'GI', 11: 'LD',  12: 'OthCard', 13: 'PVD'}
    
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
        
        
    def get_correlation(self,sparsify=1,diffs=False):
        """Computes a correlation matrix between variables. Leaves out some of the data if absent.
        Returns a correlation matrix, and the labels of the data used."""
        df = pd.merge(left=self.dialysis,right=self.lab, left_on='num_date', right_on='lt_date')
        variables = ['dt_duration','days_since_dialysis','dt_prescr_duration','dt_achiev_duration','dt_blood_volume','dt_prescr_blood_flow',
             'dt_achiev_blood_flow','dt_dialysate_flow','dt_weight_pre','dt_weight_post','dt_systolic_pre',
             'dt_systolic_post','dt_diastolic_pre','dt_diastolic_post','dt_heart_rate_pre','dt_heart_rate_post',
             'dt_va_max_flow','weight_change_rate','duration_shortfall','dt_pulse_pressure_post',
             'dt_pulse_pressure_pre','dt_systolic_drop','blood_flow_shortfall','lt_value_albumin','lt_value_calcium',
             'lt_value_cholesterol','lt_value_crp','lt_value_ferritin','lt_value_hdl','lt_value_hb','lt_value_iron',
             'lt_value_pth','lt_value_transferrin','lt_value_wbc','lt_value_phosphate','lt_value_ureapre',
             'lt_value_ureapost','lt_value_triglycerides','lt_value_ldl','lt_value_tsat','lt_value_ekt_v',
             'lt_value_creatinine','lt_value_k','lt_value_na','lt_value_totprot','lt_value_rbc',
             'lt_value_hba1c','lt_value_b2m','lt_value_glycemia','lt_value_platelets',
             'lt_value_prealb','lt_value_folates','lt_value_uricac','lt_value_b12','lt_value_alp']
        shortvariables = ['dur','dsd','prs_dur','ach_dur','blood_vol','prs_bl_flw',
             'ach_bl_flw','dial_flw','wei_pre','wei_post','sys_pre',
             'sys_post','dia_pre','dia_post','hr_pre','hr_post',
             'va_maxflw','wei_chng_rt','dur_short','pulspres_post',
             'pulspres_pre','sys_drop','bl_flw_short','albumin','calcium',
             'cholest','crp','ferritin','hdl','hb','iron',
             'pth','transferrin','wbc','phosphate','ureapre',
             'ureapost','triglycerides','ldl','tsat','ekt_v',
             'creatinine','k','na','totprot','rbc',
             'hba1c','b2m','glycemia','platelets',
             'prealb','folates','uricac','b12','alp']             
        df1 = pd.DataFrame(df, columns=variables)
        if diffs:
            df1 = df1.diff()
#        mat = np.array(df1[::sparsify].corr(),dtype=float)
#        mat[np.isnan(mat)]=0
#        return mat, variables, shortvariables
#        corrs, stats = np.zeros([len(variables),len(variables)]), np.zeros([len(variables),len(variables)])
        corrs = np.full([len(variables),len(variables)],np.nan)
        stats = corrs.copy()
        lens = corrs.copy()
        for i1,c1 in enumerate(df1.columns):
            for i2,c2 in enumerate(df1.columns):
                if i1<i2:
                    pass
                df_clean = df1[[c1,c2]].dropna()
                corrs[i1,i2], stats[i1,i2] = pearsonr(df_clean.iloc[::sparsify,0],df_clean.iloc[::sparsify,1])
                lens[i1,i2] = len(df_clean.iloc[::sparsify,0])
        if diffs:
            self.diff_correlations = {'corrs':corrs, 'stats':stats, 'lens':lens, 'variables':variables, 'shortvariables':shortvariables,'df':df1}
        else:
            self.correlations = {'corrs':corrs, 'stats':stats, 'lens':lens, 'variables':variables, 'shortvariables':shortvariables,'df':df1}            
        
    def build_model_matrices(self,inputdialysis,outputdialysis,outputlab,startvintage,endvintage,startfullresvintage=0,keepratio=0.25):
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
        startfullresvintage: when should every dialysis session be included?
        keepratio: what proportion of earlier dialysis sessions should we keep?
            
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
        
        #we only keep some of the older dialysis sessions
        keep = fullXdial[:,0]>(startfullresvintage)
        keep[np.random.choice(len(keep),int(len(keep)*keepratio))]=True
        fullXdial = fullXdial[keep,:]
        Ydial = Ydial[keep,:]
        
        X = np.r_[fullXdial,fullXlab]
        Y = np.r_[Ydial,Ylab]
        included = (X[:,0]>=startvintage) & (X[:,0]<=endvintage)

        if sum(included)==0:
            copyfrom = max(X[:,0])
            included = (X[:,0]==copyfrom)
            previousvintage = X[X[:,0]<startvintage,0]
            if len(previousvintage)==0:
                previousvintage = startvintage
            else:
                previousvintage = max(previousvintage)
            X = X[included,:]
            X[:,0] = startvintage

            X[:,1] = startvintage-previousvintage
            Y = np.full([X.shape[0],1],np.nan)
        else:
            X = X[included,:]
            Y = Y[included,:]
        return X,Y

    def generate_prophet(self,prophetclass,date,traininglength,inputdialysis,outputdialysis,outputlab,delta_dialysis=None,delta_lab=None,prior_means=None,prior_models=None,fullrestraininglength=np.inf,keepratio=0.25):
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
        fullrestraininglength = how long we should use every dialysis session prior to prediction
        keepratio = what proportion of dialysis sessions we should use before that.            
        """

        X, Y = self.build_model_matrices(inputdialysis, outputdialysis, outputlab, date-traininglength, date-1,date-fullrestraininglength,keepratio)
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
        #regions =len(inputdialysis) #WRONG
        regions = len(outputdialysis)+len(outputlab)
        return prophetclass(X,Y,testX,testY,regions,deltaX = deltaX, deltaY = deltaY, deltaOption=deltaOption, demographics=demographics, prior_means = prior_means, prior_models=prior_models, frompatient=self,  inputdialysis=inputdialysis,outputdialysis=outputdialysis,outputlab=outputlab,delta_dialysis=delta_dialysis,delta_lab=delta_lab)

    def gen_list_of_dates(self,dates):
        newd = []
        if len(dates)>=1: newd.append(dates[0])
        if len(dates)>=2: newd.append(dates[1])
        i = 0
        for i in range(2,len(dates)-1):
            
            newd.append(dates[i])
            j=len(newd)-1
#            if dates[i]-dates[i-1]==3 or dates[i-1]-dates[i-2]==3:
            if newd[j]-newd[j-1]==3 or newd[j-1]-newd[j-2]==3:
                expected_step = 2
            else:
                expected_step = 3
            #if dates[i+1] > dates[i]+expected_step:
            if dates[i+1] > dates[i]+expected_step:
                print("Missing date, expecting %d, but found %d. Adding." % (dates[i]+expected_step, dates[i+1]))
                newd.append(dates[i]+expected_step)
        if len(dates)>2: newd.append(dates[i+1])
        return np.array(newd)


    def generate_all_prophets(self,prophetclass,traininglength,inputdialysis,outputdialysis,outputlab,delta_dialysis=None,delta_lab=None,skipstep=1,stopearly=np.inf,prior_models=None,prior_means=None,fullrestraininglength=np.inf,keepratio=0.25):
        """
        Produces a prediction object for every time point in the dialysis of the patient.
        See generate_prophet for parameter details.
        
        e.g. generate_all_prophets(100,['num_date','days_since_dialysis'],['dt_heart_rate_pre'],['lt_value_calcium'])
        
        set skipstep to a value larger than one to skip some
        set stopearly to only use earlier data from patient (e.g. if interested in vintages <100 days, set to 100)
        """
        prophets = []
        for d in self.gen_list_of_dates(self.dialysis['num_date'][0::skipstep].values): #self.dialysis['num_date'][0::skipstep]:
            if d>=stopearly:
                #print("Stopped early")
                break
            try:
                prophets.append(
                    self.generate_prophet(prophetclass,d,traininglength,
                                          inputdialysis,outputdialysis,
                                          outputlab,
                                          delta_dialysis,delta_lab,
                                          prior_models=prior_models,prior_means=prior_means,fullrestraininglength=fullrestraininglength,keepratio=keepratio))
            except PatientException as e:
                if verbose: print("PatientException: skipping time point %d (%s)" % (d,e))
            except ProphetException as e:
                if verbose: print("ProphetException: skipping time point %d (%s)" % (d,e))
        return prophets
    
    def generate_prehospitalisation_prophets(self,prophetclass,traininglength,inputdialysis,outputdialysis,outputlab,delta_dialysis=None,delta_lab=None,skipstep=1,prehospitalperiod=np.inf,prior_models=None,prior_means=None,fullrestraininglength=np.inf,keepratio=0.25):
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
                                          prior_models=prior_models,fullrestraininglength=fullrestraininglength,keepratio=keepratio))
            except PatientException as e:
                if verbose: print("PatientException: skipping time point %d (%s)" % (d,e))
            except ProphetException as e:
                if verbose: print("ProphetException: skipping time point %d (%s)" % (d,e))
        return prophets 
    
    def plot():
        d = self.dialysis
        plt.plot(d['num_date'],d['weight_change_rate'])
