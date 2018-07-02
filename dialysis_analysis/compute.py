import dialysis_analysis
from dialysis_analysis.patient import Patient, PatientException
from dialysis_analysis.prophet import ProphetCoregionalised, ProphetSimple, ProphetException, ProphetSimpleGaussian
from dialysis_analysis.dialysis_delta_model import DeltaModel
import dask_dp4gp
from dialysis_analysis import compute_results_dataframe
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

#Application specific analysis (using repo as distribution method)

ip = sys.argv[1]#'34.242.203.46'

ps = pickle.load(open('patients.p','rb'))

startps = 0
endps = len(ps)
if len(sys.argv)==3:
    startps = int(sys.argv[2])
if len(sys.argv)>=4:
    startps = int(sys.argv[2])
    endps = int(sys.argv[3])

def getprophets(prophetclass,ps,outputdialysis,outputlab,prior_means=None):
    np.random.seed(0)

    traininglength = 900
    inputdialysis = ['num_date','days_since_dialysis']    
    delta_dialysis = None
    delta_lab = None
    prehospitalperiod=100
    skipstep=1#<<<<
    fullrestraininglength=60 #changed from 90
    keepratio = 0.25

    gen_prophets_from = 'all' #or prehosp
    prophets = []
    for p in ps:
        print("-------------------------------------------------------------------------------")
        print(p.pat['proband'])
        print("hospital rows %d" % p.hospital.shape[0])
        print("lab rows %d" % p.lab.shape[0])
        print("dialysis rows %d" % p.dialysis.shape[0])        
        try:
            if gen_prophets_from == 'all':
                patient_prophets = p.generate_all_prophets(prophetclass,traininglength,inputdialysis,outputdialysis,outputlab,
                                       delta_dialysis=delta_dialysis,
                                       delta_lab=delta_lab,skipstep=skipstep,prior_means=prior_means,
                                      fullrestraininglength=fullrestraininglength,keepratio=keepratio)
            if gen_prophets_from == 'prehosp':
                patient_prophets = p.generate_all_prophets(prophetclass,traininglength,inputdialysis,outputdialysis,outputlab,
                                       delta_dialysis=delta_dialysis,prior_means=prior_means,
                                       delta_lab=delta_lab,prehospitalperiod=prehospitalperiod,skipstep=skipstep,
                                      fullrestraininglength=fullrestraininglength,keepratio=keepratio)
            print("Total prophets initially for this patient: %d" % len(patient_prophets))
            for proph in patient_prophets[:]:
                try:
                    proph.remove_outliers()
                except ProphetException as e:
                    print("Error removing outliers: %s" % e)
                    print("Removing prophet")
                    patient_prophets.remove(proph)
            print("Total prophets for this patient: %d" % len(patient_prophets))
            
            prophets.extend(patient_prophets)        
        except ProphetException as e:
            print("Error creating prophets: %s" % e)

    return prophets
    
prior_means = np.array([  0.67, 356.76, 132.95,  65.  , 233.86,   5.22,  22.  , 117.75,  977.95,   9.01,  39.52, 637.64,   2.15,   6.39,   1.3 ])    


step=10
experiments = []
experiments.append([
['weight_change_rate','dt_achiev_blood_flow', 'dt_systolic_pre', 'dt_diastolic_pre', 'dt_achiev_duration'],
['lt_value_k','lt_value_ureapre','lt_value_hb','lt_value_ferritin','lt_value_crp','lt_value_albumin','lt_value_creatinine','lt_value_calcium','lt_value_wbc','lt_value_phosphate']
])

for i in range(startps,endps,step):
    for expi, experiment in enumerate(experiments):
        outputdialysis_subset = experiment[0]
        outputlab_subset = experiment[1]
        track = {}
        print("> %d of %d" % (i,len(ps)))
        gp_prophets = getprophets(ProphetSimpleGaussian,ps[i:i+step],outputdialysis_subset,outputlab_subset,prior_means=prior_means)
        
                #set the prechosen hyperparameters:
        for gp_prop in gp_prophets:
            gp_prop.preset_hyperparameters = np.array([[  0.69, 186.12, 187.96,   0.51],
       [  0.5 , 186.58, 495.93,   0.97],
       [  0.79, 191.89, 192.  ,   0.41],
       [  0.8 , 204.66, 173.09,   0.3 ],
       [  0.39, 187.46, 423.93,   3.48],
       [  0.78, 211.48, 102.82,   0.72],
       [  1.03, 207.03,  69.02,   1.12],
       [  2.85, 172.73, 278.15,   3.57],
       [ 22.97, 201.56, 203.8 ,  23.53],
       [  7.06, 215.54,  81.85,   7.12],
       [  8.47, 208.38, 128.99,   8.62],
       [ 54.97, 207.82, 153.85,  55.23],
       [  0.72, 209.36, 122.61,   0.58],
       [  0.76, 216.64,  71.61,   0.46],
       [  0.74, 208.02, 119.83,   0.62]])
            
            
            gp_prop.optimize_model = False

        for tracki,g in enumerate(gp_prophets):
            track[tracki]=g.frompatient
            g.frompatient = None
        dialysis_analysis.compute_results(gp_prophets,ip=ip,chunksize=1000)
        for tracki,g in enumerate(gp_prophets): #TO FIX THIS MAYBE POINTS ALL PROPHETS AT SAME PATIENT!
            g.frompatient = track[tracki]
        df = dialysis_analysis.compute_results_dataframe(gp_prophets)
        df.to_csv('exp%03dpatblock%05d.csv' % (expi,i))
    
