import dialysis_analysis
import pandas as pd
import numpy as np
from datetime import date,datetime,timedelta
import unittest
from dialysis_analysis import *
from dialysis_analysis.patient import Patient, strtodatenum
from dialysis_analysis.prophet import ProphetSimple, ProphetCoregionalised
from dialysis_analysis.dialysis_delta_model import DeltaModel

class TestDialysisAnalysis(unittest.TestCase):
    def build_test_patient(self,startdate,enddate):
        """
        To test if the deltamodel works
        Pass the startdate and enddate as date objects, e.g. build_test_patient(datetime.date(2008,07,10))
        The method picks 2 and 3 day spacings
        """

        age = 60+10*np.random.randn() #95% of test patients between 40 and 80.
        gender = np.random.choice([1,2])
        expectedhrforage = 75-(age-40)/2 #40yo=75bpm, 80yo=55
        height = np.random.randn()*10+152+10*gender
        bmi = 30+np.random.randn()*2
        if bmi<18:
            bmi = 30+np.random.randn()*2
        weight = bmi * ((height/100)**2)
        expectedhrforage += 2*(bmi-27) #increase hr with bmi
        testpat = pd.DataFrame(
            data={'baseline_pt_pop_analysis_c': [1], 
                  'baseline_pt_date_fu_start': startdate.strftime("%d.%m.%Y"), 
                  'baseline_pt_date_fu_stop': enddate.strftime("%d.%m.%Y"), 
                  'baseline_pt_dialysis_vintage':[0],
                  'baseline_pt_age':age,
                  'baseline_pt_height':height, #todo which is male? 1 or 2?
                  'baseline_pt_gender_c':gender})

        date = startdate
        gapidx = 0
        gapspacing = [2,2,3]
        freqoflabs = 21 #how many days?
        lastlab = startdate-timedelta(10)#arbitrarily earlier? when to start labs?
        datestringlist = []
        labdatestringlist = []
        while date<=enddate:
            gapidx+=1
            if gapidx>=len(gapspacing):
                gapidx = 0
            date+=timedelta(gapspacing[gapidx])
            datestringlist.append(date.strftime("%d%b%Y"))

            if lastlab-date<timedelta(-21):
                labdatestringlist.append(date.strftime("%d%b%Y"))

        testdial = pd.DataFrame(data={
            'dt_valid_session_d': ['Yes']*len(datestringlist), 
            'dt_date': datestringlist})
        testlab = pd.DataFrame(data={
            'lt_date': labdatestringlist})
        testhosp = pd.DataFrame(data={
            'hp_date_start': [],
            'hp_date_stop': []})
        testcomorb = pd.DataFrame(data={})

        for v in dialysis_analysis.Patient.labvars:
            testlab[v] = [np.NaN]*len(testlab)

        for v in dialysis_analysis.Patient.dialvars:
            testdial[v] = [np.NaN]*len(testdial)

        testdial['dt_duration'] = [60*3]*len(testdial)
        testdial['dt_weight_post'] = weight + np.random.randn(len(testdial))
        testdial['dt_heart_rate_pre'] = expectedhrforage*np.ones([len(testdial),1])+np.round(2*np.random.randn(len(testdial),1))
        testlab['lt_value_calcium'] = [100]*len(testlab)

        #add simulated duration shortenings
        for event in range(1+round(len(testdial)/10)):
            row = np.random.randint(len(testdial)-3)
            shortness = np.random.randint(10,30)
            testdial['dt_duration'][row]-=shortness
            testdial['dt_heart_rate_pre'][row+1]+=shortness*0.6+np.round(4*np.random.randn())
            testdial['dt_heart_rate_pre'][row+2]+=shortness*0.3+np.round(2*np.random.randn())
            testdial['dt_heart_rate_pre'][row+3]+=shortness*0.15+np.round(1*np.random.randn())
            dialnum = strtodatenum(testdial.iloc[row]['dt_date'])
            for i,itrow in testlab.iterrows():
                effect = (dialnum-strtodatenum(itrow['lt_date']))
                if effect<0:
                    testlab['lt_value_calcium'][i]+=10/(1+effect)
        test_patient = dialysis_analysis.Patient(testpat.copy(),testdial.copy(),testlab.copy(),testhosp.copy(),testcomorb.copy())
        return test_patient

    def test_no_peeking(self):
        """
        To confirm that the tested data point is not affected. We perturb the test point in the patient data
        and check the predictions don't change.    
        """
        testpat = pd.DataFrame(
            data={'baseline_pt_pop_analysis_c': [1], 
                  'baseline_pt_date_fu_start': ['10.07.2008'], 
                  'baseline_pt_date_fu_stop': ['10.07.2009'], 
                  'baseline_pt_dialysis_vintage':[0],
                  'baseline_pt_age':60,
                  'baseline_pt_height':180,
                  'baseline_pt_gender_c':2})        
        
        
        testdial = pd.DataFrame(data={
            'dt_valid_session_d': ['Yes']*5, 
            'dt_date': ['13SEP2008','15SEP2008','17SEP2008','19SEP2008','21SEP2008']})
        testlab = pd.DataFrame(data={
            'lt_date': ['13SEP2008','21SEP2008']})
        testhosp = pd.DataFrame(data={
            'hp_date_start': ['13SEP2008','15SEP2009'],
            'hp_date_stop': ['14SEP2008','20SEP2009']})
        testcomorb = pd.DataFrame(data={})

        for v in Patient.labvars:
            testlab[v] = [np.NaN]*len(testlab)

        for v in Patient.dialvars:
            testdial[v] = [np.NaN]*len(testdial)

        testdial['dt_heart_rate_pre'] = [1,2,3,4,5]
        testlab['lt_value_calcium'] = [1,5]
        test_patient = Patient(testpat.copy(),testdial.copy(),testlab.copy(),testhosp.copy(),testcomorb.copy())
        test_prophets = test_patient.generate_all_prophets(ProphetCoregionalised,10,['num_date','days_since_dialysis'],['dt_heart_rate_pre'],['lt_value_calcium'])
        compute_results(test_prophets)

        testdial['dt_heart_rate_pre'][4] = 0
        test_patient = Patient(testpat,testdial,testlab,testhosp,testcomorb)
        test_prophets_perturbed_last_row = test_patient.generate_all_prophets(ProphetCoregionalised,10,['num_date','days_since_dialysis'],['dt_heart_rate_pre'],['lt_value_calcium'])
        compute_results(test_prophets_perturbed_last_row)

        #changing the last element shouldn't change the prediction.
        #there is some variation as I think optimisation is stochastic.
        assert np.abs(test_prophets[-1].res['mean'][0,0]-test_prophets_perturbed_last_row[-1].res['mean'][0,0])<0.01

        assert test_prophets[-1].get_actual()[0]==5 #ensure that the data point was actually changed...
        assert test_prophets_perturbed_last_row[-1].get_actual()[0]==0

    def test_population_model(self):
        """
        To test the population model, we use the test patients to generate prophets at the beginning of their time
        series, and compare a with popmodel and without popmodel set of prophets to see which does better.
        We assert than the popmodel version should have a lower RMSE and lower MAE.
        """

        print("Generating patients")
        test_patients = []
        for it in range(300):
            test_patients.append(self.build_test_patient(date(2008,7,10),date(2008,9,10)))

        print("Generating Prophets")
        prophets = []
        for p in test_patients:
            patient_prophets = p.generate_all_prophets(ProphetSimple,300,['num_date','days_since_dialysis'],
                                           ['dt_heart_rate_pre'],['lt_value_albumin'],
                                           delta_dialysis=[('dt_heart_rate_pre','grad',3),
                                                           ('dt_duration','abs')],
                                           delta_lab=[('lt_value_albumin','grad',5)],skipstep=10)
            prophets.extend(patient_prophets)
        dialysis_analysis.compute_results(prophets[0:1000])
        print("From %d patients, %d prophets have been created" % (len(test_patients), len(prophets)))
        print("Building Population Prior Model")
        prior_models = dialysis_analysis.build_population_prior_model(prophets)

        light_pr, _ = prior_models[0].predict(np.array([[60,35,60,156,1]]))
        heavy_pr, _ = prior_models[0].predict(np.array([[60,35,120,156,1]]))

        print("Light pulse rate %0.0f, Heavy pulse rate %0.0f" % (light_pr, heavy_pr))
        assert heavy_pr[0,0]>light_pr[0,0]+25, "The heavier test patients prior should be larger (about 40 bpm) greater than the lightest test patients."

        print("Generating new set of patients")
        #generate a new set of test patients (so we aren't peeking at the training set for the population model, above)
        new_test_patients = []
        for it in range(100):
            new_test_patients.append(self.build_test_patient(date(2008,7,10),date(2008,9,10)))

        print("Generating New prophets (no pop prior)")
        prophets = []
        for p in new_test_patients:
            patient_prophets = p.generate_all_prophets(ProphetCoregionalised,300,['num_date','days_since_dialysis'],
                                           ['dt_heart_rate_pre'],['lt_value_albumin'],
                                           delta_dialysis=[('dt_heart_rate_pre','grad',3),
                                                           ('dt_duration','abs')],
                                           delta_lab=[('lt_value_albumin','grad',5)],stopearly=12)
            prophets.extend(patient_prophets)
        print("From %d patients, %d prophets have been created" % (len(test_patients), len(prophets)))  
        print("Computing results")
        dialysis_analysis.compute_results(prophets[0::20])


        print("Generating New Prophets (with pop prior)")
        pm_prophets = []
        for p in new_test_patients:
            patient_prophets = p.generate_all_prophets(ProphetCoregionalised,300,['num_date','days_since_dialysis'],
                                           ['dt_heart_rate_pre'],['lt_value_albumin'],
                                           delta_dialysis=[('dt_heart_rate_pre','grad',3),
                                                           ('dt_duration','abs')],
                                           delta_lab=[('lt_value_albumin','grad',5)],stopearly=12,
                                           prior_models = prior_models)
            pm_prophets.extend(patient_prophets)
        print("From %d patients, %d prophets have been created" % (len(test_patients), len(pm_prophets)))
        print("Computing results")
        dialysis_analysis.compute_results(pm_prophets[0::20])

        pm_mae, pm_rmse = dialysis_analysis.compute_errors(pm_prophets)
        mae, rmse = dialysis_analysis.compute_errors(prophets)

        print("Results:")
        print("No population prior model:")
        print("   MAE %0.2f, RMSE %0.2f" % (mae[0], rmse[0]))
        print("With population prior model:")
        print("   MAE %0.2f, RMSE %0.2f" % (pm_mae[0], pm_rmse[0]))

        assert pm_mae<mae, "The population model should improve the MAE"
        assert pm_rmse<rmse, "The population model should improve the RMSE"
        assert np.all(np.abs(np.diff(prophets[-1].Y-pm_prophets[-1].Y,axis=0))<1e-5),"the differences between the values in Y for the same prophet should all be the same (and just differ by the same prior)"

        ##here we try using a patient that's been used to create this model
        #(it should raise an assertion to stop us from peeking)
        #self.assertRaises not working:
        try:
            test_patients[0].generate_all_prophets(ProphetCoregionalised,300,['num_date','days_since_dialysis'],
                                       ['dt_heart_rate_pre'],['lt_value_albumin'],
                                       delta_dialysis=[('dt_heart_rate_pre','grad',3),
                                                       ('dt_duration','abs')],
                                       delta_lab=[('lt_value_albumin','grad',5)],stopearly=20,
                                       prior_models = prior_models)
            assert True, "This should have failed on our attempt at reusing a patient"
        except AssertionError:
            pass #all good - we expect this error
        


        print("Success")

    def test_delta_model(self):
        test_patients = []
        for it in range(30):
            test_patients.append(self.build_test_patient(date(2008,7,10),date(2008,9,10)))
        prophets = []
        for test_patient in test_patients:
            prophets.extend(test_patient.generate_all_prophets(ProphetCoregionalised, 100, #time window
                                                           ['num_date','days_since_dialysis'], #inputdialysis
                                                           ['dt_heart_rate_pre'], #outputdialysis
                                                           ['lt_value_calcium'],#outputlab,
                                                           delta_dialysis=[('dt_heart_rate_pre','grad',3),
                                                           ('dt_duration','diff')],
                                                           delta_lab=[('lt_value_albumin','grad',5)],skipstep=10)) #delta_lab
        dialysis_analysis.compute_results(prophets[0:300])

        deltamodel = DeltaModel(prophets)

        self = TestDialysisAnalysis()
        test_patients = []
        for it in range(30):
            test_patients.append(self.build_test_patient(date(2008,7,10),date(2008,9,10)))
        test_prophets = []
        for test_patient in test_patients:
            test_prophets.extend(test_patient.generate_all_prophets(ProphetCoregionalised, 100, #time window
                                                           ['num_date','days_since_dialysis'], #inputdialysis
                                                           ['dt_heart_rate_pre'], #outputdialysis
                                                           ['lt_value_calcium'],#outputlab,
                                                           delta_dialysis=[('dt_heart_rate_pre','grad',3),
                                                           ('dt_duration','diff')],
                                                           delta_lab=[('lt_value_albumin','grad',5)],skipstep=10)) #delta_lab

        dialysis_analysis.compute_results(test_prophets)
        mae, rmse = dialysis_analysis.compute_errors(test_prophets)
        print("Accuracy without Delta Model")
        print(mae, rmse)
        deltamodel.add_delta_to_prophets(test_prophets)
        dm_mae, dm_rmse = dialysis_analysis.compute_errors(test_prophets)
        print("Accuracy with Delta Model")
        print(dm_mae, dm_rmse)
        assert np.all([a>b+1 for a,b in zip(mae,dm_mae)]), "Delta model failed to improve prediction accuracy (MAE)"
        assert np.all([a>b+1 for a,b in zip(rmse,dm_rmse)]), "Delta model failed to improve prediction accuracy (RMSE)"
if __name__ == '__main__':
    unittest.main()    