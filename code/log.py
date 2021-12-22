
from sklearn.metrics import confusion_matrix
import numpy as np
#helper method to evaluate results that may contain zero- or NaN-values due to Daze-library is having a hard time dealing with them 
#Note that default mode is set to overwrite, to change that use either 'x' -> returns an error if file already exists, 'a' -> appends existing file
def log(target_list, prediction_list, labels, number_of_classes, logFile_Name, write_mode='w'):    
    #src: https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    cnf_matrix = confusion_matrix(target_list, prediction_list)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    samples_per_class = cnf_matrix.sum(axis=1)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, F1 = [], [], [], [], [], [], [], [], []

    for l in range(number_of_classes):
        # Sensitivity, hit rate, recall, or true positive rate
        if(TP[l]+FN[l] == 0.0):
            TPR.append(0.0)
        else:
            TPR.append(TP[l]/(TP[l]+FN[l]))
        # Specificity or true negative 
        if(TN[l]+FP[l] == 0.0):
            TNR.append(0.0)
        else:
            TNR.append(TN[l]/(TN[l]+FP[l]))
        # Precision or positive predictive value
        if(TP[l]+FP[l] == 0.0):
            PPV.append(0.0)
        else:
            PPV.append(TP[l]/(TP[l]+FP[l]))
        # Negative predictive value
        if(TN[l]+FN[l] == 0.0):
            NPV.append(0.0)
        else:        
            NPV.append(TN[l]/(TN[l]+FN[l]))
        # Fall out or false positive rate
        if(FP[l]+TN[l] == 0.0):
            FPR.append(0.0)
        else:    
            FPR.append(FP[l]/(FP[l]+TN[l]))
        # False negative rate
        if(TP[l]+FN[l] == 0.0):
            FNR.append(0.0)
        else:
            FNR.append(FN[l]/(TP[l]+FN[l]))
        # False discovery rate
        if(TP[l]+FP[l] == 0.0):
            FDR.append(0.0)
        else:    
            FDR.append(FP[l]/(TP[l]+FP[l]))
        # Overall accuracy
        if(TP[l]+FP[l]+FN[l]+TN[l] == 0.0):
            ACC.append(0.0)
        else:        
            ACC.append((TP[l]+TN[l])/(TP[l]+FP[l]+FN[l]+TN[l]))
        # F1-Score
        if(TP[l]+FP[l]+FN[l] == 0.0):
            F1.append(0.0)
        else:    
            F1.append((2*TPR[l]) / (2*TP[l] + FP[l] + FN[l]))

    f = open(r'{}'.format(logFile_Name), write_mode)

    f.write(str(cnf_matrix)+"\n")
    f.write("Labels: ")
    f.write(str(labels)+"\n")
    f.write("#samples per class:"+ str(samples_per_class)+"\n")
    f.write("FP:"+ str(FP)+"\n")
    f.write("FN:"+ str(FN)+"\n")
    f.write("TP:"+ str(TP)+"\n")
    f.write("TN:"+ str(TN)+"\n")
    #average
    f.write("TPR_avg:"+ str(np.mean(TPR))+"\n") #sensitivity, recall
    f.write("TNR_avg:"+ str(np.mean(TNR))+"\n") #specificity, selectivity
    f.write("PPV_avg:"+ str(np.mean(PPV))+"\n") #precision
    f.write("NPV_avg:"+ str(np.mean(NPV))+"\n") #negative prediction value
    f.write("FPR_avg:"+ str(np.mean(FPR))+"\n") #fall-out
    f.write("FNR_avg:"+ str(np.mean(FNR))+"\n") #miss rate
    f.write("FDR_avg:"+ str(np.mean(FDR))+"\n") #false discovery rate
    f.write("ACC_avg:"+ str(np.mean(ACC))+"\n") 
    f.write("F1_avg:"+ str(np.mean(F1))+"\n") #harmonic mean of precision and sensitivity
    #per class
    f.write("TPR:"+ str(TPR)+"\n")
    f.write("TNR:"+ str(TNR)+"\n")
    f.write("PPV:"+ str(PPV)+"\n")
    f.write("NPV:"+ str(NPV)+"\n")
    f.write("FPR:"+ str(FPR)+"\n")
    f.write("FNR:"+ str(FNR)+"\n")
    f.write("FDR:"+ str(FDR)+"\n")
    f.write("ACC:"+ str(ACC)+"\n")
    f.write("F1:"+ str(F1)+"\n")

    f.close()