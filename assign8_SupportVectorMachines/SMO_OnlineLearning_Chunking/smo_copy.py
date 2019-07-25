import numpy as np
from sklearn.model_selection import train_test_split
import kernel as kern
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from numpy.linalg import norm

#w = []
b = 0 
kernel = "Linear"
def create_kernel_matrix(X,Y):
    global kernel
    if(kernel=='Linear'):
        k=X@Y.T
    elif(kernel=='Polynomial'):
        k=np.power(X@Y.T,2)
    elif(kernel=='E'):
        sigma =40
        k= np.exp((-1/(2*sigma**2))*np.linalg.norm(X[:,None]-Y, axis=2))
    elif(kernel=='H'):
        c=10
        k = 1/np.sqrt(np.linalg.norm(X[:,None]-Y, axis=2)+c)
    return k


def get_predicted_output(test_set,support_input,support_alpha,support_y,b):
    global kernel
    predicted = np.zeros(test_set.shape[0])
    for i in range(test_set.shape[0]) :   
        sum = 0
        for j in range(support_alpha.shape[0]) :
            sum = sum + (support_alpha[j] * support_y[j] *  kern.kernel_function(kernel,test_set[i],support_input[j]))
        predicted[i] = sum + b 
           
    return predicted
        
 
def get_predicted(alpha,input,output,index) :
    global b,kernel
    sum = 0
    
    for i in range(output.shape[0]):
        sum = sum + (alpha[i] * output[i] * kern.kernel_function(kernel,input[i],input[index])) 
    predicted = sum- b
    return predicted
    
        

def takestep(c,input,output,alpha,index1,index2,E1) :
#    global w,b,kernel
    global b,kernel
    eps = 10**-3
    if index1 == index2 :
        return 0
    alpha2 = alpha[index2]
    y2 = output[index2]

    alpha1 = alpha[index1]
    y1 = output[index1]
    
    E2 = get_predicted(alpha,input,output,index2) - y2
    
    s = y1 * y2
    
    if y1 == y2 :
        L = max(0, alpha2 + alpha1 - c)
        H = min(c, alpha2 + alpha1)
    else :
        L = max(0, alpha2 - alpha1)
        H = min(c, c + alpha2 - alpha1)
    if L == H :
        return 0
    k11 = kern.kernel_function(kernel,input[index1],input[index1])
    k12 = kern.kernel_function(kernel,input[index1],input[index2])
    k22 = kern.kernel_function(kernel,input[index2],input[index2])
    eta = k11 + k22 - (2*k12)
    if eta > 0 :
        a2 = alpha2 +  y2*(E1 - E2) / eta
        if a2 < L :
            a2 = L
        elif a2 > H :
            a2 = H
    else :
        a2 = L
        f1 = y1 * (E1 + b) - (alpha1 * k11) - (s * a2 * k12)
        f2 = y2 * (E2 + b) - (a2 * k22) - (s * alpha1 * k12)
        L1 = alpha1 + s * (a2 - L)
        Lobj = (L1 * f1) + (L * f2) + (0.5 * L1 ** 2 * k11) + (0.5 * L**2 * k22) + (s * L * L1 * k12)
       
        
        a2 = H
        f1 = y1 * (E1 + b) - (alpha1 * k11) - (s * a2 * k12)
        f2 = y2 * (E2 + b) - (a2 * k22) - (s * alpha1 * k12)
        H1 = alpha1 + s * (a2 - H)
        Hobj = (H1 * f1) + (H * f2) + (0.5 * H1 ** 2 * k11) + (0.5 * H**2 * k22) + (s * H * H1 * k12)
        
       
        
        if Lobj < (Hobj - eps) :
            a2 = L
        elif Lobj > (Hobj + eps):
            a2 = H
        else :
            a2 = alpha2
    if abs(a2 - alpha2) < (eps*(a2 + alpha2 + eps)) :
        return 0
    a1 = alpha1 + (s*(alpha2 - a2))
      
    b1 = E1 + y1 * (alpha1 - a1) * k11 + y2 * (alpha2 - a2) * k12 + b
    
    b2 = E2 + y1 * (alpha1 - a1) * k12 + y2 * (alpha2 - a2) * k22 + b
    

    if (a1 != 0 and a1 != c) :
        b = b1
    
    elif (a2 != 0 and a2 != c) :
        b = b2
    elif (L != H) :
        b = (b1 + b2) / 2
#    w = w + (y1 * (a1 - alpha1) * input[index1]) + (y2 * (a2 - alpha2) * input[index2])
    alpha[index1] = a1
    alpha[index2] = a2
    return 1    

def get_nonzero_c_alpha(c,alpha):
    alpha_index_temp = []
    alpha_temp = []
    for i in range(alpha.shape[0]):
        if alpha[i] > 0 and alpha[i] < c :
            alpha_temp.append(alpha[i])
            alpha_index_temp.append(i)
    return alpha_temp, alpha_index_temp
    

def second_choice_heuristic(input,output,alpha,c,E2) :
    global b
    errorcache = np.zeros(output.shape)
    for i in range(output.shape[0]) :
        errorcache[i] = (get_predicted(alpha,input,output,i) - b) - output[i] 
    alpha_temp, alpha_index = get_nonzero_c_alpha(c,alpha)
    error = errorcache[alpha_index] -   E2
    max_index = np.argmax(abs(error))
    return max_index
    
   


def examineExample(input,output,c,alpha,index1):
    global b
    y1 = output[index1]
    alpha1 = alpha[index1]
    E1 = get_predicted(alpha,input,output,index1) - y1
    r1 = E1 * y1
    if (r1 < -10**-5 and alpha1 < c) or (r1 > 10**-5 and alpha1 > 0) :
        alpha_temp, alpha_index = get_nonzero_c_alpha(c,alpha)
        if len(alpha_temp) > 1 :
            index2 = second_choice_heuristic(input,output,alpha,c,E1)
            if takestep(c,input,output,alpha,index1,index2,E1):
                return 1
        rand_index = np.random.permutation(len(alpha_temp))
        for index2 in rand_index :
            if takestep(c,input,output,alpha,index1,index2,E1):
                return 1
        rand_index = np.random.permutation(input.shape[0])
        for index2 in rand_index:
            if takestep(c,input,output,alpha,index1,index2,E1):
                return 1
    return 0


def smo(input,output,c) :
#    global w,b
    global b
    b = 0
    alpha = np.zeros(output.shape)
    w = np.zeros(input.shape[1])
    numChanged = 0
    examineAll = 1
    while  numChanged > 0 or examineAll  :
    
            numChanged = 0
            if(examineAll) :
                for index in range(output.shape[0]) :
                    numChanged += examineExample(input,output,c,alpha,index)
            else :
                alpha_temp, alpha_index = get_nonzero_c_alpha(c,alpha)
                for index in alpha_index :
                    numChanged += examineExample(input,output,c,alpha,index)
            if examineAll == 1 :
                examineAll = 0
            elif numChanged == 0 :
                examineAll = 1
    return alpha,b


            
def smo_kfold(data) :

    skf = KFold(n_splits=2, random_state=42, shuffle=True)
    prev = -np.inf
    input = data[:,:-1]
    output = data[:,-1] 
    
    for e in range(-5,3,2):
        accuracy = []
        c = 2**e
        for train_index, test_index in skf.split(input, output):
            input_train, input_test = input[train_index], input[test_index]
            output_train, output_test = output[train_index], output[test_index]
            alpha = smo(input_train,output_train,c)
            support_alpha = alpha[alpha > 0]
            support_input = input_train[alpha > 0]
            support_y = output_train[alpha>0]
            predicted = get_predicted_output(input_test,support_input,support_alpha,support_y,b)
            
            for i in range(predicted.shape[0]) :
                if predicted[i] < 0 :
                    predicted[i] = -1
                else :
                    predicted[i] = 1
#            print(accuracy_score(output_test, predicted)*100)
            accuracy.append(accuracy_score(output_test, predicted)*100)
        average_accuracy = sum(accuracy) / len(accuracy)
#        print(average_accuracy)
        if prev < average_accuracy :
            best_c  = c
            prev = average_accuracy
    return best_c
   
    
    
def compute(input,output,kernel_val): 
#    global w,b,kernel
    global b,kernel
    kernel = kernel_val
    indices = np.arange(input.shape[0])
    prev = -np.inf
    for holdout in range(0,1):
        
        input_train, input_test, output_train, output_test,train_index,test_index = train_test_split(input, output, indices, test_size=0.3, stratify=output)
        
        training = np.zeros([input_train.shape[0],input_train.shape[1] + 1])
        training[:,:-1] = input_train
        training[:,-1] = output_train
        
        
        
        testing = np.zeros([input_test.shape[0],input_test.shape[1] + 1])
        testing[:,:-1] = input_test
        testing[:,-1] = output_test
        
        c = smo_kfold(training)
        alpha = smo(input_train,output_train,c)
        support_alpha = alpha[alpha > 0]
        support_input = input_train[alpha > 0]
        support_y = output_train[alpha>0]
        
        predicted = get_predicted_output(input_test,support_input,support_alpha,support_y,b)
        
        for i in range(predicted.shape[0]) :
            if predicted[i] < 0 :
                predicted[i] = -1
            else :
                predicted[i] = 1
                
        accuracy  = accuracy_score(output_test, predicted)
#       print(accuracy)        
        
        sigma_dual = 0        
        sigma_alpha = 0
    
        for i in range(support_input.shape[0]):
            sigma_alpha = sigma_alpha + support_alpha[i]
            for j in range(support_input.shape[0]):
                 sigma_dual = sigma_dual + (support_alpha[i] * support_alpha[j] * support_y[i] * support_y[j] * kern.kernel_function(kernel,support_input[i],support_input[j]))
                 
        dual_val = sigma_alpha - (0.5 * sigma_dual)
        
        eta = np.zeros(predicted.shape)
    
        for i in range(predicted.shape[0]):
            eta[i] = max(0,1- (output_test[i] * predicted[i]))
            
        norm_f_square = 0  
#        if kernel == "Linear" :
#            norm_f_square = (norm(w))**2
        if kernel == "Polynomial" :
            norm_f_square = sigma_dual
            
        primal = (c * np.sum(eta)) + (norm_f_square / 2)
        
        dual_gap = primal - dual_val
        
        if prev < accuracy :
#            final_w = w
            prev = accuracy     
            final_b = b
            final_c  = c
            final_dual = dual_val
            final_primal= primal
            final_dual_gap = dual_gap
            
#    print("Best w",final_w)
    print("Best Accuracy",prev)
    print("Best c",final_c)
    print("Best b",final_b)
    print("Best dual value",final_dual)
    print("Best primal value",final_primal)
    print("Dual gap",final_dual_gap)
       
    return alpha, final_b
import csv

def getdata():
    reader = csv.reader(open("data4.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
    alphas,b=smo(X_train,y_train,1000)
    print(alphas)
    index=np.where(alphas>0)[0]
    alpha_sv = alphas[index]
    X_sv = X_train[index]
    y_sv = y_train[index]
    
    y_sv=y_sv.reshape(y_sv.shape[0],1)
    alpha_sv=alpha_sv.reshape(alpha_sv.shape[0],1)
    alpha_y= alpha_sv*y_sv    
    #Creation of Kernel Matrix for Testing
    K=create_kernel_matrix(X_sv,X_test)    
    f=np.dot(alpha_y.T,K)
    f=f.reshape(f.shape[1],1)
    f_tilda=f+b
    prediction=np.sign(f_tilda)  
    score=accuracy_score(y_test, prediction, sample_weight=None)
    print(score)
getdata()