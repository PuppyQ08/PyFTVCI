import numpy as np
import math
import itertools
from collections import Counter
import sys
from numpy import linalg
from numba import jit
import time

#This function is obtained from pyvci plz see github resource code
#this function is for get the combination of all excited states
#I used max number of excited level for each mode 
#eg: nmode  = 3, maxlvl = 8(0-7) then we have 8*8*8 combines since 0 the vacuum one counts.
#if you want to get the sum max number of states like Dr.Yagi's code plz modify this fn.
#sys.stdout = open("vci_test_output.txt","w")

#function to generate the combination
t0 = time.time()
def loopfn(n,maxn):
    if n>1:
        rt = []
        for x in range(maxn):
            k = loopfn(n-1,maxn)
            for i in range(len(k)):
                k[i].append(x)
            rt += k
        return rt
    else:
        rt = []
        for x in range(maxn):
            rt.append([x])
        return rt




#It reads in the QFF force constants
def readSindoPES(filepath,nmode):
    w_omega = np.zeros(nmode)
    FCQ3 = np.zeros((nmode,nmode,nmode)) #Coefficient in Q (normal coordinates)
    #XXX Coefficient includes the 1/2 1/3! 1/4! in the front!!
    #Dr.Yagi used dimensionless q as unit so we need to transfer from q to Q by times sqrt(w1*w2*.../hbar^(...))
    FCQ4 = np.zeros((nmode,nmode,nmode,nmode))
    with open(filepath) as f:
        flines = f.readlines()
        FCstartidx = nmode#if the scanning and fetching index failed , it will be over the size of matrix. 
        Omgstartidx = nmode 
        for idx in range(len(flines)):
            if (flines[idx].split()[0] == "DALTON_FOR_MIDAS"):
                FCstartidx = idx
            if (flines[idx].split()[0] == "SCALING"):
                Omgstartidx = idx
        widx = 0
        for idx in range(Omgstartidx+1,FCstartidx):
            tl = flines[idx].split()#shortcut for this line
            w_omega[widx] = float(tl[0])
            widx += 1
        for idx in range(FCstartidx+1, len(flines)):
            tl = flines[idx].split()#shortcut for this line
            leng = len(tl)
            if (leng == 4):
                #third order force constant
                FCQ3[int(tl[1])-1,int(tl[2])-1,int(tl[3])-1] = float(tl[0])*math.sqrt(w_omega[int(tl[1])-1]*w_omega[int(tl[2])-1]*w_omega[int(tl[3])-1])
            if (leng == 5):
                #forth order force constant
                FCQ4[int(tl[1])-1,int(tl[2])-1,int(tl[3])-1,int(tl[4])-1] = float(tl[0])*math.sqrt(w_omega[int(tl[1])-1]*w_omega[int(tl[2])-1]*w_omega[int(tl[3])-1]*w_omega[int(tl[4])-1])
    return w_omega,FCQ3,FCQ4

def EvaluationList(nmode,w_omega,maxn,maxorder):
    #I used the combination to determine which operator can give us result.
    #The 1st is to indicate which normal mode is it.
    #The 2nd is to indicate which operator: 0-5 : zero(no operator, assume the basis function is orthogonal, partial deriv Q^2, Q, Q^2, Q^3, Q^4. Here we used QFF so the max order of operator is 4 and total number is 5
    #The 3rd is to the which level n is, n is the bigger one than n' 
    #The 4th is the difference between n and n'
    Evlst = np.zeros((nmode,maxorder,maxn,maxorder))
    for i in range(nmode):
        for n in range(maxn):
            Evlst[i,0,n,0] = - w_omega[i]*(n+0.5)
            Evlst[i,0,n,2] = w_omega[i]*math.sqrt(n*(n-1))/2
            Evlst[i,1,n,1] = math.sqrt(n/2/w_omega[i])
            Evlst[i,2,n,0] = (n+0.5)/w_omega[i]
            Evlst[i,2,n,2] = math.sqrt(n*(n-1))/2/w_omega[i]
            Evlst[i,3,n,1] = 3*n/2/w_omega[i]*math.sqrt(n/2/w_omega[i])
            Evlst[i,3,n,3] = math.sqrt(n*(n-1)*(n-2))/(2*w_omega[i]*math.sqrt(2*w_omega[i]))
            Evlst[i,4,n,0] = (6*n*(n+1)+3)/4/(w_omega[i]**2)
            Evlst[i,4,n,2] =  (n-0.5)*math.sqrt(n*(n-1))/(w_omega[i]**2)
            Evlst[i,4,n,4] = math.sqrt(n*(n-1)*(n-2)*(n-3))/4/(w_omega[i]**2)
    return Evlst


def VCImatrix(linrComb,Evlst,nmode,maxorder,FCQ3,FCQ4,Vref):
    leng = len(linrComb)
    VCImtrx = np.zeros((leng,leng))
    #VCI matrix is Hermitian 
    for i in range(leng):
        for j in range(i,leng):
            lhs = linrComb[i] 
            rhs = linrComb[j]
            sumofoperator = 0
            #parse each operator first
            #operator partial deriv:

            for optidx in range(nmode):
                #parse each mode in |xxxx>
                multply = 1 #the multiply product of each mode
                for modeidx in range(nmode):
                    n = max(lhs[modeidx],rhs[modeidx])
                    diff = abs(lhs[modeidx] - rhs[modeidx])
                    if (modeidx == optidx and diff < maxorder): #the operator works on the correspoding Q
                        multply *= -0.5*Evlst[modeidx,0,n,diff]
                    else: #check if they are orthogonal if not, then zero
                        if (diff!=0):
                            multply *= 0 
                            break
                sumofoperator += multply
            #operator Vref
            #Vref is a constant so only orthogonal can give the value
            multply = 1
            for modeidx in range(nmode):
                diff = abs(lhs[modeidx] - rhs[modeidx])
                if (diff!=0):
                    multply *=0
                    break
            sumofoperator += multply*Vref
            #operator sum FiQi
            #for harmonic oscilator Fi = 0 so we pass this term

            #Fij = w_omega ^2 for i == j
            for forceidx in range(nmode):
                multply = 1
                #print(" For F_ii i is ",forceidx)
                for modeidx in range(nmode):
                    diff = abs(lhs[modeidx] - rhs[modeidx])
                    n = max(lhs[modeidx],rhs[modeidx])
                    if (forceidx == modeidx and diff < maxorder):
                        multply *= 0.5*Evlst[modeidx,2,n,diff]
                    else:
                        if (diff !=0):
                            multply *= 0
                            break
                
                multply*=(w_omega[forceidx]**2)
                sumofoperator += multply

            #print("------------------")
            #print("For Fijk ")
            #operator sum Fijk Qi Qj Qk
            for ii in range(nmode):
                for jj in range(nmode):
                    for kk in range(nmode):
                        multply = 1
                        eachcount = Counter([ii,jj,kk])
                        tempstore = []
                        for modeidx in range(nmode):
                            diff = abs(lhs[modeidx] - rhs[modeidx])
                            n = max(lhs[modeidx],rhs[modeidx])
                            numberofmodeinFC = eachcount[modeidx]
                            if (numberofmodeinFC != 0 and diff < maxorder):
                                multply*= Evlst[modeidx,numberofmodeinFC,n,diff]
                            else:
                                if(diff != 0):
                                    multply*=0
                                    break
                        multply*=FCQ3[ii,jj,kk]
                        sumofoperator+=multply
            #operator sum Fijkl Qi Qj Qk Ql
            for ii in range(nmode):
                for jj in range(nmode):
                    for kk in range(nmode):
                        for ll in range(nmode):
                            multply = 1
                            eachcount = Counter([ii,jj,kk,ll])
                            for modeidx in range(nmode):
                                diff = abs(lhs[modeidx] - rhs[modeidx])
                                n = max(lhs[modeidx],rhs[modeidx])
                                numberofmodeinFC = eachcount[modeidx]
                                if (numberofmodeinFC != 0 and diff < maxorder):
                                    multply*= Evlst[modeidx,numberofmodeinFC,n,diff]
                                else:
                                    if(diff!=0):
                                        multply*=0
                                        break #break the innerest loop since they will be all zero.
                            multply*=FCQ4[ii,jj,kk,ll]
                            sumofoperator+=multply
            VCImtrx[i,j] = VCImtrx[j,i] = sumofoperator
    return VCImtrx

def DiagonalVCI(VCImtrx):
    w,v = linalg.eigh(VCImtrx)
    HatreeTocm = 219474.63
    print(w)
    for i in range(len(w)):
        print("_+++++++++++++++++++++++++++++++++++++")
        print(w[i]*HatreeTocm)
        print("Then the Coeff")
        print(v[:,i])
    print(w*HatreeTocm)
    return w,v
    
def ThemoCalc(Temprt,Energylist,ret):
    #kb = 1 at a.u.
    #Calculate Grand partition function, grand potential and internal energy based on grand canonical ensemble.
    #Grand partition function: GPF
    b_beta = 1/(Temprt)
    print(b_beta)
    GPF_Xi = 0
    for eachE in Energylist:
        GPF_Xi += math.exp(-b_beta*eachE)
        
    #grand potential
    GP_Omg = -math.log(GPF_Xi)/b_beta
    #internal energy U
    IE_U = 0
    for eachE in Energylist:
        IE_U += eachE*math.exp(-b_beta * eachE)
    IE_U/=GPF_Xi
    #entropy S
    entropy_S = 0
    #just for math domain error 
    for eachE in Energylist:
        entropy_S += eachE*math.exp(-b_beta*eachE)
    entropy_S /= (Temprt*GPF_Xi)
    entropy_S += math.log(GPF_Xi)

    ret[0] = GPF_Xi
    ret[1] = GP_Omg
    ret[2] = IE_U
    ret[3] = entropy_S
    print("Xi, Omg, U ,S is")
    print(GPF_Xi)
    print(GP_Omg) 
    print(IE_U)
    print(entropy_S) 
    print("verify")
    print(IE_U-Temprt*entropy_S)
    



#Bose-Einstein statistics
def Bose_EinsteinStat(Temprt,w_omega,ret):
    b_beta= 1/Temprt
    #f_i 
    f_i = np.zeros(len(w_omega))
    for i in range(len(w_omega)):
        f_i[i] = 1/(1-math.exp(-b_beta*w_omega[i]))
    print(f_i)
    #partition function
    GPF_Xi = 1
    for ii in range(len(w_omega)):
        #GPF_Xi *=  math.exp(-b_beta*eachw/2)/(1-math.exp(-b_beta*eachw))
        GPF_Xi *=  math.exp(-b_beta*w_omega[ii]/2)*f_i[ii]
    #grand potential 
    GP_Omg = 0
    for eachw in w_omega:
        GP_Omg += 0.5*eachw +  math.log(1-math.exp(-b_beta*eachw))/b_beta
    #internal energy
    IE_U = 0
    for eachw in w_omega:
        IE_U += 0.5*eachw +  eachw*math.exp(-b_beta*eachw)/(1-math.exp(-b_beta*eachw))
    #entropy
    entropy_S = 0
    for eachw in w_omega:
        entropy_S += - math.log(1-math.exp(-b_beta*eachw)) + eachw*math.exp(-b_beta*eachw)/(Temprt*(1-math.exp(-b_beta*eachw)))
    ret[0] = GPF_Xi
    ret[1] = GP_Omg
    ret[2] = IE_U
    ret[3] = entropy_S
    print("analytical Bose-Einstein stat result:")
    print("Xi, Omg, U ,S is")
    print(GPF_Xi)
    print(GP_Omg) 
    print(IE_U)
    print(entropy_S) 
    print("verify")
    print(IE_U-Temprt*entropy_S)

#FCI bose-einstein with finite N
def FiniteBE(Temprt,w_omega,maxn,ret):
    N = maxn -1 
    b_beta = 1/Temprt
    f_i = np.zeros(len(w_omega))
    tildef_i = np.zeros(len(w_omega))
    for i in range(len(w_omega)):
        f_i[i] = 1/(1 - math.exp(- b_beta * w_omega[i]))
        tildef_i[i] = 1/(1 - math.exp( - b_beta * (N+1) * w_omega[i]))
    #partition function
    GPF_Xi = 1
    for eachw in w_omega:
        GPF_Xi *= math.exp(-b_beta*eachw/2)*(1-math.exp(-b_beta*(N+1)*eachw))/(1-math.exp(-b_beta*eachw))
    #Grand Potential
    GP_Omg = 0
    for ii in range(len(w_omega)):
        GP_Omg += 0.5 * w_omega[ii] - math.log(f_i[ii]/tildef_i[ii])/b_beta
    #Internal Energy
    IE_U = 0
    for ii in range(len(w_omega)):
        IE_U += 0.5 * w_omega[ii] + w_omega[ii]*(f_i[ii] - 1) - (N + 1) * w_omega[ii] * (tildef_i[ii] - 1)
    #entropy S
    entropy_S = 0
    for ii in range(len(w_omega)):
        entropy_S += (math.log(f_i[ii]/tildef_i[ii])/b_beta + w_omega[ii]*(f_i[ii]-1) - (N+1)*w_omega[ii]*(tildef_i[ii]-1))/Temprt
    ret[0] = GPF_Xi
    ret[1] = GP_Omg
    ret[2] = IE_U
    ret[3] = entropy_S
    print("finite analytical Bose-Einstein stat result:")
    print("Xi, Omg, U ,S is")
    print(GPF_Xi)
    print(GP_Omg) 
    print(IE_U)
    print(entropy_S) 
    print("verify")
    print(IE_U-Temprt*entropy_S)

def multitask(Temprt,Energylist,thermoresults):
    for ii in range(len(Temprt)):
        ThemoCalc(Temprt[ii],Energylist,thermoresults[ii,0,:])
        Bose_EinsteinStat(Temprt[ii],w_omega,thermoresults[ii,1,:])
        FiniteBE(Temprt[ii],w_omega,maxn,thermoresults[ii,2,:])
    np.save("../data/thermoresult",thermoresults)

    

#number of normal mode like H2O is 3 here I mannuly set up but later can read in from input file
nmode = 3
#maxium number of excited level for each mode.
maxn = 3 
#maxium order of force field like QFF is 5 since we have kinetic operator at front
maxorder = 5
#by default Vref = 0
Vref= 0 
#Temperature unit is K then transfer to a.u. by 3.1577464*10^5 ( Eh/kb)
#Temprt = 100000
Ehbykb = 3.1577464*100000
Temprt = np.array([100,1000,10000,100000,1000000,10000000,100000000])
Temprt = Temprt/Ehbykb
thermoresults = np.zeros((7,3,4))


linrComb = loopfn(nmode,maxn)
#print(linrComb)
print(len(linrComb))
filepath = "../../data/prop_no_1.mop"

w_omega,FCQ3,FCQ4 = readSindoPES(filepath,nmode)
Evlst = EvaluationList(nmode,w_omega,maxn,maxorder)# The list of the evaluation from Hermes xvscf table.
VCImtrx = VCImatrix(linrComb,Evlst,nmode,maxorder,FCQ3,FCQ4,Vref)
Energylist, Coefficient = DiagonalVCI(VCImtrx)

#np.save("../data/Energylistwithmaxn"+str(maxn),Energylist)
#Energylist = np.load("../data/Energylistwithmaxn"+str(maxn)+".npy")

#for ii in range(len(Temprt)):
#ThemoCalc(Temprt[0],Energylist,thermoresults[0,0,:])
#Bose_EinsteinStat(Temprt[0],w_omega,thermoresults[1,1,:])
#FiniteBE(Temprt[0],w_omega,maxn,thermoresults[0,2,:])

#multitask(Temprt,Energylist,thermoresults)

t1 = time.time()
print("time is /min",(t1-t0)/60)
#sys.stdout.close()

