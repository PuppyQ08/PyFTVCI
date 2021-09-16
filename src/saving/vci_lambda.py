import numpy as np
import math
import itertools
from collections import Counter
import sys
from numpy import linalg
from numba import jit
import time
from multiprocessing import Process, Queue
import csv

#This function is obtained from pyvci plz see github resource code
#this function is for get the combination of all excited states
#I used max number of excited level for each mode 
#eg: nmode  = 3, maxlvl = 8(0-7) then we have 8*8*8 combines since 0 the vacuum one counts.
#if you want to get the sum max number of states like Dr.Yagi's code plz modify this fn.
#XXX add lambda verification
#sys.stdout = open("vci_test_output.txt","w")

#function to generate the combination
t0 = time.time()
class VCIthermo:
    def __init__(self,Lambd,Temprt,maxn,calVCI):#calVCI= 1 or filename
        Vref= 0 
        maxorder = 5
        nmode = 3
        filepath = "../data/prop_no_1.mop"
        w_omega,FCQ3,FCQ4 = self.readSindoPES(filepath,nmode)
        linrComb = self.loopfn(nmode,maxn)
        Evlst = self.EvaluationList(nmode,w_omega,maxn,maxorder)# The list of the evaluation from Hermes xvscf table.
        self.Lbd = Lambd
        if(calVCI):
            VCImtrx = self.VCImatrix(w_omega,linrComb,Evlst,nmode,maxorder,FCQ3,FCQ4,Vref,Lambd)
            Energylist, Coefficient = self.DiagonalVCI(VCImtrx)
        else:
            Energylist,Coefficient = self.readVCI(calVCI)
        self.thermoresults = np.zeros((len(Temprt),3,4))
        #XXX instruct: 7:7 temperatures 3: three methods 4: four variable(Xi,Omg,U,S)
        for ii in range(len(Temprt)):
            self.ThemoCalc(Temprt[ii],Energylist,self.thermoresults[ii,0,:])
            #self.Bose_EinsteinStat(Temprt[ii],w_omega,self.thermoresults[ii,1,:])
            #self.FiniteBE(Temprt[ii],w_omega,maxn,self.thermoresults[ii,2,:])


    def readVCI(self,calVCI):
        Energylist,Coefficient = np.load(calVCI)
        
    def loopfn(self,n,maxn):
        if n>1:
            rt = []
            for x in range(maxn):
                k = self.loopfn(n-1,maxn)
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
    def readSindoPES(self,filepath,nmode):
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

    def EvaluationList(self,nmode,w_omega,maxn,maxorder):
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


    def VCImatrix(self,w_omega,linrComb,Evlst,nmode,maxorder,FCQ3,FCQ4,Vref,Lambd):
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
                            multply *= FCQ3[ii,jj,kk]
                            sumofoperator+=Lambd*multply
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
                                sumofoperator+=Lambd*multply
                VCImtrx[i,j] = VCImtrx[j,i] = sumofoperator
        return VCImtrx

    def DiagonalVCI(self,VCImtrx):
        w,v = linalg.eigh(VCImtrx)
        HatreeTocm = 219474.63
        #print(w)
        #for i in range(len(w)):
        #    print("_+++++++++++++++++++++++++++++++++++++")
        #    print(w[i]*HatreeTocm)
        #    print("Then the Coeff")
        #    print(v[:,i])
        #print(w*HatreeTocm)
        print("Here::::::::::::::::::::::::::::::")
        print(np.sum(w))

        return w,v
    
    def ThemoCalc(self,Temprt,Energylist,ret):
        #kb = 1 at a.u.
        #Calculate Grand partition function, grand potential and internal energy based on grand canonical ensemble.
        #Grand partition function: GPF
        b_beta = 1/(Temprt)
        print(b_beta)
        GPF_Xi = 0
        print(np.sum(Energylist))
        for eachE in Energylist:
            try:
                math.exp(-b_beta*eachE)
            except OverflowError:
                print("temp is",Temprt)
                print("Lambd is", self.Lbd)
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
    def Bose_EinsteinStat(self,Temprt,w_omega,ret):
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
        #print("analytical Bose-Einstein stat result:")
        #print("Xi, Omg, U ,S is")
        #print(GPF_Xi)
        #print(GP_Omg) 
        #print(IE_U)
        #print(entropy_S) 
        #print("verify")
        #print(IE_U-Temprt*entropy_S)

    #FCI bose-einstein with finite N
    def FiniteBE(self,Temprt,w_omega,maxn,ret):
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
        #print("finite analytical Bose-Einstein stat result:")
        #print("Xi, Omg, U ,S is")
        #print(GPF_Xi)
        #print(GP_Omg) 
        #print(IE_U)
        #print(entropy_S) 
        #print("verify")
        #print(IE_U-Temprt*entropy_S)

def multitask(Temprt,Energylist,thermoresults):
    for ii in range(len(Temprt)):
        ThemoCalc(Temprt[ii],Energylist,thermoresults[ii,0,:])
        Bose_EinsteinStat(Temprt[ii],w_omega,thermoresults[ii,1,:])
        FiniteBE(Temprt[ii],w_omega,maxn,thermoresults[ii,2,:])
    #np.save("../data/thermoresult_Lambda0",thermoresults)

def vector1stderiv(p,p2,p3,n,n2,n3):
    return (-n3 + 9*n2 - 45*n + 45*p - 9*p2 + p3)/60
    
def vector2ndderiv(p,p2,p3,z,n,n2,n3):
    return (2*n3 - 27*n2 + 270*n - 490*z + 270*p - 27*p2 + 2*p3)/180


def Parallel_VCI(Temprt,maxn,Lambd):
    calVCI =1
    zero_l = VCIthermo(0,Temprt,maxn,calVCI)
    np.save("../data/TempgridLambd_0_"+str(maxn)+".npy",zero_l.thermoresults)
    #pos_l  = VCIthermo(Lambd,Temprt,maxn,calVCI)
    #pos_2l = VCIthermo(2*Lambd,Temprt,maxn,calVCI)
    #pos_3l = VCIthermo(3*Lambd,Temprt,maxn,calVCI)
    #neg_l  = VCIthermo(-Lambd,Temprt,maxn,calVCI)
    #neg_2l = VCIthermo(-2*Lambd,Temprt,maxn,calVCI)
    #neg_3l = VCIthermo(-3*Lambd,Temprt,maxn,calVCI)
    ##first-order:
    #vec1stderiv = np.vectorize(vector1stderiv)
    ##second-order:
    #vec2ndderiv = np.vectorize(vector2ndderiv)
    #deriv1stthermoresults = vec1stderiv(pos_l.thermoresults[:,0,:],pos_2l.thermoresults[:,0,:],pos_3l.thermoresults[:,0,:],neg_l.thermoresults[:,0,:],neg_2l.thermoresults[:,0,:],neg_3l.thermoresults[:,0,:])/Lambd
    #deriv2ndthermoresults = vec2ndderiv(pos_l.thermoresults[:,0,:],pos_2l.thermoresults[:,0,:],pos_3l.thermoresults[:,0,:],zero_l.thermoresults[:,0,:],neg_l.thermoresults[:,0,:],neg_2l.thermoresults[:,0,:],neg_3l.thermoresults[:,0,:])/(Lambd*Lambd)
    #np.save("../data/TempgridLambd_max_"+str(maxn)+"_1st.npy",deriv1stthermoresults)
    #np.save("../data/TempgridLambd_max_"+str(maxn)+"_2nd.npy",deriv2ndthermoresults)

    #filename = "../data/deriva1stResult_"+"Lambda"+str(idx)+".csv"
    #reslt = deriv1stthermoresults
    #with open(filename,'w') as csvfile:
    #    csvwriter = csv.writer(csvfile)
    #    for ii in range(len(Temprt)):
    #        csvwriter.writerow([reslt[ii,0],reslt[ii,1],reslt[ii,2],reslt[ii,3]])
    #filename = "../data/deriva2ndResult_"+"Lambda"+str(idx)+".csv"
    #reslt = deriv2ndthermoresults
    #with open(filename,'w') as csvfile:
    #    csvwriter = csv.writer(csvfile)
    #    for ii in range(7):
    #        csvwriter.writerow([reslt[ii,0],reslt[ii,1],reslt[ii,2],reslt[ii,3]])
#number of normal mode like H2O is 3 here I mannuly set up but later can read in from input file
nmode = 3
#maxium number of excited level for each mode.
maxnlist = [9,13,17,21] 
#maxium order of force field like QFF is 5 since we have kinetic operator at front
maxorder = 5
#by default Vref = 0
Vref= 0 
#Temperature unit is K then transfer to a.u. by 3.1577464*10^5 ( Eh/kb)
Ehbykb = 3.1577464*100000
Tlist = np.arange(2,8.1,0.1)
Temprt = np.zeros(np.shape(Tlist))
for i in range(np.shape(Tlist)[0]):
    Temprt[i] = 10**(Tlist[i])
#Temprt = np.array([100,1000,10000,100000,1000000,10000000,100000000])
Temprt = Temprt/Ehbykb


#linrComb = loopfn(nmode,maxn)
#filepath = "../data/prop_no_1.mop"
#Lambdlist = [0.2,0.1,0.01,0.001,0.0001]
Lambd = 0.0001
#w_omega,FCQ3,FCQ4 = readSindoPES(filepath,nmode)
#Evlst = EvaluationList(nmode,w_omega,maxn,maxorder)# The list of the evaluation from Hermes xvscf table.
#VCImtrx = VCImatrix(linrComb,Evlst,nmode,maxorder,FCQ3,FCQ4,Vref,Lambd)
#Energylist, Coefficient = DiagonalVCI(VCImtrx)
calVCI = 1
#XXX instruct: 7:7 temperatures 3: three methods 4: four variable(Xi,Omg,U,S)

#zero_l = VCIthermo(0,Temprt,maxn,calVCI)
#filename = "../data/deriva0Result_"+".csv"
#reslt = zero_l.thermoresults
#with open(filename,'w') as csvfile:
#    csvwriter = csv.writer(csvfile)
#    for ii in range(7):
#        csvwriter.writerow([reslt[ii,0,0],reslt[ii,0,1],reslt[ii,0,2],reslt[ii,0,3]])
#XXX Lambda list parallel
procs = []
for ii in range(len(maxnlist)):
    proc = Process(target = Parallel_VCI, args= (Temprt,maxnlist[ii],Lambd))
    procs.append(proc)
    proc.start()
for procc in procs:
    procc.join()
#np.save("../data/Energylistwithmaxn"+str(maxn),Energylist)
#Energylist = np.load("../data/Energylistwithmaxn"+str(maxn)+".npy")
#Parallel_VCI(Temprt,maxn,0.0001,0)

#for ii in range(len(Temprt)):
#ThemoCalc(Temprt[0],Energylist,thermoresults[0,0,:])
#    Bose_EinsteinStat(Temprt[ii],w_omega,thermoresults[ii,1,:])
#    FiniteBE(Temprt[ii],w_omega,maxn,thermoresults[ii,2,:])

#multitask(Temprt,Energylist,thermoresults)

t1 = time.time()
print("time is /min",(t1-t0)/60)
#sys.stdout.close()

