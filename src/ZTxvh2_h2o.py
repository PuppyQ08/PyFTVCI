"""
This is the XVH2 for H2O to make sure my g09FC program is right.
TODO: Finish it as soon as possible
"""
import sys
import numpy as np
import math
import time
from itertools import permutations
#TODO: First to get Fijk and Fijkl form the cartesian coordinates FC in txt file
#Fijk = sum(Cai Cbj Cck) Fabc Cai is a th element in ith model
class ZTxvh2:
    def __init__(self,nmode,Temprt):
        #filepath = "../data/prop_no_1.mop"
        filepath = "../data/prop_no_3.hs"
        self.nmode = nmode
        w_omega,FCQ3,FCQ4 = self.readSindoPES(filepath,self.nmode)
        const = 4.359743E-18/(1.660538E-27 * 0.5292E-10 * 0.5292E-10/meconstant)#hatree/(amu*bohr*bohr) it transfer to SI unit
        Hatreeconst = 219474.6
        #v1 = self.findv(0,w_omega,const,FCQ3,FCQ4) 
        #v2 = self.findv(1,w_omega,const,FCQ3,FCQ4) 
        #v3 = self.findv(2,w_omega,const,FCQ3,FCQ4) 
        v1 = self.func(w_omega,0,w_omega[0],FCQ3,FCQ4)
        v2 = self.func(w_omega,1,w_omega[1],FCQ3,FCQ4)
        v3 = self.func(w_omega,2,w_omega[2],FCQ3,FCQ4)
        print(v1)
        print(v2)
        print(v3)
        #print("sec order 0")
        #print((w_omega[0] - v1))
        #print((w_omega[0] - v1)*Hatreeconst)
        #print("sec order 1")
        #print((w_omega[1] - v2))
        #print((w_omega[1] - v2)*Hatreeconst)
        #print("sec order 2")
        #print((w_omega[2] - v3))
        #print((w_omega[2] - v3)*Hatreeconst)
        
        #for eachtemp in Temprt:
            #b = self.BEfunc(w_omega[0],eachtemp)
        #eachtemp =1
        #a= self.totalenergy(w_omega,FCQ3,FCQ4,eachtemp)
        #b= np.sum(w_omega)/2
        ##print("HHHHHHHHHHHHHHHHHHHHHHERE harmonic")
        ##print(b)
        ##print(b*Hatreeconst)
        #print("______________ perturbation")
        #print(a)
        #print(a*Hatreeconst)
        #print("total")
        #print((a+b)*Hatreeconst)

    def readSindoPES(self,filepath,nmode):
        w_omega = np.zeros(nmode)
        FCQ3 = np.zeros((nmode,nmode,nmode)) #Coefficient in Q (normal coordinates)
        #XXX Coefficient includes the 1/2 1/3! 1/4! in the front!!
        #Dr.Yagi used dimensionless q as unit so we need to transfer from q to Q by times sqrt(w1*w2*.../hbar^(...))
        FCQ4 = np.zeros((nmode,nmode,nmode,nmode))
        with open(filepath) as f:
            flines = f.readlines()
            for idx in range(len(flines)):
                if( len(flines[idx].split())>1):

                    if (flines[idx].split()[1] == "Hessian(i,i)"):
                        tl = flines[idx+1].split()#shortcut for this line
                        leng=  len(tl)
                        if (leng == 2):
                            for i in range(nmode):
                                tl2 = flines[idx+1+i].split()
                                w_omega[i] = math.sqrt(float(tl2[1]))
                                #print("Hessian",math.sqrt(float(tl2[1])/(1.88973**2*math.sqrt(1822.888486**2)))*219474.63)
                    if (flines[idx].split()[1] == "Cubic(i,i,i)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            FCQ3[int(tl[0])-1,int(tl[0])-1,int(tl[0])-1] = float(tl[1])
                            #print("Cubic3",tl[1])
                    if (flines[idx].split()[1] == "Cubic(i,i,j)"):
                        for i in range(nmode*2):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[1])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ3[i[0],i[1],i[2]] = float(tl[2])
                            #print("Cubic2",tl[2])
                    if (flines[idx].split()[1] == "Cubic(i,j,k)"):
                        tl = flines[idx+1].split()#shortcut for this line
                        listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[2])-1]
                        perm = permutations(listidx)
                        for i in list(perm):
                            FCQ3[i[0],i[1],i[2]] = float(tl[3])
                        #print("Cubic1",tl[3])

                    if (flines[idx].split()[1] == "Quartic(i,i,i,i)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            FCQ4[int(tl[0])-1,int(tl[0])-1,int(tl[0])-1,int(tl[0])-1] = float(tl[1])
                            #print("Quar4",tl[1])
                    if (flines[idx].split()[1] == "Quartic(i,i,j,j)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[1])-1,int(tl[1])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ4[i[0],i[1],i[2],i[3]] = float(tl[2])
                            #print("Quar22",tl[2])
                    if (flines[idx].split()[1] == "Quartic(i,i,i,j)"):
                        for i in range(nmode*2):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[0])-1,int(tl[1])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ4[i[0],i[1],i[2],i[3]] = float(tl[2])
                            #print("Quar21",tl[2])
                    if (flines[idx].split()[1] == "Quartic(i,i,j,k)"):
                        for i in range(nmode):
                            tl = flines[idx+1+i].split()#shortcut for this line
                            listidx = [int(tl[0])-1,int(tl[0])-1,int(tl[1])-1,int(tl[2])-1]
                            perm = permutations(listidx)
                            for i in list(perm):
                                FCQ4[i[0],i[1],i[2],i[3]] = float(tl[3])
                            #print("Quar3",tl[3])
                        

        FCQ3 = np.true_divide(FCQ3,(1.88973**3*math.sqrt(1822.888486**3)))    
        FCQ4 = np.true_divide(FCQ4,(1.88973**4*math.sqrt(1822.888486**4)))    
        w_omega = np.true_divide(w_omega,math.sqrt(1.88973**2*1822.888486))    
        #with open(filepath) as f:
        #    flines = f.readlines()
        #    FCstartidx = nmode#if the scanning and fetching index failed , it will be over the size of matrix. 
        #    Omgstartidx = nmode 
        #    for idx in range(len(flines)):
        #        if (flines[idx].split()[0] == "DALTON_FOR_MIDAS"):
        #            FCstartidx = idx
        #        if (flines[idx].split()[0] == "SCALING"):
        #            Omgstartidx = idx
        #    widx = 0
        #    for idx in range(Omgstartidx+1,FCstartidx):
        #        tl = flines[idx].split()#shortcut for this line
        #        w_omega[widx] = float(tl[0])
        #        widx += 1
        #    for idx in range(FCstartidx+1, len(flines)):
        #        tl = flines[idx].split()#shortcut for this line
        #        leng = len(tl)
        #        if (leng == 4):
        #            #third order force constant
        #            temp1 = float(tl[0])*math.sqrt(w_omega[int(tl[1])-1]*w_omega[int(tl[2])-1]*w_omega[int(tl[3])-1])/6
        #            FCQ3[int(tl[1])-1,int(tl[2])-1,int(tl[3])-1] = temp1
        #            FCQ3[int(tl[1])-1,int(tl[3])-1,int(tl[2])-1] = temp1
        #            FCQ3[int(tl[2])-1,int(tl[1])-1,int(tl[3])-1] = temp1
        #            FCQ3[int(tl[2])-1,int(tl[3])-1,int(tl[1])-1] = temp1
        #            FCQ3[int(tl[3])-1,int(tl[1])-1,int(tl[2])-1] = temp1
        #            FCQ3[int(tl[3])-1,int(tl[2])-1,int(tl[1])-1] = temp1
        #        if (leng == 5):
        #            #forth order force constant
        #            temp2 = float(tl[0])*math.sqrt(w_omega[int(tl[1])-1]*w_omega[int(tl[2])-1]*w_omega[int(tl[3])-1]*w_omega[int(tl[4])-1])/24
        #            perm = permutations([1,2,3,4])
        #            for i in list(perm):
        #                FCQ4[int(tl[i[0]])-1,int(tl[i[1]])-1,int(tl[i[2]])-1,int(tl[i[3]])-1] = temp2   
        return w_omega,FCQ3,FCQ4

    def getidx(self,i,j):
        output = [i,j]
        output.sort()
        return int(output[1])*(int(output[1])+1)/2 + int(output[0]) 

    def func(self,w_omega,m,v,FCQ3,FCQ4):
        a1m = 0.0
        abm = 0.0
        cdm = 0.0
        em = 0.0
        fm = 0.0
        gm = 0.0
        hm = 0.0
        mm = 0.0
        nm = 0.0
        w_omg = w_omega
        for i in range(self.nmode):
            a1m += FCQ4[m,m,i,i]/(8*w_omg[m]*w_omg[i])
            for j in range(self.nmode):
                abm -= FCQ3[m,m,i]*FCQ3[i,j,j]/(8*w_omg[i]**2*w_omg[j]*w_omg[m])
                em += FCQ3[m,i,j]**2/(16*w_omg[i]*w_omg[j]*w_omg[m]*(v-w_omg[i]-w_omg[j]))
                fm -= FCQ3[m,i,j]**2/(16*w_omg[i]*w_omg[j]*w_omg[m]*(v + w_omg[i] + w_omg[j]))
                for k in range(self.nmode):
                    cdm -= FCQ4[m,m,i,j]*FCQ4[i,j,k,k]/(32*w_omg[i]*w_omg[j]*w_omg[m]*w_omg[k]*(w_omg[i]+w_omg[j]))
                    gm += FCQ4[m,i,j,k]**2/(6*16* w_omg[i]* w_omg[j]* w_omg[k]* w_omg[m]* (v-w_omg[i]-w_omg[j]-w_omg[k]))
                    hm += FCQ4[m,i,j,k]**2/(6*16* w_omg[i]* w_omg[j]* w_omg[k]* w_omg[m]* (- v-w_omg[i]-w_omg[j]-w_omg[k]))
                    if (i!=m):
                        mm += FCQ4[m,i,j,j]*FCQ4[m,i,k,k]/(4*16* w_omg[i]* w_omg[j]* w_omg[k]* w_omg[m]* (v-w_omg[i]))
                        nm -= FCQ4[m,i,j,j]*FCQ4[m,i,k,k]/(4*16* w_omg[i]* w_omg[j]* w_omg[k]* w_omg[m]* (v+w_omg[i]))

        sigmam =  abm+cdm+em+fm+gm+hm+mm+nm
        return sigmam#math.sqrt(w_omg[m]**2 + 2*w_omg[m] * sigmam) #w_omg[m]**2 + 2*w_omg[m] * sigmam - v**2

    def BEfunc(self,womg,Temprt):

        return 1
        #return 1/(math.exp(womg/Temprt)-1)
        
            
    #It is just a guess for Omega(2) 
    def totalenergy(self, w_omega,FCQ3,FCQ4,Temprt):
        a2c2=0
        b2d2=0
        a1 = 0
        for i in range(3):
            for j in range(3):
                a1 += FCQ4[i,i,j,j]*self.BEfunc(w_omega[i],Temprt)*self.BEfunc(w_omega[j],Temprt)/(32*w_omega[i]*w_omega[j])
                for k in range(3):
                    a2c2 -= FCQ3[i,j,j]*FCQ3[i,k,k]*self.BEfunc(w_omega[i],Temprt)*self.BEfunc(w_omega[j],Temprt)*self.BEfunc(w_omega[k],Temprt)/(32*w_omega[i]**2*w_omega[j]*w_omega[k]) + FCQ3[i,j,k]**2*self.BEfunc(w_omega[i],Temprt)*self.BEfunc(w_omega[j],Temprt)*self.BEfunc(w_omega[k],Temprt)/(48*w_omega[i]*w_omega[j]*w_omega[k]*(w_omega[i]+w_omega[j]+w_omega[k]))
                    for l in range(3):
                        b2d2 -= FCQ4[i,j,k,k]*FCQ4[i,j,l,l]*self.BEfunc(w_omega[i],Temprt)*self.BEfunc(w_omega[j],Temprt)*self.BEfunc(w_omega[k],Temprt)*self.BEfunc(w_omega[l],Temprt) + FCQ4[i,j,k,l]**2*self.BEfunc(w_omega[i],Temprt)*self.BEfunc(w_omega[j],Temprt)*self.BEfunc(w_omega[k],Temprt)*self.BEfunc(w_omega[l],Temprt)
        sec = a2c2+b2d2
        Hatreeconst = 219474.6
        print("first_______________")
        print(a1)
        print(a1*Hatreeconst)
        print("secondt_______________")
        print(sec)
        print(sec*Hatreeconst)
        return sec+a1



        

    def singulars(self,w_omega):
        w_omg = w_omega
        singul = [0.0]
        for i in range(3):
            for j in range(3):
                if not(w_omg[i]+w_omg[j] in singul):
                    singul.append(w_omg[i]+w_omg[j])
        singul.sort()
        return singul

    def findv(self,m,w_omega,const,FCQ3,FCQ4):
        #get the singularities first
        singul = self.singulars(w_omega)
        #print("interv",singul)
        #print(func(m,0.007))
        for i in range(1,len(singul)):
            right = singul[i] - 0.00005 
            left = singul[i-1] + 1E-4
            leftv = self.func(w_omega,m,left,FCQ3,FCQ4)
            rightv = self.func(w_omega,m,right,FCQ3,FCQ4)
            #print(":",left,right,leftv,rightv)
            if np.sign(leftv)!=np.sign(rightv):
                while abs(left - right) > 1E-10:
                    mid = (left + right)/2
                    midv = self.func(w_omega,m,mid,FCQ3,FCQ4)
                    if midv == 0.0:
                        solu = mid 
                        print("solution found!")
                        break
                    else:
                        if np.sign(leftv) == np.sign(midv):
                            left = mid
                            leftv = midv
                        elif np.sign(rightv) == np.sign(midv):
                            right = mid
                            rightv = midv
                print ("solu",mid*math.sqrt(const)/(2.99792458E10 * 2 * math.pi))
                print(mid*219474.631)
                #if (mid*219474.631 <4100):
                #    ret = mid
                
        return ret 
#==========================================TEST PART ===========================#



#inputCoefOmg()
meconstant = 1822.888486
const = 4.359743E-18/(1.660538E-27 * 0.5292E-10 * 0.5292E-10/meconstant)#hatree/(amu*bohr*bohr) it transfer to SI unit
Temprt = np.array([100,1000,10000,100000,1000000,10000000,100000000])
Ehbykb = 3.1577464*100000
Temprt = Temprt/Ehbykb
ZTX = ZTxvh2(3,Temprt)
#freq = []
#for idx in range(len(w_omg)):
    #freq.append(w_omg[idx]*math.sqrt(const)/(2.99792458E10 * 2 * math.pi))#from angular freq to wave number: nu = omg/(2pic) omg is angular frequency
#print(w_omg)
#print(freq)
#print(findv(1)*math.sqrt(const)/(2.99792458E10 * 2 * math.pi))
#print(findv(2)*math.sqrt(const)/(2.99792458E10 * 2 * math.pi))
#print(findv(3)*math.sqrt(const)/(2.99792458E10 * 2 * math.pi))
