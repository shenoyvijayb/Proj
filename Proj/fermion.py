from __future__ import print_function

import copy
import numpy as np
from scipy.special import comb
from numpy import linalg as la
import json
from pprint import pprint
import sys
import os
import os.path


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#--------------------------------------------------------------------
# function to convert state number into baseN string
def number_to_baseN_string(i, N, nsLen, baseN):
    num = i
    for j in range(0,nsLen):
        quo = num//N # note // stands for integer division
        rem = num - N*quo
        baseN[j] = rem
        num=quo

#--------------------------------------------------------------------
# function of covert configuration to global state number
def baseN_string_to_number(N, baseN):
    ans = 0
    i = 0 
    for j in baseN:
        ans += j*(N**i)
        i+=1
    return ans

#--------------------------------------------------------------------
# function to count number of fermions in a configuration
def number_of_fermions(fconfig):
    ans=0
    for i in fconfig:
        ans += i
    return ans

#--------------------------------------------------------------------
# sign of fermionic operator
def sign_of_c(fconfig, norb):
    ans=1
    for i in range(0,norb):
        if fconfig[i] == 1 :
            ans*=-1
    return np.double(ans)

#--------------------------------------------------------------------
# fermion state class
class FermionState:

    fermiN = 2
    
    #-----constructor
    def __init__(self, nFOrbitals, statenum):
        self.isZero = False
        self.nFOrbitals = nFOrbitals
        self.Fconfig = np.zeros(nFOrbitals,dtype=np.int)
        self.phase = 1 + 0j
        number_to_baseN_string(statenum, self.fermiN, self.nFOrbitals,
                               self.Fconfig)
    
    #-----return statenumber
    def statenum(self):
        return baseN_string_to_number(self.fermiN,self.Fconfig)

    #-----return number of fermions
    def num_fermions(self):
        return number_of_fermions(self.Fconfig)

    #-----CdagC_Op-----This is the number operator
    def CdagC_Op(self,norb):
        return self.Fconfig[norb]

    #-----Cdagger------ updn=1 is c^\dagger and updn=-1 is c
    def Cdagger_Op(self, norb, updn):
        if self.isZero:
            return
        else :
            old_occ = self.Fconfig[norb]
            
            if old_occ == ((1+updn)//2) :
                self.isZero = True
                return
            else :
                new_occ = (1-old_occ)
                sign = sign_of_c(self.Fconfig,norb)
                self.phase *= sign+0j
                self.Fconfig[norb] = new_occ
    
    #------printing    
    def print(self):
        print("Fermion state")
        if self.isZero :
            print(" Zero state ")
        else :
            print(" number of orbitals = ", self.nFOrbitals)
            print(" phase = ", self.phase)
            print(self.Fconfig)
        return
            
#--------------------------------------------------------------------
class FockSpaceState:    
    #----
    # Mf is number of fermion flavors

    fermiN = 2

    def __init__(self, Mf, nFOrbitals, fstatenum) :
        self.isZero = False
        self.phase = 1+0j
        self.Mf = Mf
        self.nForbitals = nFOrbitals
        self.nstates_per_flavor = (self.fermiN)**nFOrbitals
        self.FermState = []
        for iflav in range(0,Mf):
            self.FermState.append(FermionState(nFOrbitals,
                                               fstatenum[iflav]))
    #------------
    # CdagC
    def CdagC_Op(self, norb, nflavor):
        return self.FermState[nflavor].Fconfig[norb]
    
    #------------
    # Prod_CdagC
    def Prod_CdagC_Op(self, norb):
        ans = 1
        for iflv in range(0,self.Mf):
            ans *= self.CdagC_Op(norb, iflv)
        return ans

    #------------
    def Cdagger_Op(self, norb, nflavor, updn):
        if self.isZero :
            return
        else :
            old_phase = self.FermState[nflavor].phase
            self.FermState[nflavor].Cdagger_Op(norb,updn)
            if self.FermState[nflavor].isZero :
                self.isZero = True
                return
            new_phase = self.FermState[nflavor].phase

            ncrossed = 0
            for iflv in range(0,nflavor):
                ncrossed += self.FermState[iflv].num_fermions()
            cphase = 1+0j
            if ncrossed % 2 == 0:
                cphase = 1+0j
            else :
                cphase = -1+0j
            self.phase *= ((cphase*new_phase)/old_phase)

    #--------------
    def fock_state_number(self):
        fstatenum = np.zeros(self.Mf, dtype=np.int)
        for iflv in range(0,self.Mf):
            fstatenum[iflv] = self.FermState[iflv].statenum()
        return  baseN_string_to_number(self.nstates_per_flavor, fstatenum)
        
    #------------
    def print(self):
        print("FockSpaceState---")
        if self.isZero :
            print("Zero Fock Space State")
        else :
            print("Mf = ", self.Mf)
            print("phase = ", self.phase)
            print("norbitals = ", self.nForbitals)
            for iflav in range(0,Mf):
                print(" Favlour ", iflav)
                self.FermState[iflav].print()
        return
    
#--------------------------------------------------------------------
class EOper : #elementary operator (action on a basis state gives another basis state)

    #-----------------------------------
    def __init__(self, which_dof_, flavor_, dof_num_, action_, dagger_, tag_):
        self.which_dof = which_dof_
        self.flavor = flavor_
        self.dof_num = dof_num_
        self.action = action_
        self.dagger = dagger_
        self.tag = tag_
                
    #-----------------------------------
    def print(self) :
        print(" EOper :")
        print("    which_dof : ", self.which_dof)
        print("    flavor    : ", self.flavor)
        print("    dof_num   : ", self.dof_num)
        print("    action    : ", self.action)
        print("    dagger    : ", self.dagger)
        print("    tag       : ", self.tag)
        

#--------------------------------------------------------------------
class FockSpaceOperator:

    #-----------------------------------
    def __init__(self, Mf_, coeff_, hermitian_, tag_):
        self.Mf    = Mf_
        self.coeff = coeff_
        self.hermitian = hermitian_ #hermitain_ true means we have to
                                    #add hermitian conjugate in all calculations    
        self.tag = tag_
        self.oper_list = []

    #-----------------------------------
    def act_on(self, fsstate):
        factor = 1+0j

        if fsstate.isZero :
            return factor

        for io,oper in enumerate(self.oper_list):
            if oper.which_dof == 'F':
                if oper.action == 'C' :
                    fsstate.Cdagger_Op(oper.dof_num, oper.flavor, oper.dagger)
                elif oper.action == 'N' :
                    num = fsstate.CdagC_Op(oper.dof_num, oper.flavor)
                    factor*=num
                elif oper.action == 'D' :
                    docc = fsstate.Prod_CdagC_Op(oper.dof_num)
                    factor*=docc
                else :
                   print("What F-action is this? : ", oper.action)
                   oper.print()
                   exit() 
            else :
                print("What DOF is this? : ", oper.which_dof)
                oper.print()
                exit()
        factor*=(fsstate.phase)
        fsstate.phase = 1+0j
        return factor

    #-----------------------------------        
    def print(self) :
        print(" FockSpaceOperator ")
        print("    Mf : ", self.Mf)
        print("    coeff : ", self.coeff)
        print("    hermitian : ", self.hermitian)
        print("    tag : ", self.tag)
        print("    number of elementary operators ", len(self.oper_list))
        for ie,eoper in enumerate(self.oper_list):
            print(" operator number ", ie)
            eoper.print()
            

#--------------------------------------------------------------------
def ExpectationValue(fsoperarray_, fockspace_, mat_idx_, evec_):
    ans = np.cdouble(0.e0+0.e0j)
    for fsoper in fsoperarray_:
        for isnum in fockspace_:
            ifsstate = copy.deepcopy(fockspace_[isnum])
            i = mat_idx_[isnum]
            factor = fsoper.act_on(ifsstate)
            if not ifsstate.isZero:
                jsnum = ifsstate.fock_state_number()
                j=mat_idx_[jsnum]
                mat_elem = factor*fsoper.coeff*np.conj(evec_[j])*evec_[i]
                if fsoper.hermitian :
                    mat_elem += np.conj(mat_elem)
                ans += mat_elem
            del ifsstate
    return ans
    
#--------------------------------------------------------------------
# some standard opearators
#
DAGGER=1
NO_DAGGER=-1
ADD_HERMITIAN=True
DONT_ADD_HERMITIAN=False
FERMION='F'
def DoubleOcc(Mf,U,isite,tag) :
    D0 = FockSpaceOperator(Mf,U,DONT_ADD_HERMITIAN,tag)
    d0oper = EOper(FERMION,Mf,isite,'D',DAGGER,tag)
    D0.oper_list.append(d0oper)
    return D0

def Number(Mf,mu,isite,iflv,tag):
    oer = FockSpaceOperator(Mf,mu,DONT_ADD_HERMITIAN,tag)
    eroper = EOper(FERMION,iflv,isite,'N',DAGGER,tag)
    oer.oper_list.append(eroper)
    return oer

def Hop(Mf,t,isite,jsite,iflv,tag):
    hop = FockSpaceOperator(Mf,t,ADD_HERMITIAN,tag)
    destroy_i = EOper(FERMION,iflv,isite,'C',NO_DAGGER,tag)
    create_j = EOper(FERMION,iflv,jsite,'C',DAGGER,tag)
    hop.oper_list.append(destroy_i)
    hop.oper_list.append(create_j)
    return hop

#-------------this works only for Mf=2-------------------------
def SdotS(Mf,Jex,isite,jsite,tag):
    if not Mf==2:
        print("SdoS defined only for Mf=2, not for MF =", Mf)
        exit()
    UP=0
    DN=1
    sdots = []

    niupnjup = FockSpaceOperator(Mf,Jex/4.e0,DONT_ADD_HERMITIAN,tag)
    niupnjup.oper_list.append(EOper(FERMION,UP,isite,'N',DAGGER,tag))
    niupnjup.oper_list.append(EOper(FERMION,UP,jsite,'N',DAGGER,tag))
    sdots.append(niupnjup)

    niupnjdn = FockSpaceOperator(Mf,-Jex/4.e0,DONT_ADD_HERMITIAN,tag)
    niupnjdn.oper_list.append(EOper(FERMION,UP,isite,'N',DAGGER,tag))
    niupnjdn.oper_list.append(EOper(FERMION,DN,jsite,'N',DAGGER,tag))
    sdots.append(niupnjdn)

    nidnnjup = FockSpaceOperator(Mf,-Jex/4.e0,DONT_ADD_HERMITIAN,tag)
    nidnnjup.oper_list.append(EOper(FERMION,DN,isite,'N',DAGGER,tag))
    nidnnjup.oper_list.append(EOper(FERMION,UP,jsite,'N',DAGGER,tag))
    sdots.append(nidnnjup)
    
    nidnnjdn = FockSpaceOperator(Mf,Jex/4.e0,DONT_ADD_HERMITIAN,tag)
    nidnnjdn.oper_list.append(EOper(FERMION,DN,isite,'N',DAGGER,tag))
    nidnnjdn.oper_list.append(EOper(FERMION,DN,jsite,'N',DAGGER,tag))
    sdots.append(nidnnjdn)

    sipsjm = FockSpaceOperator(Mf,Jex/2.e0,ADD_HERMITIAN,tag)
    sipsjm.oper_list.append(EOper(FERMION,UP,jsite,'C',NO_DAGGER,tag))
    sipsjm.oper_list.append(EOper(FERMION,DN,jsite,'C',DAGGER,tag))
    sipsjm.oper_list.append(EOper(FERMION,DN,isite,'C',NO_DAGGER,tag))
    sipsjm.oper_list.append(EOper(FERMION,UP,isite,'C',DAGGER,tag))
    sdots.append(sipsjm)

    return sdots
    




    

