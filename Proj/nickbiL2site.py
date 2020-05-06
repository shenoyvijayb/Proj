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

import fermion


#--------------------------------------------------------------------
# main code for 4 sites bilayer nickelate
casename="test"
U=10.e0
t=1.e0
tp=0.1e0
tR=1.e0
eR=2.e0
nparticles = [1,1]
num_outp_states =2

#------------------------read input if it is there
if(len(sys.argv) > 1) :
    runpfile = sys.argv[1]
    with open(runpfile) as runp_file:
        runpdata = json.load(runp_file)
        pprint(runpdata)
        for key in runpdata:
            if (key == 'casename') :
                casename = runpdata['casename']
            elif (key == 'U' ):
                U = runpdata['U']
            elif (key == 't' ):
                t = runpdata['t']
            elif (key == 'tp'):
                tp = runpdata['tp']
            elif (key == 'tR'):
                tR = runpdata['tR']
            elif (key == 'eR'):
                eR = runpdata['eR']
            elif (key == 'nparticles') :
                nparticles = runpdata['nparticles']
            elif (key == 'num_outp_states') :
                num_outp_states = runpdata['num_outp_states']

#----------------------echo input
print(" ")
print("****BILAYER NICKELATE****")
print("  casename    : ", casename)
print("         t    = ", t)
print("         tp   = ", tp)
print("         tR   = ", tR)
print("         eR   = ", eR)
print("         U    = ", U)
print(" nparticles   = ", nparticles)
print("output states = ", num_outp_states)

print(" ")
  

fermiN=2             #number of states per site for fermions
Mf=2
nsites = 4
n_all_states = fermiN**nsites
n_fock_space_states = n_all_states**Mf



#-------------set up fock space ------------------------------------
fstatenum = np.zeros(Mf,dtype=np.int)
fockspace = {}
for ifstate in range(0,n_fock_space_states):
    fermion.number_to_baseN_string(ifstate, n_all_states, Mf, fstatenum)
    fsstate = fermion.FockSpaceState(Mf, nsites, fstatenum)
    fnum = []
    for i in range(0,Mf):
        fnum.append(fsstate.FermState[i].num_fermions())
    if fnum == nparticles :
        fockspace[ifstate] = fsstate
    else :
        del fsstate
print("Dimension of Fock Space : ", len(fockspace))


n_dim_fs = len(fockspace)
statelist = list(fockspace.keys())
print("statelist", statelist)
mat_idx = {}
for i in range(0,len(statelist)):
    mat_idx[statelist[i]] = i
print("mat_idx", mat_idx)


#-------------setup hamiltonian operator----------------------------
ham = []
#set hubbard U operator
for isite in range(0,2):
    ham.append(fermion.DoubleOcc(Mf,U,isite,'D'))

#set eR operator
for isite in range(2,4):
    for iflv in range(0,Mf):
        ham.append(fermion.Number(Mf,eR,isite,iflv,'R'))

#set kinetic energy operator
thop_ar = [t,tp,tR,tp]
for isite in range(0,4):
    jsite = (isite+1)%4
    for iflv in range(0,Mf):
        ham.append(fermion.Hop(Mf,-thop_ar[isite], isite, jsite, iflv, 'H'))

#-------------set up hamiltonian matrix-----------------------------

hmat = np.zeros((n_dim_fs,n_dim_fs), np.cdouble)
    
for fsoper in ham:
    for isnum in fockspace:
        ifsstate = copy.deepcopy(fockspace[isnum])
        i = mat_idx[isnum]
        factor = fsoper.act_on(ifsstate)
        if  not ifsstate.isZero :
            jsnum = ifsstate.fock_state_number()
            j = mat_idx[jsnum]
            mat_elem = factor*fsoper.coeff   
            hmat[j,i] += mat_elem
            if fsoper.hermitian :
                hmat[i,j] += np.conj(mat_elem)
        del ifsstate
                
eig,evs = la.eigh(hmat)


#---------------output expectation values---------------------------

#gs = np.zeros(n_dim_fs,np.cdouble)

#print(eig)

for iopstate in range(0,num_outp_states):
    gs = evs[:,iopstate]
    ##print(gs)

    keystr = ""
    outpstr = ""
    keystr += (" t " + " tp " + " tR " + " eR " + " U ")
    outpstr += (" %14.6E"%(t) + " %14.6E"%(tp) + " %14.6E"%(tR) + " %14.6E"%(eR)  + " %14.6E"%(U)  )

    keystr += " E "
    outpstr += (" %25.15E"%(eig[iopstate]))

    keystr += " Eex "
    outpstr += (" %25.15E"%(eig[iopstate]-eig[0]))
    
    mean_num =  np.zeros(nsites,dtype=np.double)
    mean_dble_occ = np.zeros(nsites,dtype=np.double)
    mean_sdots = np.zeros((nsites, nsites),dtype=np.double)

    for isite in range(0,nsites):
        tmp_mean_num = 0.e0
        for iflv in range(0,Mf):
            num = [fermion.Number(Mf,1+0j,isite,iflv,'N')]
            tmp_mean_num += np.real(fermion.ExpectationValue(num, fockspace, mat_idx, gs))
        mean_num[isite] = tmp_mean_num
            
        dble_occ = [fermion.DoubleOcc(Mf,1+0j,isite,'D')]
        mean_dble_occ[isite] =  np.real(fermion.ExpectationValue(dble_occ,
                                                         fockspace, mat_idx, gs))
        for jsite in range(isite,nsites) :
            sdots_oper = fermion.SdotS(Mf,1+0j,isite,jsite,'S')
            tmp_mean_sdots = np.real(fermion.ExpectationValue(sdots_oper,
                                                               fockspace,
                                                               mat_idx, gs))
            mean_sdots[isite,jsite] = tmp_mean_sdots
            if jsite > isite :
                mean_sdots[jsite,isite] = tmp_mean_sdots
            
            
    for isite in range(0,nsites):
        keystr += (" N"+"%0d"%(isite))
        outpstr += " %25.15E"%(mean_num[isite])


    for isite in range(0,nsites):
        keystr += (" D"+"%0d"%(isite))
        outpstr += " %25.15E"%(mean_dble_occ[isite])

    for isite in range(0,nsites):
        for jsite in range(isite,nsites):
            keystr += (" SdS"+"%0d"%(isite)+"%0d"%(jsite))
            outpstr += " %25.15E"%(mean_sdots[isite,jsite])
    
    keystr += (" Stot2"+ " Stot")
    stot2 = np.sum(mean_sdots)
    stot = (-1.e0 + np.sqrt(1.e0 + 4.e0*stot2))/2.e0
    outpstr += (" %25.15E"%(stot2) + " %25.15E"%(stot))
            
            
    # print("----------------------------------------------")
    # print(" Eigenstate ", iopstate)
    # print("double occ")
    # print(mean_dble_occ)
    # print("sdots")
    # print(mean_sdots)
    # print("s_tot = ", np.sum(mean_sdots))

    # print(keystr)
    # print(outpstr)

    filestub = casename + "_out"+"%02d"%(iopstate)
    keyfp = open(filestub+".key",'w')
    print(keystr,file=keyfp)
    keyfp.close()
    outfp = open(filestub+".plt",'w')
    print(outpstr,file=outfp)
    outfp.close()

exit()
    


print(ExpectationValue(ham, fockspace, mat_idx, gs))

sdots00 = SdotS(Mf,1+0j,0,0,'S')
sdots01 = SdotS(Mf,1+0j,0,1,'S')
print("gs")
mean_sdots00 = np.real(ExpectationValue(sdots00, fockspace, mat_idx, gs))
mean_sdots01 = np.real(ExpectationValue(sdots01, fockspace, mat_idx, gs))
print(mean_sdots00, mean_sdots01, 2.e0*(mean_sdots00 + mean_sdots01))


print("ex 1")
gs = evs[:,1]
mean_sdots00 = np.real(ExpectationValue(sdots00, fockspace, mat_idx, gs))
mean_sdots01 = np.real(ExpectationValue(sdots01, fockspace, mat_idx, gs))
print(mean_sdots00, mean_sdots01, 2.e0*(mean_sdots00 + mean_sdots01))

