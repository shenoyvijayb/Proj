import copy
import numpy as np
from scipy.special import comb
from numpy import linalg as la

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
        number_to_baseN_string(statenum, fermiN, self.nFOrbitals,
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
        self.nstates_per_flavor = fermiN**nFOrbitals
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
                j=mat_idx[jsnum]
                mat_elem = factor*fsoper.coeff*np.conj(evec_[j])*evec_[i]
                if fsoper.hermitian :
                    mat_elem += np.conj(mat_elem)
                ans += mat_elem
            del ifsstate
    return ans
    
            
#--------------------------------------------------------------------
# main code 
fermiN=2             #number of states per site for fermions
Mf=2
nsites = 4
nparticles = [1,1]
n_all_states = fermiN**nsites
n_fock_space_states = n_all_states**Mf
#nstates = comb(nsites,nparticles,True)
#print(nstates)
print(n_fock_space_states)

#exit()

for istate in range(0,n_all_states):
    fconfig = np.zeros(nsites,dtype=np.int)
    print(istate)
    number_to_baseN_string(istate,fermiN,nsites,fconfig)
    print(fconfig, " ", number_of_fermions(fconfig), " ",
          baseN_string_to_number(fermiN, fconfig),
          np.int(sign_of_c(fconfig,3)))
                        

statenum = int(0.3927*n_all_states)
print(statenum)
fs1 = FermionState(nsites, statenum)
print("fs1")
fs1.print()
print(fs1.fermiN)
fs2 = copy.deepcopy(fs1)
fs2.print()
fs2.Fconfig[0] = 1
fs2.print()
fs1.print()
fs2.isZero=True
fs2.print()
fs1.print()

print("fs1 ", fs1.statenum(), fs1.num_fermions())
print("fs2 ", fs2.statenum(), fs2.num_fermions())

fs1.print()
for isite in range(0,nsites):
    print(fs1.CdagC_Op(isite))

fs=FermionState(nsites, 0)
fs.print()
for isite in range(0,nsites):
    fs.Cdagger_Op(isite,1)
    print(" ")
    print("Created fermion at ", isite)
    fs.print()

for isite in range(nsites-1,-1,-1):
    fs.Cdagger_Op(isite,-1)
    print(" ")
    print("Destroyed fermion at ", isite)
    fs.print()

fstatenum = np.zeros(Mf,dtype=np.int)
fockspace = {}
for ifstate in range(0,n_fock_space_states):
    number_to_baseN_string(ifstate, n_all_states, Mf, fstatenum)
    fsstate = FockSpaceState(Mf, nsites, fstatenum)
    fnum = []
    for i in range(0,Mf):
        fnum.append(fsstate.FermState[i].num_fermions())
    if fnum == nparticles :
        print(" Hilbstate ")
        fockspace[ifstate] = fsstate
        print(ifstate, fstatenum)
    else :
        del fsstate
print("Dimension of Fock Space : ", len(fockspace))


#print(fockspace)

#exit()

#for fstate in fockspace:
#    print("==================================================")
#    print(fstate,fockspace[fstate].fock_state_number())
#    fockspace[fstate].print()

n_dim_fs = len(fockspace)
statelist = list(fockspace.keys())
print(statelist)
mat_idx = {}
for i in range(0,len(statelist)):
    mat_idx[statelist[i]] = i
print(mat_idx)

fst = fockspace[statelist[1]]
fst.print()

# for isite in range(0,nsites):
#     print(" ")
#     print(bcolors.FAIL+"------------New Site-------------"+bcolors.ENDC)
#     for iflv in range(0,Mf):
#         print(" ")
#         print("--------------------------------------------------------")
#         print("Operator acting at site ", isite, "on flavor", iflv)

#         st = copy.deepcopy(fst)
#         print(">>-----------Before number ")
#         st.print()
#         num = st.CdagC_Op(isite,iflv)
#         print("number operator acting at site ", isite, "on flavor", iflv)
#         print(bcolors.OKGREEN+">>------- value of number operator "+bcolors.ENDC, num)

#         print(bcolors.WARNING+"Creation Operator acting at site "+bcolors.ENDC, isite, "on flavor", iflv)
#         st.Cdagger_Op(isite, iflv, 1)

#         st.print()
#         del st
        
#         st = copy.deepcopy(fst)
#         print("----------------")
#         st.print()
#         print(bcolors.HEADER+"Annihilation Operator acting at site "+bcolors.ENDC, isite, "on flavor", iflv)
#         st.Cdagger_Op(isite, iflv, -1)
#         st.print()
#         del st
        
        
#     print(">>---------------Double Occupancy at site ")
#     fst.print()
#     docc=fst.Prod_CdagC_Op(isite)
#     print(bcolors.OKBLUE+">>------- value of double occupancy "+bcolors.ENDC, "at site",isite, " is ", docc)
    

# U=1
# Isite=0
# D0 = FockSpaceOperator(Mf,U,False,'D')
# d0oper = EOper('F',Mf,Isite,'D',1,'D')
# D0.oper_list.append(d0oper)
# D0.print()
# Isite=1
# D1 = FockSpaceOperator(Mf,U,False,'D')
# d1oper = EOper('F',Mf,Isite,'D',1,'D')
# D1.oper_list.append(d1oper)
# D1.print()

ham = []
#set hubbard U
U=6
for isite in range(0,2):
    D0 = FockSpaceOperator(Mf,U,False,'D')
    d0oper = EOper('F',Mf,isite,'D',1,'D')
    D0.oper_list.append(d0oper)
    D0.print()
    ham.append(D0)
    del D0
#set er
er=1 
for isite in range(2,4):
    for iflv in range(0,Mf):
        oer = FockSpaceOperator(Mf,er,False,'R')
        eroper = EOper('F',iflv,isite,'N',1,'R')
        oer.oper_list.append(eroper)
        oer.print()
        ham.append(oer)
        del oer
#set kinetic energy
t=np.pi
tr=np.pi
tp=np.pi
thop_ar = [t,tp,tr,tp]
for isite in range(0,4):
    jsite = (isite+1)%4
    for iflv in range(0,Mf):
        hop = FockSpaceOperator(Mf,-thop_ar[isite],True,'H')
        destroy_i = EOper('F',iflv,isite,'C',-1,'H')
        create_j = EOper('F',iflv,jsite,'C',1,'H')
        hop.oper_list.append(destroy_i)
        hop.oper_list.append(create_j)
        
        ham.append(hop)
        del destroy_i
        del create_j
    
for fsoper in ham:
    print("=====================================")
    fsoper.print()


hmat = np.zeros((n_dim_fs,n_dim_fs), np.cdouble)
    
for fsoper in ham:
    for isnum in fockspace:
        ifsstate = copy.deepcopy(fockspace[isnum])
        i = mat_idx[isnum]
        factor = fsoper.act_on(ifsstate)
        if  not ifsstate.isZero :
            jsnum = ifsstate.fock_state_number()
            j = mat_idx[jsnum]

            print(i,j)

            if i == j :
                print("-----------------------------------")
                fsoper.print()

            mat_elem = factor*fsoper.coeff   
            hmat[j,i] += mat_elem
            if fsoper.hermitian :
                hmat[i,j] += np.conj(mat_elem)
        del ifsstate

##print(hmat)
                
eig,evs = la.eigh(hmat)

#gs = np.zeros(n_dim_fs,np.cdouble)

print(eig)

gs = evs[:,0]
print(gs)

print(ExpectationValue(ham, fockspace, mat_idx, gs))

#print(evs)

exit()

for istate in statelist:
    print(bcolors.OKBLUE+" State : "+bcolors.ENDC, istate)

    fockspace[istate].print()
    st = copy.deepcopy(fockspace[istate])
    coeff = D0.act_on(st)
    print("coeff = ", coeff)
    st.print()
    print(st.fock_state_number())





    

