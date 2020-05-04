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
# main code for 4 sites bilayer nickelate
runcasename="junk.run.junk"
num_outp_states=2
runpcard = { 'casename': runcasename,
             'U':0.e0,
             't':1.e0,
             'tp':0.0e0,
             'tR':1.0e0,
             'eR':1000.e0,
             'nparticles': [1,1],
             'num_outp_states':num_outp_states
}


Ustart = 0.e0
Uend = 100.e0
nU = 10
dU = (Uend-Ustart)/np.double(nU)
casename="hub2site"
jin_file ="junk."+casename+".json.jnk"
opfilename = []
pltfilename = []
keyfilename = []
first_time = []
for iop in range(0,num_outp_states):
    iopstr = "%02d"%(iop)
    opfilename.append(casename+"_res"+iopstr+".plt")
    pltfilename.append(runcasename+"_out"+iopstr+".plt")
    keyfilename.append(runcasename+"_out"+iopstr+".key")
    first_time.append(True)

print(opfilename)
print(keyfilename)
print(pltfilename)
print(first_time)
for iU in range(0,nU+1):
    tmpprint = runpcard
    tmpprint['U'] = Ustart + np.double(iU)*dU
    with open(jin_file, 'w') as writefile:
        json.dump(tmpprint,writefile)
    rcmd = "python3 fermion.py "+ jin_file
    print("Running U = ", tmpprint['U'])
    os.system(rcmd)
    for iop in range(0,num_outp_states):
        if first_time[iop]:
            first_time[iop] = False
            cmd = "cat "+keyfilename[iop]+" > "+opfilename[iop]
            os.system(cmd)
        cmd = "cat "+pltfilename[iop]+" >> "+opfilename[iop]
        os.system(cmd)

    

