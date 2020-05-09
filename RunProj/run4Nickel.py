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
num_outp_states=12
U=6.e0
tp = 0.2e0
np=4
eRstart = 0.e0
eRend = 12.e0
neR = 150
num_outp_states=12

#------------------------read input if it is there
if(len(sys.argv) > 1) :
    runpfile = sys.argv[1]
    with open(runpfile) as runp_file:
        runpdata = json.load(runp_file)
        pprint(runpdata)
        for key in runpdata:
            if (key == 'casename') :
                runcasename = runpdata['runcasename']
            elif (key == 'U' ):
                U = runpdata['U']
            elif (key == 'tp' ):
                tp = runpdata['tp']
            elif (key == 'np'):
                np = runpdata['np']
            elif (key == 'tR'):
                tR = runpdata['tR']
            elif (key == 'eRstart'):
                eRstart = runpdata['eRstart']
            elif (key == 'eRend'):
                eRend = runpdata['eRend']
            elif (key == 'neR'):
                neR = runpdata['neR']
            elif (key == 'num_outp_states'):
                num_outp_states = runpdata['num_outp_states']





runpcard = { 'casename': runcasename,
             'U':U,
             't':1.e0,
             'tp':tp,
             'tR':1.e0,
             'eR':0.e0,
             'nparticles': [np,np],
             'num_outp_states':num_outp_states
}



print(runpcard)
print("eRstart : ", eRstart, "eRend : ", eRend, "neR : ", neR)



deR = (eRend-eRstart)/neR


casename="biL4site"+"_U"+"%3.1f"%(U)+"_tp"+"%3.1f"%(tp)+"_np"+"%0d"%(np)
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
for ieR in range(0,neR+1):
    tmpprint = runpcard
    tmpprint['eR'] = eRstart + ieR*deR
    with open(jin_file, 'w') as writefile:
        json.dump(tmpprint,writefile)
    rcmd = "python3 nickbiL4site.py "+ jin_file
    print("Running eR = ", tmpprint['eR'])
    os.system(rcmd)
    for iop in range(0,num_outp_states):
        if first_time[iop]:
            first_time[iop] = False
            cmd = "cat "+keyfilename[iop]+" > "+opfilename[iop]
            os.system(cmd)
        cmd = "cat "+pltfilename[iop]+" >> "+opfilename[iop]
        os.system(cmd)

    

