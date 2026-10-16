import numpy
import math
import random
import time
import freud
import gsd.hoomd
import cupy as cp
import hoomd
import h5py
import datetime
import sys
import time
from collections import defaultdict
import csv
import ast

################################################################################
########################## Initial values#######################################
################################################################################
seed3 = random.randint(1,65535) ##Random seed
print(seed3)
nlat=[4,2,1]
k_stiff =1
a = 1
alat = 50
rcut = math.pow(2/a,1/6.)
Lx=(alat+0.85)*(nlat[0])*2
Ly=(alat)*nlat[1]*2
Lz=(alat+0.6)*(nlat[2])*2
Lxby2=Lx/2.
Lyby2=Ly/2.
Lzby2=Lz/2.
rmax = (math.sqrt(3)/4+0.15)*alat
print(rmax)
coords=[]
period1 = 100
nsteps=1e5
kbT = 1.0
kbT1 = 1.0
sigma=1.0
eps = 1.0
#dt = 0.001
dt = 0.0001
dof =0.0
Krelax=600
k = int(sys.argv[9])
#k1=5000
k1=int(sys.argv[8])
kang = 1000
r0=1.5*sigma
delta=0.0
Ktors = float(sys.argv[2])
print(Ktors)
kmax = float(sys.argv[6])
#kdihed = numpy.round(-4*Ktors+4*numpy.sqrt(kmax*Ktors),decimals=2)
kdihed = numpy.round(-4*Ktors+4*numpy.sqrt(kmax*Ktors),decimals=2)
Kdihed = kmax-kdihed
no_beads = int(sys.argv[1])
sample = sys.argv[3]
k_constraint =10000
rotperiod = int(sys.argv[4])
relaxationp = int(sys.argv[5])
langperiod = int(sys.argv[7])
interval =1
period_angle = 20
freq = int((rotperiod+relaxationp)/period_angle)
#freq1= 4*int(rotperiod/period_angle)
freq1= int((rotperiod+relaxationp)/period_angle)
rm =.957
k_stiff = float(sys.argv[10])
nsidebead=0
################################################################################
#######      Status class to print timesteps and remaining time ################
################################################################################
class Status():

    def __init__(self, sim):
        self.sim = sim

    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))

class anglelog():

    def __init__(self, trapobj,sim,rot,rel,lang):
        self.obj = trapobj
        self.sim = sim
        self.rot = rot
        self.rel = rel
        self.l = lang

    @property
    def angles(self):
        #print("timestep for anglesnap :",self.sim.timestep,(self.sim.timestep-(self.rot+self.l+self.rel))%(self.rot+self.rel),(self.sim.timestep-(self.rot+self.l))%(self.rot+self.rel))
        return numpy.asarray(self.obj.get_angles('ditype'))
###############################################################################
###############Custom Trigger##################################################
################################################################################
class CustomTrigger(hoomd.trigger.Trigger):

    def __init__(self,period,phase):
        hoomd.trigger.Trigger.__init__(self)
        self.period = period
        self.phase = phase


    def compute(self, timestep):
        if (timestep -(self.phase))%self.period==0:
            return True
        else:
            return False

################################################################################

class TorsionalModifer(hoomd.custom.Action):

    def __init__(self, trap_sinaction,di_har,period1,period2,phase1,phase2,integ):
        super().__init__()
        self.trap = trap_sinaction
        self.dih = di_har
        self.period1 = period1
        self.period2 = period2
        self.phase1 = phase1
        self.phase2 = phase2
        self.integ = integ
        self.integ.forces.append(self.trap)
    def act(self, timestep):
        # epsilon = 1.0 + 4.0 * timestep / 1e6
        # self.lj.params[('A', 'A')] = dict(epsilon=epsilon, sigma=1)
        if (timestep-self.phase1)%self.period1==0:
            if ((timestep-self.phase1)//self.period1)%2==0:
                print("triggered rotation: ", timestep,(timestep-self.phase1)%self.period1,((timestep-self.phase1)//self.period1)%2)
                self.integ.forces.append(self.dih)
                self.trap.params['ditype'] = dict(k =Ktors,d=-1,n=2,chk = -1)
                # self.trap.set_userangles('ditype',0)
                # self.trap.set_shiftangles('ditype',0)
                self.trap.set_angles('ditype',refang_bool =True)
                
                self.dih.params['ditype'] = dict(k =kdihed,d=1,n=1,phi0 = 0)
            else:
                print("triggered rotation: ",timestep,(timestep-self.phase1)%self.period1,((timestep-self.phase1)//self.period1)%2)
                self.integ.forces.append(self.dih)
                self.trap.params['ditype'] = dict(k =Ktors,d=-1,n=2,chk = -1)
                # self.trap.set_userangles('ditype',numpy.pi)
                # self.trap.set_shiftangles('ditype',numpy.pi)
                self.trap.set_angles('ditype',refang_bool =True)
                self.dih.params['ditype'] = dict(k =kdihed,d=-1,n=1,phi0 = 0)
        else:
            if (timestep-self.phase2)%self.period2==0:
                print("triggered relaxation: ",timestep,(timestep-self.phase2)%self.period2)

                self.integ.forces.remove(self.dih)
                #self.trap.params['ditype'] = dict(k =44.5,d=-1,n=2,chk = 0)
                self.trap.params['ditype'] = dict(k =Ktors,d=-1,n=2,chk = 0)

                # self.trap.set_userangles('ditype',0)
                # self.trap.set_shiftangles('ditype',0)
                self.trap.set_angles('ditype',refang_bool =True)


###############################################################################
####################Selecting device ###########################################
################################################################################
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu,seed=seed3)
################################################################################
################## Getting the particle data from initial file ################
################################################################################
#filename="../initial"+"_ucellHarBig_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#filename="initial"+"_ucellHarMOI_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#filename ="initialtest1.gsd"

#filename="./PulledFinalGel"+"_PBCyzHar"+str(sample)+"_stiff"+str(k_stiff)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(cforce)+"_kT_"+str(kbT)+".gsd"
filename="RelaxedGel"+str(int(sample))+"_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".gsd"

################################################################################
######### Reading last line of initial sim file and creating a state ###########
################################################################################


sim.create_state_from_gsd(filename,frame=-1)

##############################################################################
############### parameters for the integrators############################
##############################################################################
integrator = hoomd.md.Integrator(dt=dt,integrate_rotational_dof=True)

ntree = hoomd.md.nlist.Tree(buffer=.5,exclusions=['body','bond','special_pair'])


################################################################################
##### WCA  interaction #########################################################
################################################################################
wca = hoomd.md.pair.LJ(nlist =ntree,default_r_cut = rcut*sigma,mode = "shift")

wca.params[('X','A')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','R')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','C')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','B')] = {'sigma':sigma,'epsilon': eps}


wca.params[('C','A')] = {'sigma':sigma,'epsilon': eps}
wca.params[('C','B')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','B')] = {'sigma':sigma,'epsilon': eps}


wca.params[('R','C')] = {'sigma':sigma,'epsilon': eps}
wca.params[('R','A')] = {'sigma':sigma,'epsilon': eps}
wca.params[('R','B')] = {'sigma':sigma,'epsilon': eps}


wca.params[('R','R')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','X')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','A')] = {'sigma':sigma,'epsilon': eps}
wca.params[('C','C')] = {'sigma':sigma,'epsilon': eps}
wca.params[('B','B')] = {'sigma':sigma,'epsilon': eps}

wca.params[('R','S')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','S')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','S')] = {'sigma':sigma,'epsilon': eps}
wca.params[('C','S')] = {'sigma':sigma,'epsilon': eps}
wca.params[('B','S')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','S')] = {'sigma':sigma,'epsilon': eps}


################################################################################
##################  Dihedral potential #########################################
################################################################################
#di_har = hoomd.md.dihedral.Harmonic()
di_har = hoomd.md.dihedral.Periodic()
di_har.params['ditype'] = dict(k =kdihed,d=-1,n=1,phi0 = 0)
#di_har.params['ditype2'] = dict(k =4*kdihed,d=-1,n=1,phi0 = 0)
#di_har.params['ditype3'] = dict(k =4*kdihed,d=-1,n=1,phi0 = 0)
################################################################################
########### Impropertypes #####################################################
###############################################################################
improp = hoomd.md.improper.Harmonic()
improp.params['imp'] = dict(k =kdihed,chi0 = 0)


######################FENEWCA bonds ############################################
###############################################################################

fenewca = hoomd.md.bond.FENEWCA()
fenewca.params['R-X'] = dict(k=k,r0=r0,epsilon=eps,sigma=sigma,delta=0)
#fenewca.params['R-X'] = dict(k=k,r0=r0,epsilon=eps,sigma=sigma,delta=delta)
fenewca.params['poly'] = dict(k=k1,r0=r0,epsilon=eps,sigma=sigma,delta=0)
fenewca.params['wall'] = dict(k=0,r0=r0,epsilon=0,sigma=sigma,delta=0)
fenewca.params['wall-poly'] = dict(k=0,r0=r0,epsilon=0,sigma=sigma,delta=0)

Harmonic = hoomd.md.bond.Harmonic()

Harmonic.params['R-X'] =  dict(k=0,r0=rm)
Harmonic.params['poly'] =  dict(k=0,r0=rm)
Harmonic.params['wall'] =  dict(k=1000,r0=rm)
Harmonic.params['wall-poly'] =  dict(k=1000,r0=math.sqrt(2))


################################################################################
######## Angles#################################################################
################################################################################
harmonic = hoomd.md.angle.Harmonic()
harmonic.params['A-R-A'] = dict(k=kang,t0=math.pi)
harmonic.params['A-R-X'] = dict(k=kang,t0=math.pi/2.)
harmonic.params['pangles'] = dict(k=k_stiff,t0=math.pi)
harmonic.params['wall_angles'] = dict(k=kang,t0=math.pi)
################################################################################
################# Trapping potential ###########################################
################################################################################

trap_sinaction = hoomd.md.dihedral.TorsionalDihedral()
trap_sinaction.params['ditype'] = dict(k =Ktors,d=-1,n=2,chk = -1)

loggerTors = hoomd.logging.Logger(categories=['particle', 'particle','particle','particle'])
#loggerTors=hoomd.logging.Logger(categories['dihedral','dihedral','dihedral','dihedral'])
loggerTors.add(trap_sinaction, quantities=['energies', 'forces','torques','virials'])
loggerTors.add(di_har, quantities=['energies', 'forces','torques','virials'])
#exit()
############################constrain distance##################################
Constraint=hoomd.md.special_pair.LJ()
Constraint.params['const'] = dict(epsilon=0, sigma=1.0)
Constraint.r_cut['const'] = math.pow(2,1./6)

# constant = hoomd.md.force.Constant(
#     filter=hoomd.filter.All()
#     )
# constant.constant_force['D'] = (cforce,0,0)
# constant.constant_force['E'] = (-cforce,0,0)

################################################################################
##### Adding forces to calculate during integration
################################################################################
integrator.forces.append(wca)
integrator.forces.append(di_har)
integrator.forces.append(harmonic)
integrator.forces.append(Harmonic)
#integrator.forces.append(improp)
integrator.forces.append(fenewca)


################################################################################
###### Adding rigid body constraints
################################################################################
rigid = hoomd.md.constrain.Rigid(); ## Setting up constraints
rigid.body['R'] = {
    "constituent_types": ['A','A'],
    "positions":[(0.85,0,0),(-0.85,0,0)],
    "orientations":[(1,0,0,0),(1,0,0,0)]  }

rigid.body['X'] = {
    "constituent_types": ['C','C'],
    "positions":[(0.85,0,0),(-0.85,0,0)],
    "orientations":[(1,0,0,0),(1,0,0,0)] }
##############################################################################
### Create constituent particles
################################################################################
integrator.rigid = rigid

###############################################################################
### Filtering integration in hoomd and adding all operations
###############################################################################
rigid_centers_and_free_filter = hoomd.filter.Rigid(("center", "free"))
rigid_centers = hoomd.filter.Rigid(("center",))

free = hoomd.filter.Rigid(("free",))
selected=hoomd.filter.Tags(tags=[10,11,12,13])
unions=hoomd.filter.Union(free,selected)

#integrator.methods.append(npt)
mttk = hoomd.md.methods.thermostats.MTTK(kT=1.0,tau=dt*100)
npt = hoomd.md.methods.ConstantPressure(filter=rigid_centers_and_free_filter , thermostat=mttk,
tauS = dt*1000, S=[0.0,0.0,0.0,0,0,0], couple="yz",box_dof=[False,True,True,False,False,False],gamma=dof)


#nvtLang1 = hoomd.md.methods.Langevin(kT = kbT1, filter = unions)
nvtLang1 = hoomd.md.methods.Langevin(kT = kbT1, filter = rigid_centers_and_free_filter )
nvtLang1.gamma.default = 1

nvtLang1.gamma_r.default = [1.0, 1.0, 1]


integrator.methods.append(nvtLang1)

sim.operations.integrator = integrator
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(rotperiod),
#                        filename="trajN"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                        mode='wb',dynamic=['momentum','property'])

#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(20000),
#                         filename="trajExtendnew"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".gsd",
#                         mode='wb',dynamic=['momentum','property'])


#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic((rotperiod)),
#                          filename="trajExtendnewer"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                          mode='wb',dynamic=['momentum','property'])

# gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(100),
#                          filename="traj"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                          mode='wb',dynamic=['momentum','property'])

trigger1= CustomTrigger(int(rotperiod+relaxationp),int(rotperiod+relaxationp))
trigger2 =CustomTrigger(int(rotperiod+relaxationp),int(rotperiod))
triggerOr = hoomd.trigger.Or([trigger1,trigger2])


mtrigger1=CustomTrigger(int(rotperiod+relaxationp),int(langperiod+rotperiod+relaxationp))
mtrigger2=CustomTrigger(int(rotperiod+relaxationp),int(langperiod+rotperiod))
triggerOr2 = hoomd.trigger.Or([mtrigger1,mtrigger2])

torstrigger1=CustomTrigger(int(rotperiod+relaxationp),int(langperiod+rotperiod))
torstrigger2=CustomTrigger(int(rotperiod+relaxationp),int(langperiod))
torstriggerOr2 = hoomd.trigger.Or([torstrigger1,torstrigger2])

gsd_oper = hoomd.write.GSD(trigger=triggerOr2,filename="traj"+str(int(sample))+"_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".gsd",mode='wb',dynamic=['momentum','property'])




file="Angle"+str(int(sample))+"_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".h5"


###############################################################################
############ Adding logger  to operations.writers #############################
################################################################################
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter= rigid_centers_and_free_filter)

log1 = hoomd.logging.Logger(categories=['scalar','scalar'])
log1.add(thermodynamic_properties,quantities=['potential_energy','kinetic_energy'])
#gsd_oper.logger = log1
#gsd_oper.logger = loggerTors



logger = hoomd.logging.Logger(categories=['scalar', 'string','scalar'])
logger.add(sim, quantities=['timestep', 'tps'])
logger.add(thermodynamic_properties,quantities=['kinetic_temperature'])

status = Status(sim)
logger[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=triggerOr,
                          logger=logger)
Anglelog=anglelog(trap_sinaction,sim,rotperiod,relaxationp,langperiod)
sim.operations.writers.append(table)
sim.operations.computes.append(thermodynamic_properties)
###############################################################################

fenewca.params['R-X'] = dict(k=1000,r0=r0*0.85,epsilon=eps,sigma=0.85*sigma,delta=0)
#fenewca.params['poly'] = dict(k=1000,r0=r0,epsilon=eps,sigma=sigma,delta=0)

wca.params[('R','B')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','B')] = {'sigma':sigma,'epsilon': eps}

#gsd_oper1 = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(langperiod)),filename="NewPulledFinalGel_PBCyzHar"+str(int(sample))+"_stiff"+str(k_stiff)+"_FeringGel_"+"nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".gsd",mode='wb',dynamic=['momentum','property'])




#sim.operations += gsd_oper1
sim.run(langperiod)
#sim.operations -= gsd_oper1
integrator.forces.append(trap_sinaction)
###############################################################################
sim.operations += gsd_oper
#gsd_oper.logger = log1
gsd_oper.logger = loggerTors

#anglelog = hoomd.logging.log(trap_sinaction.get_angles('ditype'),category='sequence',requires_run=True))
#exit()
anglelog_ = hoomd.logging.Logger(hoomd.write.HDF5Log.accepted_categories)
anglelog_[('anglelog','angles')] = (Anglelog,'angles','sequence')

hdf5_log = hoomd.write.HDF5Log(
    trigger=triggerOr2,
    filename=file,
    logger=anglelog_,mode='w')
sim.operations.writers.append(hdf5_log)
#exit()
trap_sinaction1 = hoomd.md.dihedral.TorsionalDihedral()
trap_sinaction1.params['ditype'] = dict(k =10,d=-1,n=2,chk = -1)
integrator.forces.remove(di_har)
integrator.forces.remove(trap_sinaction)

Tors_modifier = TorsionalModifer(trap_sinaction,di_har,int(rotperiod+relaxationp),int(rotperiod+relaxationp),int(langperiod),int(rotperiod+langperiod),integrator)
Tors_updater = hoomd.update.CustomUpdater(
    trigger=torstriggerOr2, action=Tors_modifier)
sim.operations.updaters.append(Tors_updater)

sim.run(int(600*(2*rotperiod+2*relaxationp)))
