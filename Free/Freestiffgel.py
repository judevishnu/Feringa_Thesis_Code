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
dt = 0.0005
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
###############################################################################
########################Dict reader #########################################
##############################################################################
def listofdictreader(filename0):
    tc_trial = defaultdict(list)

    with open(filename0,'r') as data:
        reader = csv.reader(data, delimiter='\t', quoting=csv.QUOTE_NONE)
        #reader = csv.reader(data, delimiter='\t')
        for x in reader:
        # y=list(map(str.strip,x[0].strip('][').replace('"', '').split(',')))

            y = ast.literal_eval(x[0])
            if isinstance(y[1],list)==False:
                z=list(map(float, ast.literal_eval(y[1])))
            else:
                z = y[1]
            tc_trial[int(y[0])]=z
    return tc_trial

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

################################################################################

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
##############################################################################
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
filename="initialSidechain_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"

#filename="../initial"+"_ucellHarsing_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#filename="initial"+"_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
traj = gsd.hoomd.open(name= filename,mode='r')
snap = gsd.hoomd.Frame()

bonds  = numpy.copy(traj[len(traj)-1].bonds.group)
bondtypeid  = numpy.copy(traj[len(traj)-1].bonds.typeid)
angles = numpy.copy(traj[len(traj)-1].angles.group)
angletype =traj[len(traj)-1].angles.types
angletypeid = numpy.copy(traj[len(traj)-1].angles.typeid)
position = numpy.copy(traj[len(traj)-1].particles.position)
typeid = numpy.copy(traj[len(traj)-1].particles.typeid)
types = traj[len(traj)-1].particles.types
diameter = numpy.copy(traj[len(traj)-1].particles.diameter )
orientation = numpy.copy(traj[len(traj)-1].particles.orientation)
moment_inertia = numpy.copy(traj[len(traj)-1].particles.moment_inertia )
angmom = numpy.copy(traj[len(traj)-1].particles.angmom)
image = numpy.copy(traj[len(traj)-1].particles.image)
body = numpy.copy(traj[len(traj)-1].particles.body)
vel = numpy.copy(traj[len(traj)-1].particles.velocity)
Lx = numpy.copy(traj[len(traj)-1].configuration.box[0])
Ly = numpy.copy(traj[len(traj)-1].configuration.box[1])
Lz = numpy.copy(traj[len(traj)-1].configuration.box[2])
box1 = freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
bonds1 = numpy.copy(bonds)


################################################################################
###### Determining length of the initial traj and selcting
###### particles which are of types 0's and 1's ################################
################################################################################
bodyid1 = numpy.where((typeid==0))[0]
bodyid2 = numpy.where((typeid==1))[0]
bodyid = numpy.union1d(bodyid1,bodyid2)
bodyid = numpy.sort(bodyid)
################################################################################
################### selecting positive taus ids and negartive tau ids #########
################################################################################
posttau_ids = numpy.empty(0,dtype=int)
negttau_ids = numpy.empty(0,dtype=int)
body_bonds = numpy.empty((0,2),dtype=int)
for i in range(0,len(bodyid)-1,2):
     body_bonds = numpy.append(body_bonds,[numpy.asarray([bodyid[i],bodyid[i+1]])],axis=0)
for i in range(len(body_bonds)):
     temp = body_bonds[i]
     temp1 = temp.tolist()
#     #print(temp1[0],temp1[1])
     posttau_ids = numpy.append(posttau_ids,random.choice(temp1),axis=None)
#     posttau_ids = numpy.append(posttau_ids,temp1[0],axis=None)
# #exit()
id_seq = numpy.arange(0,len(bodyid),1)
negttau_ids = numpy.setdiff1d(id_seq,posttau_ids)

#posttau_ids = numpy.asarray([10,13])
#negttau_ids = numpy.asarray([11,12])

n_ids = numpy.copy(negttau_ids)
p_ids = numpy.copy(posttau_ids)
n_ids = numpy.sort(n_ids)
p_ids = numpy.sort(p_ids)

posttau_ids = cp.asarray(posttau_ids)
negttau_ids = cp.asarray(negttau_ids)

print(n_ids.shape,p_ids.shape)
#exit()

molfile="./Molecularids"+str(int(sample))+"_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".txt"

arraymol = numpy.asarray((p_ids,n_ids)).T

with open(molfile,'w+') as f:
    numpy.savetxt(f,arraymol)

################################################################################
############    Constraint pair and distance ##################################
################################################################################

#################################################################################
################## determining constrain distance pairs #########################
###############################################################################
num_body = len(bodyid)
################################################################################
##################### Selecting molecular ids that are of importance############
####################Constraint pair and distance################################
################################################################################
molecular_ids = numpy.empty(0,dtype=int)
constraint_pair = numpy.empty((0,2),dtype=int)
constraint_pair1 = numpy.empty((0,2),dtype=int)
newbonds = numpy.empty((0,2),dtype=int)
for i in range(0, num_body-1,2):
    x1 = 2*i+num_body
    x2 = 2*i + num_body+1
    x3 = 2*(i+1) + num_body
    x4 = 2*(i+1) + num_body+1
    p1 = len(numpy.where(x1==bonds1)[0])
    p2 = len(numpy.where(x2==bonds1)[0])
    p3 = len(numpy.where(x3==bonds1)[0])
    p4 = len(numpy.where(x4==bonds1)[0])
    sum_p = p1+p2+p3+p4
    constraint_pair = numpy.append(constraint_pair,[[x1,x3]],axis=0)
    constraint_pair = numpy.append(constraint_pair,[[x1,x4]],axis=0)
    constraint_pair = numpy.append(constraint_pair,[[x2,x3]],axis=0)
    constraint_pair = numpy.append(constraint_pair,[[x2,x4]],axis=0)

    constraint_pair1 = numpy.append(constraint_pair1,[[i,i+1]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x1,x3]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x1,x4]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x2,x3]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x2,x4]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x1,i+1]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x2,i+1]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x3,i]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x4,i]],axis=0)

    newbonds =numpy.append(newbonds,[[i,i+1]],axis=0)


    if sum_p ==5 or sum_p==6 or sum_p ==7:
        molecular_ids = numpy.append(molecular_ids,[int(i/2)],axis=0)


molecular_ids = numpy.unique(molecular_ids)
body_id_index = numpy.arange(0,len(p_ids),1)
molecular_ids = numpy.setdiff1d(body_id_index,molecular_ids)
constraint_type1 = numpy.zeros(len(constraint_pair1))
print(molecular_ids)
dist_pair =numpy.linalg.norm(box1.wrap(position[constraint_pair[:,0]] - position[constraint_pair[:,1]]),axis=1)


#print(molecular_ids1)
print(len(molecular_ids))


selfile="./Selectedids"+str(int(sample))+"_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".txt"

with open(selfile,'w+') as f11:
    numpy.savetxt(f11,molecular_ids.T)

#exit()

################################################################################
####################### Dihedral ###############################################
################################################################################
dihedral = numpy.empty((0,4),dtype=int)
dihedraltypes = numpy.empty(0,dtype=int)




for i in range(0,num_body//2):
    x1 = 2*p_ids[i]+num_body
    x2 = p_ids[i]
    x3 = n_ids[i]
    x4 = 2*n_ids[i]+num_body



    dihedral = numpy.append(dihedral,[[x1,x2,x3,x4]],axis=0)

    dihedraltypes = numpy.append(dihedraltypes,0)

################################################################################
######### Reading last line of initial sim file and creating a state ###########
################################################################################
real_pos=image*numpy.array([Lx,Ly,Lz])+position
maxx = numpy.max(real_pos[:,0])
minx = numpy.min(real_pos[:,0])
maxy = numpy.max(real_pos[:,1])
miny = numpy.min(real_pos[:,1])
maxz = numpy.max(real_pos[:,2])
minz = numpy.min(real_pos[:,2])
Lx = 3*(maxx-minx)
Ly =3*(maxy-miny)
Lz = 3*(maxz-minz)

bondtypeid[len(newbonds):]=1
snap.particles.N = len(position)
snap.particles.position = position[:]
snap.particles.position = real_pos[:]
snap.particles.types= types[:]
snap.particles.body = body[:]
snap.particles.mass = [1]*len(position)
snap.particles.typeid =typeid[:]
snap.particles.orientation = orientation[:]
snap.particles.moment_inertia = moment_inertia[:]

snap.particles.velocity = vel[:]
#snap.particles.image = image[:]
snap.particles.angmom = angmom[:]
snap.angles.types= angletype
#
snap.bonds.N = len(bonds)
snap.angles.N = len(angles)
snap.bonds.types = ['R-X','poly']

snap.bonds.group = bonds[:]
snap.bonds.typeid = bondtypeid[:]


snap.angles.typeid = angletypeid[:]
snap.angles.group = angles[:]
snap.configuration.box = [Lx, Ly, Lz, 0, 0, 0]
snap.particles.diameter = diameter[:]

snap.dihedrals.N = len(dihedral)
snap.dihedrals.group = dihedral
snap.dihedrals.typeid = dihedraltypes
snap.dihedrals.types = ['ditype']
#snap.dihedrals.types = ['ditype','ditype2']




snap.pairs.N = len(constraint_pair1)
snap.pairs.typeid = constraint_type1
snap.pairs.types = ['const']
snap.pairs.group = constraint_pair1
sim.create_state_from_snapshot(snap)
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

wca.params[('S','R')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','X')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','A')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','C')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','B')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','S')] = {'sigma':sigma,'epsilon': eps}

################################################################################
##################  Dihedral potential #########################################
################################################################################
#di_har = hoomd.md.dihedral.Harmonic()
di_har = hoomd.md.dihedral.Periodic()
di_har.params['ditype'] = dict(k =kdihed,d=-1,n=1,phi0 = 0)
#di_har.params['ditype2'] = dict(k =4*kdihed,d=-1,n=1,phi0 = 0)
#di_har.params['ditype3'] = dict(k =4*kdihed,d=-1,n=1,phi0 = 0)


######################FENEWCA bonds ############################################
###############################################################################
fenewca = hoomd.md.bond.FENEWCA()
fenewca.params['R-X'] = dict(k=30,r0=r0,epsilon=eps,sigma=sigma,delta=0)
fenewca.params['poly'] = dict(k=k,r0=r0,epsilon=eps,sigma=sigma,delta=0)
# #fenewca.params['poly'] = dict(k=k,r0=r0,epsilon=eps,sigma=sigma,delta=0.)

Harmonic = hoomd.md.bond.Harmonic()
# Harmonic.params['R-X'] =  dict(k=k,r0=rm)
# Harmonic.params['poly'] =  dict(k=0,r0=rm)
Harmonic.params['R-X'] =  dict(k=500,r0=rm)
Harmonic.params['poly'] =  dict(k=500,r0=rm)

################################################################################
######## Angles#################################################################
################################################################################
harmonic = hoomd.md.angle.Harmonic()
harmonic.params['A-R-A'] = dict(k=kang,t0=math.pi)
harmonic.params['A-R-X'] = dict(k=kang,t0=math.pi/2.)
harmonic.params['pangles'] = dict(k=k_stiff,t0=math.pi)

################################################################################
################# Trapping potential ###########################################
################################################################################

trap_sinaction = hoomd.md.dihedral.TorsionalDihedral()
trap_sinaction.params['ditype'] = dict(k =Ktors,d=-1,n=2,chk = -1)

loggerTors = hoomd.logging.Logger(categories=['dihedral', 'dihedral','dihedral','particle'])
#loggerTors=hoomd.logging.Logger(categories['dihedral','dihedral','dihedral','dihedral'])
loggerTors.add(trap_sinaction, quantities=['energies', 'forces','torques','virials'])
############################constrain distance##################################
Constraint=hoomd.md.special_pair.LJ()
Constraint.params['const'] = dict(epsilon=0, sigma=1.0)
Constraint.r_cut['const'] = math.pow(2,1./6)

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

mttk = hoomd.md.methods.thermostats.MTTK(kT=1.0,tau=dt*100)
npt = hoomd.md.methods.ConstantPressure(filter=rigid_centers_and_free_filter , thermostat=mttk,
tauS = dt*1000, S=[0.0,0.0,0.0,0,0,0], couple="yz",box_dof=[False,True,True,False,False,False],gamma=dof)


nvtLang1 = hoomd.md.methods.Langevin(kT = 1.0, filter = rigid_centers_and_free_filter )
nvtLang1.gamma.default = 1

nvtLang1.gamma_r.default = [1.0, 1.0, 1]
trigger1= CustomTrigger(int(rotperiod+relaxationp),int(rotperiod+relaxationp))
trigger2 =CustomTrigger(int(rotperiod+relaxationp),int(rotperiod))
triggerOr = hoomd.trigger.Or([trigger1,trigger2])


mtrigger1=CustomTrigger(int(rotperiod+relaxationp),int(langperiod+rotperiod+relaxationp))
mtrigger2=CustomTrigger(int(rotperiod+relaxationp),int(langperiod+rotperiod))
triggerOr2 = hoomd.trigger.Or([mtrigger1,mtrigger2])

torstrigger1=CustomTrigger(int(rotperiod+relaxationp),int(langperiod+rotperiod))
torstrigger2=CustomTrigger(int(rotperiod+relaxationp),int(langperiod))
torstriggerOr2 = hoomd.trigger.Or([torstrigger1,torstrigger2])
integrator.methods.append(npt)

sim.operations.integrator = integrator
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(10),
#                         filename="traj"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                         mode='wb',dynamic=['momentum','property'])
# gsd_oper = hoomd.write.GSD(trigger=triggerOr2,
#                          filename="traj"+str(int(sample))+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_stiff"+str(k_stiff)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".gsd",
#                          mode='wb',dynamic=['momentum','property'])

gsd_oper1 = hoomd.write.GSD(trigger=int(langperiod),
                         filename="RelaxedGel"+str(int(sample))+"_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".gsd",
                         mode='wb',dynamic=['momentum','property'])






#file="Angle"+str(int(sample))+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_stiff"+str(k_stiff)+"_kdih_"+str(kdihed)+"_Ktors_"+str(Ktors)+"_Kmax_"+str(kmax)+"_k1_"+str(k1)+"_k_"+str(k)+"_dof_"+str(dof)+".h5"



###############################################################################
############ Adding logger  to operations.writers #############################
################################################################################
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter= rigid_centers_and_free_filter)

log1 = hoomd.logging.Logger(categories=['scalar','scalar'])
log1.add(thermodynamic_properties,quantities=['potential_energy','kinetic_energy'])
#gsd_oper.log = log1



logger = hoomd.logging.Logger(categories=['scalar', 'string','scalar'])
logger.add(sim, quantities=['timestep', 'tps'])
logger.add(thermodynamic_properties,quantities=['kinetic_temperature'])

status = Status(sim)
logger[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=triggerOr,
                          logger=logger)
#Anglelog=anglelog(trap_sinaction,sim,rotperiod,relaxationp,langperiod)
sim.operations.writers.append(table)
sim.operations.computes.append(thermodynamic_properties)
###############################################################################

integrator.forces.append(Constraint)
#sim.run(langperiod)

integrator.methods.remove(npt)

integrator.methods.append(nvtLang1)

integrator.forces.remove(Harmonic)

fenewca.params['R-X'] = dict(k=1000,r0=r0*0.85,epsilon=eps,sigma=0.85*sigma,delta=0)
fenewca.params['poly'] = dict(k=200,r0=r0,epsilon=eps,sigma=sigma,delta=0)

# wca.params[('R','B')] = {'sigma':sigma,'epsilon': eps}
# wca.params[('X','B')] = {'sigma':sigma,'epsilon': eps}


#filename="initialfinal"+"_ucellHarnonPBC_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#gsd_oper1 = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(langperiod),
#                         filename=filename,
#                         mode='wb',dynamic=['momentum','property'])
sim.operations += gsd_oper1

sim.run(langperiod)

