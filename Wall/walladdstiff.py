import hoomd
import numpy
import math
import random
import time
import freud
import gsd.hoomd
import sys
import cupy as cp
import h5py
import datetime

#############################
seed3 = random.randint(1,65535) ##Random seed
print(seed3)
nlat=[4,2,1]
a = 1
no_beads = int(sys.argv[2])
cforce = 0.1
k_stiff = float(sys.argv[3])
dof=0
sample =str(sys.argv[1])
rcut = math.pow(2/a,1/6.)


coords=[]
period1 = 100
nsteps=1e5
kbT = 1.0
sigma=1.0
eps = 1.0
dt = 0.0005
k = 200
k1=200
kang = 1000
kdihed=100
r0=1.5
delta=0.0
Torque=70
rm=0.957
#k1=5000
rotperiod = 20000
relaxationp = 20000
langperiod = 50000
interval = 5
period_angle = 20
freq = int((rotperiod+relaxationp)/period_angle)
#freq1= 4*int(rotperiod/period_angle)
freq1= int((rotperiod+relaxationp)/period_angle)
###############################################################################
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
        return str(datetime.timedelta(seconds=self.seconds_remaining))#############################################################################
################################################################################
filename="EquilibriumSlab_stiff"+str(k_stiff)+"_PBCyzHar_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"

with gsd.hoomd.open(filename,mode='r') as traj:
    snap =traj[0]
    bonds  = numpy.copy(snap.bonds.group)
    bondtypeid  = numpy.copy(snap.bonds.typeid)
    angles = numpy.copy(snap.angles.group)
    angletype =snap.angles.types
    bondtype =snap.bonds.types
    angletypeid=numpy.copy(snap.angles.typeid)
    dihedrals = numpy.copy(snap.dihedrals.group)
    dihedralstype =snap.dihedrals.types
    dihedralstypeid = numpy.copy(snap.dihedrals.typeid)
    impropers = numpy.copy(snap.impropers.group)
    improperstype =snap.impropers.types
    improperstypeid = numpy.copy(snap.impropers.typeid)
    spairs = numpy.copy(snap.pairs.group)
    spairtype =snap.pairs.types
    spairtypeid = numpy.copy(snap.pairs.typeid)
    position = numpy.copy(snap.particles.position)
    typeid = numpy.copy(snap.particles.typeid)
    types =snap.particles.types
    diameter = numpy.copy(snap.particles.diameter )
    orientation = numpy.copy(snap.particles.orientation)
    moment_inertia = numpy.copy(snap.particles.moment_inertia )
    angmom = numpy.copy(snap.particles.angmom)
    image = numpy.copy(snap.particles.image)
    body = numpy.copy(snap.particles.body)
    vel = numpy.copy(snap.particles.velocity)
    Lx = numpy.copy(snap.configuration.box[0])
    Ly = numpy.copy(snap.configuration.box[1])
    Lz = numpy.copy(snap.configuration.box[2])

box1 = freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
real_pos=image*numpy.array([Lx,0,0])+position

Lyby2=Ly/2
Lzby2=Lz/2

yplus=numpy.arange(-Lyby2,Lyby2,1)
zplus=numpy.arange(-Lzby2,Lzby2,1)
last_id = len(position)
till_pos = numpy.copy(real_pos)
maxcoordx = numpy.max(real_pos[:,0])
mincoordx = numpy.min(real_pos[:,0])
image[:,0]=0

cfx = (maxcoordx+mincoordx)/2
nLxby2 = (maxcoordx-mincoordx)/2

real_pos[:,0]=real_pos[:,0]-cfx
maxcoordx = numpy.max(real_pos[:,0])
mincoordx = numpy.min(real_pos[:,0])

for j in range(1,4):

    for i in range(len(yplus)):

        xrepeat = numpy.repeat((maxcoordx+(1)*sigma)+j,len(zplus))
        yrepeat = numpy.repeat(yplus[i],len(zplus))
        point = numpy.asarray((xrepeat,yrepeat,zplus)).T

        real_pos=numpy.append(real_pos,point,axis=0)

        typeid = numpy.append(typeid,numpy.repeat(5,len(zplus)))
        moment_inertia = numpy.append(moment_inertia,numpy.repeat(numpy.asarray([[0,0,0]]),len(zplus),axis=0),axis=0)
        orientation = numpy.append(orientation,numpy.repeat(numpy.asarray([[1,0,0,0]]),len(zplus),axis=0),axis=0)
        image = numpy.append(image,numpy.repeat(numpy.asarray([[0,0,0]]),len(zplus),axis=0),axis=0)
        diameter = numpy.append(diameter,numpy.repeat(1,len(zplus)))
        body = numpy.append(body,numpy.repeat(-1,len(zplus)))
        vel = numpy.append(vel,numpy.repeat(numpy.asarray([[0,0,0]]),len(zplus),axis=0),axis=0)
        angmom = numpy.append(angmom,numpy.repeat(numpy.asarray([[0,0,0,0]]),len(zplus),axis=0),axis=0)




    for i in range(len(yplus)):

        xrepeat = numpy.repeat((mincoordx-(1)*sigma*2)-j,len(zplus))
        yrepeat = numpy.repeat(yplus[i],len(zplus))
        point = numpy.asarray((xrepeat,yrepeat,zplus)).T

        real_pos=numpy.append(real_pos,point,axis=0)

        typeid = numpy.append(typeid,numpy.repeat(6,len(zplus)))
        moment_inertia = numpy.append(moment_inertia,numpy.repeat(numpy.asarray([[0,0,0]]),len(zplus),axis=0),axis=0)
        orientation = numpy.append(orientation,numpy.repeat(numpy.asarray([[1,0,0,0]]),len(zplus),axis=0),axis=0)

        image = numpy.append(image,numpy.repeat(numpy.asarray([[0,0,0]]),len(zplus),axis=0),axis=0)
        diameter = numpy.append(diameter,numpy.repeat(1,len(zplus)))
        body = numpy.append(body,numpy.repeat(-1,len(zplus)))
        vel = numpy.append(vel,numpy.repeat(numpy.asarray([[0,0,0]]),len(zplus),axis=0),axis=0)
        angmom = numpy.append(angmom,numpy.repeat(numpy.asarray([[0,0,0,0]]),len(zplus),axis=0),axis=0)

box1 = freud.box.Box(Lx=nLxby2*2.+80,Ly=Ly,Lz=Lz)

idsel=((typeid==5)|(typeid==6))
idpos = real_pos[idsel]
totids = numpy.arange(0,len(real_pos))
seltotids = totids[idsel]
new_aq1 = freud.locality.AABBQuery(box1,idpos)
new_query_result1 = new_aq1.query(idpos, dict(r_max=math.sqrt(2)*sigma+0.15,exclude_ii=True))
new_nlist1 = new_query_result1.toNeighborList()
s1 =numpy.asarray(new_nlist1)
print(s1)
s1[:,0] = seltotids[s1[:,0]]
s1[:,1] = seltotids[s1[:,1]]


s1=numpy.sort(s1,axis=1)
s1 = numpy.unique(s1,axis=0)
bonds = numpy.append(bonds,s1,axis=0)
bondtypeid = numpy.append(bondtypeid,numpy.repeat(2,len(s1)))
print(types)
print(bondtype)
print(angletype)
print(dihedralstype)

snap1 = gsd.hoomd.Frame()
snap1.particles.N = len(real_pos)
#snap.particles.position = position[:]
snap1.particles.position = real_pos[:]
snap1.particles.types= types
snap1.particles.body = body[:]
snap1.particles.mass = [1]*len(real_pos)

snap1.particles.typeid =typeid[:]
snap1.particles.orientation = orientation[:len(orientation)]
snap1.particles.moment_inertia = moment_inertia
snap1.particles.velocity = vel
snap1.particles.image = image[:]
snap1.particles.angmom = angmom[:]
#
snap1.bonds.N = len(bonds)

snap1.angles.N = len(angles)
snap1.bonds.group = bonds[:]
snap1.bonds.types = bondtype
snap1.bonds.typeid = bondtypeid[:]

snap1.angles.types = angletype
snap1.angles.typeid = angletypeid[:]
snap1.angles.group = angles[:]

snap1.configuration.box = [nLxby2*2.+100, Ly, Lz, 0, 0, 0]
snap1.particles.diameter = diameter[:]

snap1.dihedrals.N = len(dihedrals)
snap1.dihedrals.group = dihedrals
snap1.dihedrals.types = dihedralstype
snap1.dihedrals.typeid = dihedralstypeid

snap1.pairs.N = len(spairs)
snap1.pairs.types = spairtype
snap1.pairs.typeid = spairtypeid
snap1.pairs.group = spairs
# print(snap1.bonds.type)
# print(snap1.bonds.typeid)
############### parameters for the integrators############################
##############################################################################
integrator = hoomd.md.Integrator(dt=dt,integrate_rotational_dof=True)
ntree = hoomd.md.nlist.Tree(buffer=.51,exclusions=['bond','body','special_pair'])
##### WCA  interaction #####################################
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

wca.params[('R','D')] = {'sigma':sigma,'epsilon': 0}
wca.params[('X','D')] = {'sigma':sigma,'epsilon': 0}
wca.params[('A','D')] = {'sigma':sigma,'epsilon': 0}
wca.params[('C','D')] = {'sigma':sigma,'epsilon': 0}
wca.params[('B','D')] = {'sigma':sigma,'epsilon': 0}
wca.params[('D','D')] = {'sigma':sigma,'epsilon': eps}

wca.params[('R','E')] = {'sigma':sigma,'epsilon': 0}
wca.params[('X','E')] = {'sigma':sigma,'epsilon': 0}
wca.params[('A','E')] = {'sigma':sigma,'epsilon': 0}
wca.params[('C','E')] = {'sigma':sigma,'epsilon': 0}
wca.params[('B','E')] = {'sigma':sigma,'epsilon': 0}
wca.params[('D','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('E','E')] = {'sigma':sigma,'epsilon': eps}


LJ = hoomd.md.pair.LJ(nlist =ntree,default_r_cut = 2.5*sigma*math.sqrt(2))
LJ.params[('X','A')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('X','R')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('X','C')] = {'sigma':math.sqrt(2)*sigma,'epsilon':0}
LJ.params[('X','B')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}


LJ.params[('C','A')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('C','B')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('A','B')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}


LJ.params[('R','C')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('R','A')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('R','B')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}


LJ.params[('R','R')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('X','X')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('A','A')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('C','C')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}
LJ.params[('B','B')] = {'sigma':math.sqrt(2)*sigma,'epsilon': 0}

LJ.params[('R','D')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('X','D')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('A','D')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('C','D')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('B','D')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
#LJ.params[('B','D')] = {'sigma':sigma,'epsilon': eps}
LJ.params[('D','D')] = {'sigma':sigma,'epsilon': 0}


LJ.params[('R','E')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('X','E')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('A','E')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('C','E')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
LJ.params[('B','E')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
#LJ.params[('B','D')] = {'sigma':sigma,'epsilon': eps}
LJ.params[('D','E')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('E','E')] = {'sigma':sigma,'epsilon': 0}


################################################################################
#########  Dihedral potential #################################################
#############################################################################3#
di_har = hoomd.md.dihedral.Periodic()
di_har.params['ditype'] = dict(k =100,d=-1,n=1,phi0 = 0)

################################################################################
####### FENEWCA ################################################################
################################################################################
fenewca = hoomd.md.bond.FENEWCA()
#fenewca.params['R-X'] = dict(k=k,r0=r0,epsilon=0,sigma=sigma,delta=sigma)
fenewca.params['R-X'] = dict(k=k1,r0=r0,epsilon=eps,sigma=sigma,delta=0)
fenewca.params['poly'] = dict(k=k1,r0=r0,epsilon=eps,sigma=sigma,delta=0)
fenewca.params['wall'] =  dict(k=0,r0=rm,epsilon=0,sigma=sigma,delta=0)
fenewca.params['wall-poly'] =  dict(k=0,r0=rm,epsilon=0,sigma=sigma,delta=0)


Harmonic = hoomd.md.bond.Harmonic()

Harmonic.params['R-X'] =  dict(k=500,r0=rm)
Harmonic.params['poly'] =  dict(k=500,r0=rm)
Harmonic.params['wall'] =  dict(k=1000,r0=rm)
Harmonic.params['wall-poly'] =  dict(k=1000,r0=math.sqrt(2))



# ################################################################################
######## Angles#################################################################
################################################################################
harmonic = hoomd.md.angle.Harmonic()
harmonic.params['A-R-A'] = dict(k=kang,t0=math.pi)
harmonic.params['A-R-X'] = dict(k=kang,t0=math.pi/2.)
harmonic.params['pangles'] = dict(k=k_stiff,t0=math.pi)

###############################################################################
############ Constraint#######################################################
#############################################################################
Constraint = hoomd.md.special_pair.LJ()
Constraint.params['const'] = dict(epsilon=0,sigma=1.0)
Constraint.r_cut['const'] = math.pow(2,1./6)
######################Constant Force ###########################################
################################################################################
constant = hoomd.md.force.Constant(
    filter=hoomd.filter.All()
    )
constant.constant_force['D'] = (cforce,0,0)
constant.constant_force['E'] = (-cforce,0,0)
################################################################################
#####   To compress the walls ################################################
###############################################################################
constant1 = hoomd.md.force.Constant(
    filter=hoomd.filter.All()
    )
constant1.constant_force['D'] = (1,0,0)
constant1.constant_force['E'] = (-1,0,0)
#constant1.constant_force['D'] = (-1,0,0)
#constant1.constant_force['E'] = (1,0,0)


################################################################################
##### Adding forces to calculate during integration
################################################################################
integrator.forces.append(wca)
#integrator.forces.append(fenewca)
integrator.forces.append(harmonic)
integrator.forces.append(Harmonic)
integrator.forces.append(di_har)
integrator.forces.append(Constraint)


################################################################################
###### Adding rigid body constraints
################################################################################
rigid = hoomd.md.constrain.Rigid(); ## Setting up constraints


rigid = hoomd.md.constrain.Rigid(); ## Setting up constraints
rigid.body['R'] = {
    "constituent_types": ['A','A'],
    "positions":[(0.85,0,0),(-0.85,0,0)],
    "orientations":[(1,0,0,0),(1,0,0,0)]  }

rigid.body['X'] = {
    "constituent_types": ['C','C'],
    "positions":[(0.85,0,0),(-0.85,0,0)],
    "orientations":[(1,0,0,0),(1,0,0,0)] }
###############################################################################
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu,seed=seed3)
sim.create_state_from_snapshot(snap1)
###############################################################################
### Create constituent particles
################################################################################
integrator.rigid = rigid
integrator.forces.append(LJ)
# integrator.forces.append(constant1)
rigid_centers_and_free_filter = hoomd.filter.Rigid(("center", "free"))


mttk = hoomd.md.methods.thermostats.MTTK(kT=1.0,tau=dt*100)
npt1 = hoomd.md.methods.ConstantPressure(filter=rigid_centers_and_free_filter , thermostat=mttk,
tauS = dt*1000, S=[0.02,0.001,0.001,0,0,0], couple="none",box_dof=[True,False,False,False,False,False],gamma=dof)
integrator.methods.append(npt1)
filename="Walladd"+"_PBCyzHar"+str(sample)+"_stiff"+str(k_stiff)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"

gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(period=500000), filename=filename,mode='wb',dynamic=['momentum','property'])
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.And([hoomd.trigger.After(400000),hoomd.trigger.Periodic(100)]), filename=filename,mode='wb',dynamic=['momentum','property'])
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(100), filename=filename,mode='wb',dynamic=['momentum','property'])
# logger = hoomd.logging.Logger(['particle','constraint'])
sim.operations.integrator = integrator

sim.operations += gsd_oper

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=rigid_centers_and_free_filter)
sim.operations.computes.append(thermodynamic_properties)

logg1= hoomd.logging.Logger(categories=['scalar', 'string','scalar'])
logg1.add(sim, quantities=['timestep', 'tps'])
logg1.add(thermodynamic_properties,quantities=['kinetic_temperature'])

status = Status(sim)
logg1[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(100000),
                          logger=logg1)
sim.operations.writers.append(table)
################################################################################
sim.run(400000)
# integrator.forces.remove(constant1)
integrator.methods.remove(npt1)
nvtLang1 = hoomd.md.methods.Langevin(kT = kbT, filter = rigid_centers_and_free_filter)
integrator.methods.append(nvtLang1)

#integrator.forces.remove(Harmonic)
Harmonic.params['R-X'] =  dict(k=0,r0=rm)
Harmonic.params['poly'] =  dict(k=0,r0=rm)

integrator.forces.append(fenewca)
fenewca.params['poly'] = dict(k=k1,r0=0.85*sigma*r0,epsilon=eps,sigma=0.85*sigma,delta=0)

sim.run(100000)
