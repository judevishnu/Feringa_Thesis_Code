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
alat = float(sys.argv[2])
cforce = 0.1
k_stiff = float(sys.argv[3])
dof=0
sample =str(sys.argv[1])
rcut = math.pow(2/a,1/6.)
# Lx=(alat+0.85)*(nlat[0])*2
# Ly=(alat)*nlat[1]*2
# Lz=(alat+0.6)*(nlat[2])*2

Lx=(alat+0.5)*(nlat[0])*2
Ly=(alat)*nlat[1]*2
#Lz=(alat+0.6)*(nlat[2])*2
Lz=(alat+0.)*(nlat[2])*2
Lz_1=(alat+0.6)*(nlat[2])*2
print(Lz_1-Lz,Lz,Lz_1)
#exit()
Lxby2=Lx/2.
Lyby2=Ly/2.
Lzby2=Lz_1/2.
#rmax = (math.sqrt(3)/4+0.15)*(alat)
rmax = (math.sqrt(3)/4)*alat

print(rmax)
coords=[]
period1 = 100
nsteps=1e5
kbT = 1.0
sigma=1.0
eps = 1.0
dt = 0.001
k = 500
k1=500
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
#################################################
#############Diamond lattice#####################
def initialize_coordinates():
    basis =[[0.0,0.0,0.0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0,0.5,0.5],
            [0.25,0.25,0.25],[0.75,0.75,0.25],[0.75,0.25,0.75],[0.25,0.75,0.75]]
    for i in range(-nlat[0],nlat[0]):
        for j in range(-nlat[1],nlat[1]):
            for k in range(-nlat[2],nlat[2]):
                for m in range(8):
                    x = (basis[m][0]+i)*(alat)
                    y = (basis[m][1]+j)*(alat)
                    z = (basis[m][2]+k)*(alat)
                    # x = (basis[m][0]+i)*(alat+0.85)
                    # y = (basis[m][1]+j)*(alat+0.85)
                    # z = (basis[m][2]+k)*(alat+0.85)
                    coords.append([x,y,z])
##########################################################################
def Calculate_MOI():
    general_positions = numpy.array([[0.85, 0, 0], [-0.85, 0, 0]])
    particle_mass = 1
    particle_radius = 0.5
    I_ref = numpy.array([[2 / 5 * particle_mass * particle_radius**2, 0, 0],
                         [0, 2 / 5 * particle_mass * particle_radius**2, 0],
                         [0, 0, 2 / 5 * particle_mass * particle_radius**2]])
    print("refMOI",I_ref)
    I_general = numpy.zeros(shape=(3, 3))
    for r in general_positions:
        I_general += I_ref + particle_mass * (numpy.dot(r, r) * numpy.identity(3)
                                          - numpy.outer(r, r))

    print("generalMOI",I_general)
    I_diagonal, E_vec = numpy.linalg.eig(I_general)
    print("diagonalMOI",I_diagonal)
    R = E_vec.T
    diagonal_positions = numpy.dot(R, general_positions.T).T
    print("diagonalPos",diagonal_positions)
    return I_diagonal

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


#################################################
###Setting up cross-linkers which are rigid bodies
#################################################
initialize_coordinates()
coordinates=numpy.asarray(coords)

############Central particls position and moment of inertia#############
I = Calculate_MOI() ##moment of inertia of rigid bodies
I = numpy.array([0,0,I[2]])

pos = numpy.empty((0,3),dtype=float)
typearr = numpy.empty(0,dtype=int)
orientation=numpy.empty((0,4),dtype=float)
mass=numpy.empty(0,dtype=float)
moment_inertia=numpy.empty((0,3),dtype=float)
bonds=numpy.empty((0,2),dtype=int)
#bonds1=numpy.empty((0,2),dtype=int)3
angles =numpy.empty((0,3),dtype=int)
bondid = numpy.empty(0,dtype=int)
angleid = numpy.empty(0,dtype=int)
diameter=numpy.empty(0,dtype=float)
positive_tau=numpy.empty(0,dtype=int)
body=numpy.empty(0,dtype=int)
print(len(coordinates))
c= numpy.isclose(numpy.fabs(coordinates[:,0]),Lx/2)
coord_rem_id=numpy.where(c==True)[0]
coordinates = numpy.delete(coordinates,coord_rem_id,axis=0)
maxcoordx=numpy.max(coordinates[:,0])
mincoordx=numpy.min(coordinates[:,0])
#selcoords = coordinates[numpy.where((coordinates[:,0]==maxcoordx)|(coordinates[:,0]==mincoordx))[0]]
#coordinates = numpy.delete(coordinates,numpy.where((coordinates[:,0]==maxcoordx)|(coordinates[:,0]==mincoordx))[0],axis=0)

#print(selcoords)
# exit()

for i in range(len(coordinates)):

    temp=numpy.asarray((coordinates[i,0],coordinates[i,1],coordinates[i,2]+0.6)).T
    pos=numpy.append(pos,[temp],axis=0)
    typearr=numpy.append(typearr,0,axis=None)
    orientation=numpy.append(orientation,[[0,0,0,1]],axis=0)
    #orientation=numpy.append(orientation,[[1./sqrt(4),1./sqrt(4),1./sqrt(4),1./sqrt(4)]],axis=0)

    moment_inertia=numpy.append(moment_inertia,[I],axis=0)
    mass=numpy.append(mass,1,axis=None)
    diameter=numpy.append(diameter,1,axis=None)
    index=len(pos)-1
####################################################################
    temp=numpy.asarray((coordinates[i,0],coordinates[i,1],coordinates[i,2]-0.6)).T
    pos=numpy.append(pos,[temp],axis=0)
    typearr=numpy.append(typearr,1,axis=None)
    orientation=numpy.append(orientation,[[0,0,0,1]],axis=0)
    moment_inertia=numpy.append(moment_inertia,[I],axis=0)
    mass=numpy.append(mass,1,axis=None)
    diameter=numpy.append(diameter,1,axis=None)
    index1=len(pos)-1
    bonds=numpy.append(bonds,[[index,index1]],axis=0)
    bondid = numpy.append(bondid,0)
#N = len(pos)

################################################################################
#############           Dihedral ###############################################
################################################################################
# all_bonds = numpy.stack(
#     [
#         pos[bonds[:, 0]],
#         pos[bonds[:, 1]],
#     ],
#     axis=1,
# )
# bond_distances = numpy.linalg.norm(all_bonds[:,0,:]-all_bonds[:,1,:], axis=1)
# bond_indices = numpy.where(bond_distances >= Lx/2)[0]
# filtered_bonds = all_bonds[bond_indices, :, :]
# print(bonds[bond_indices])
# c= numpy.isclose(numpy.fabs(coordinates[:,0]),Lx/2)
# coord_rem_id=numpy.where(c==True)[0]
# coordinates = numpy.delete(coordinates,coord_rem_id,axis=0)
#
# a=numpy.isclose(numpy.fabs(pos[:,0]),Lx/2)
# rem_id=numpy.where(a==True)[0]
#
# rem_bnds = rem_id.reshape(-1,2)
# rem_bndid = numpy.unique(numpy.where(numpy.isin(bonds,rem_bnds)==True)[0])
# bonds = numpy.delete(bonds,rem_bndid,axis=0)
# bondid = numpy.delete(bondid,rem_bndid,axis=0)
#
# typearr=numpy.delete(typearr,rem_id)
# orientation=numpy.delete(orientation,rem_id,axis=0)
# Lxpos = pos[rem_id]
# pos=numpy.delete(pos,rem_id,axis=0)
# idpos = numpy.arange(0,len(pos))
# moment_inertia=numpy.delete(moment_inertia,rem_id,axis=0)
# diameter=numpy.delete(diameter,rem_id)
# mass=numpy.delete(mass,rem_id)
# bonds =idpos.reshape(-1,2)
#
# print(len(pos),len(bonds))
N = len(pos)

dihedral = numpy.empty((0,4),dtype=int)
dihedraltypes = numpy.empty(0,dtype=int)
num_body = len(numpy.where(typearr==1)[0])+len(numpy.where(typearr==0)[0])

constraint_pair1 = numpy.empty((0,2),dtype=int)
for i in range(0,num_body-1,2):
    x1 = 2*i+num_body
    x2 = i
    x3 = i+1
    x4 = 2*(i+1)+num_body

    x5 = 2*i+num_body+1
    x6 = i
    x7 = i+1
    x8 = 2*(i+1)+num_body+1

    dihedral = numpy.append(dihedral,[[x1,x2,x3,x4]],axis=0)
    # dihedral = numpy.append(dihedral,[[x5,x6,x7,x8]],axis=0)
    # dihedraltypes = numpy.append(dihedraltypes,0)
    dihedraltypes = numpy.append(dihedraltypes,0)


    constraint_pair1 = numpy.append(constraint_pair1,[[x2,x3]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x2,x4]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x2,x8]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x3,x1]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x3,x5]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x1,x4]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x5,x8]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x5,x4]],axis=0)
    constraint_pair1 = numpy.append(constraint_pair1,[[x1,x8]],axis=0)
#for i in range(len(constraint_pair1)):
#    print(constraint_pair1[i])
#exit()
print(constraint_pair1)
print(num_body)
#exit()
constraint_type = numpy.zeros(len(constraint_pair1))
############################################################
#### Creating snapshot to read the system ############
#################################################
type_= ['R','X','A','C','B','D','E']
snap = gsd.hoomd.Frame()
snap.particles.N = N
snap.particles.position = pos[0:N]

snap.particles.types = type_
snap.particles.typeid = typearr[:]
snap.dihedrals.types = ['ditype']
snap.bonds.types= ['R-X','poly','wall','wall-poly']
snap.angles.types=['A-R-A','A-R-X','pangles']
snap.particles.diameter = diameter[:]
snap.particles.orientation = orientation[:]
snap.particles.moment_inertia = moment_inertia[:]
snap.configuration.box = [Lx, Ly,Lz_1, 0, 0, 0]
#snap.pairs.N = len(constraint_pair1)
#snap.pairs.typeid = constraint_type
snap.pairs.types = ['const']
#snap.pairs.group = constraint_pair1

###############################################################################
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

wca.params[('R','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('C','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('B','D')] = {'sigma':sigma,'epsilon': 0}
wca.params[('D','D')] = {'sigma':sigma,'epsilon': eps}

wca.params[('R','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('C','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('B','E')] = {'sigma':sigma,'epsilon': 0}
wca.params[('D','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('E','E')] = {'sigma':sigma,'epsilon': eps}


LJ = hoomd.md.pair.LJ(nlist =ntree,default_r_cut = 2.5*sigma*math.sqrt(2))
LJ.params[('X','A')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('X','R')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('X','C')] = {'sigma':sigma,'epsilon':0}
LJ.params[('X','B')] = {'sigma':sigma,'epsilon': 0}


LJ.params[('C','A')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('C','B')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('A','B')] = {'sigma':sigma,'epsilon': 0}


LJ.params[('R','C')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('R','A')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('R','B')] = {'sigma':sigma,'epsilon': 0}


LJ.params[('R','R')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('X','X')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('A','A')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('C','C')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('B','B')] = {'sigma':sigma,'epsilon': 0}

LJ.params[('R','D')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('X','D')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('A','D')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('C','D')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('B','D')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
#LJ.params[('B','D')] = {'sigma':sigma,'epsilon': eps}
LJ.params[('D','D')] = {'sigma':sigma,'epsilon': 0}


LJ.params[('R','E')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('X','E')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('A','E')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('C','E')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('B','E')] = {'sigma':math.sqrt(2)*sigma,'epsilon': eps}
#LJ.params[('B','D')] = {'sigma':sigma,'epsilon': eps}
LJ.params[('D','E')] = {'sigma':sigma,'epsilon': 0}
LJ.params[('E','E')] = {'sigma':sigma,'epsilon': 0}


################################################################################
#########  Dihedral potential #################################################
#############################################################################3#
#di_har = hoomd.md.dihedral.Harmonic()
di_har = hoomd.md.dihedral.Periodic()

di_har.params['ditype'] = dict(k =100,d=-1,n=1,phi0 = 0)

################################################################################
####### FENEWCA ################################################################
################################################################################
fenewca = hoomd.md.bond.FENEWCA()
fenewca.params['R-X'] = dict(k=200,r0=r0,epsilon=eps,sigma=sigma,delta=0)
#fenewca.params['R-X'] = dict(k=k,r0=r0,epsilon=eps,sigma=sigma,delta=delta)
fenewca.params['poly'] = dict(k=200,r0=r0,epsilon=eps,sigma=sigma,delta=0)
fenewca.params['wall'] =  dict(k=0,r0=rm,epsilon=0,sigma=sigma,delta=0)
fenewca.params['wall-poly'] =  dict(k=0,r0=rm,epsilon=0,sigma=sigma,delta=0)

Harmonic = hoomd.md.bond.Harmonic()

Harmonic.params['R-X'] =  dict(k=k1,r0=rm)
Harmonic.params['poly'] =  dict(k=k,r0=rm)
Harmonic.params['wall'] =  dict(k=1000,r0=rm)
#Harmonic.params['wall'] =  dict(k=1000,r0=math.sqrt(2))
#Harmonic.params['wall-poly'] =  dict(k=1000,r0=rm)
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


################################################################################
##### Adding forces to calculate during integration
################################################################################
integrator.forces.append(wca)
#integrator.forces.append(fenewca)
integrator.forces.append(harmonic)
integrator.forces.append(Harmonic)
integrator.forces.append(di_har)
#integrator.forces.append(Constraint)


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
################################################################################
############### bonds between central and constituents, angle potential #######
###############################################################################
for i in range(2*len(coordinates)):
    start = 2*len(coordinates)+i*2
    index2 = start+0
    index3 = start+1
    bonds = numpy.append(bonds,[[i,index2]],axis=0)
    bondid = numpy.append(bondid,1)
    bonds = numpy.append(bonds,[[i,index3]],axis=0)
    bondid = numpy.append(bondid,1)




for i in range(0,2*len(coordinates)-1,2):
    start = 2*len(coordinates)+i*2
    index2 = start+0
    index3 = start+1
    start1 =  2*len(coordinates)+(i+1)*2
    index12 =start1+0
    index13 =start1+1
    angles = numpy.append(angles,[[i,index2,index3]],axis=0)

    angleid = numpy.append(angleid,0)
    angles = numpy.append(angles,[[i+1,index12,index13]],axis=0)
    angleid = numpy.append(angleid,0)
    angles = numpy.append(angles,[[i+1,i,index2]],axis=0)
    angleid = numpy.append(angleid,1)
    angles = numpy.append(angles,[[i+1,i,index3]],axis=0)
    angleid = numpy.append(angleid,1)
    angles = numpy.append(angles,[[i,i+1,index13]],axis=0)
    angleid = numpy.append(angleid,1)
    angles = numpy.append(angles,[[i,i+1,index12]],axis=0)
    angleid = numpy.append(angleid,1)

################################################################################
############ Simulation context ####################################
################################################################################
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu,seed=seed3)
sim.create_state_from_snapshot(snap)
###############################################################################
### Create constituent particles
################################################################################
rigid.create_bodies(sim.state)
integrator.rigid = rigid
###############################################################################
### creating a snapshot of the current state to add particles later ############
################################################################################

snap1 = sim.state.get_snapshot()
position = numpy.copy(snap1.particles.position[:])

typeid = numpy.copy(snap1.particles.typeid[:])
bonds1 = numpy.copy(bonds)

################################################################################
###### Defining freud boxes for adding polymer beads
##############################################################################
print("Lx",Lx,"Ly",Ly,"Lz",Lz)
box = freud.box.Box(Lx =Lx, Ly= Ly, Lz = Lz)  #periodic
box4 = freud.box.Box(Lx =10*Lx, Ly= Ly, Lz =Lz)  #aperiodic in Lx
box2 = freud.box.Box(Lx =10*Lx, Ly= 10*Ly,Lz = 10*Lz) #non-periodic
box1 = freud.box.Box(Lx =Lx, Ly= Ly,Lz = Lz) # this is for neighbor finding
coordinates1 = numpy.copy(coordinates)

aq = freud.locality.AABBQuery(box4,coordinates1)
query_result = aq.query(coordinates1, dict(r_max=rmax+.25,exclude_ii=True))
nlist = query_result.toNeighborList()
################################################################################
###### End points to put the polymer beads #####################################
################################################################################
store_bonds=nlist
store_bonds=numpy.sort(store_bonds,axis=1)
store_bonds=numpy.unique(store_bonds,axis=0)

store_bonds=numpy.sort(store_bonds,axis=1)
store_bonds=numpy.unique(store_bonds,axis=0)
store_bonds1=numpy.empty((0,2),dtype=int)

for x in store_bonds:
    thisid=numpy.empty(0,dtype=int)
    otherid=numpy.empty(0,dtype=int)

    id1=2*x[0]+0
    id2=2*x[0]+1
    id3=2*x[1]+0
    id4=2*x[1]+1

    thisid1=numpy.where(bonds1==id1)[0]
    thisid11=numpy.where(bonds1==id1)[1]
    thisid2=numpy.where(bonds1==id2)[0]
    thisid21=numpy.where(bonds1==id2)[1]

    otherid1=numpy.where(bonds1==id3)[0]
    otherid11=numpy.where(bonds1==id3)[1]
    otherid2=numpy.where(bonds1==id4)[0]
    otherid21=numpy.where(bonds1==id4)[1]



    for i in range(1,3):

        thisid = numpy.append(thisid,[bonds1[thisid1[i],1-thisid11[i]]],axis=0)
        thisid = numpy.append(thisid,[bonds1[thisid2[i],1-thisid21[i]]],axis=0)

        otherid = numpy.append(otherid,[bonds1[otherid1[i],1-otherid11[i]]],axis=0)
        otherid = numpy.append(otherid,[bonds1[otherid2[i],1-otherid21[i]]],axis=0)

    distance = numpy.empty(0,dtype=float)
    for i in range(len(thisid)):
        for j in range(len(otherid)):
            dist=numpy.linalg.norm(box1.wrap(position[thisid[i]]-position[otherid[j]]))
            distance=numpy.append(distance,dist)
    min_dist=numpy.min(distance)
    ids=numpy.where(distance==min_dist)[0][0]
    x_id = ids%4
    baseid = int((ids-x_id)/4)

    store_bonds1 = numpy.append(store_bonds1,[[thisid[baseid],otherid[x_id]]],axis=0)



store_bonds1 = numpy.sort(store_bonds1,axis =1 )
store_bonds1 = numpy.unique(store_bonds1,axis=0)
###############################################################################
#### Adding polymer beads #####################################################
###############################################################################
dia2=1.0
temp=len(bonds)
for x in store_bonds1:
    dist = numpy.linalg.norm(box1.wrap(position[x[1]]-position[x[0]]))
    dr = box1.wrap(position[x[1]]-position[x[0]])

    division = int(dist/dia2)
    const = dr/division
    b0 = dist/division
    no_beads = division-1
    bonds=numpy.append(bonds,[[x[0],len(position)]],axis=0)
    bondid = numpy.append(bondid,2)

    for k in range(1,no_beads+1):
        pos = box1.wrap(k*const+position[x[0]])
        if k == no_beads:
            bonds = numpy.append(bonds,[[len(position),x[1]]],axis=0)
            bondid = numpy.append(bondid,2)


        else:
            bonds=numpy.append(bonds,[[len(position),len(position)+1]],axis=0)
            bondid = numpy.append(bondid,2)
            if k<=no_beads-2:
                angles = numpy.append(angles,[[len(position),len(position)+1,len(position)+2]],axis=0)
                angleid = numpy.append(angleid,2,axis=None)


        position=numpy.append(position,[pos],axis=0)
        typeid = numpy.append(typeid,4)


################################################################################

#for x in s:
#    print(seltotids[x[0]],x[1])
#print(s1)
#exit()
#print(position.shape)
#exit()



#print(len(zplus),len(yplus))
# numpy.repeat()

################################################################################
#### Adding particles and bonds to snap and giving back to sim state
################################################################################
# max_x=numpy.max(position[:,0])
# min_x=numpy.min(position[:,0])
# cfx = (max_x+min_x)/2
# nLxby2 = (max_x-min_x)/2
# nLx = 2*nLxby2+40
# position[:,0]=position[:,0]-cfx
#snap1.configuration.box = [nLx, Ly,Lz_1, 0, 0, 0]

# exit()
snap1.particles.N = len(position)

snap1.particles.position[:len(position)] = position[:]
#snap1.particles.position[len(position):] = Lxpos

snap1.particles.typeid[:len(typeid)] =typeid[:]
#snap1.particles.typeid[len(typeid):] = 5
snap1.bonds.N = len(bonds)
snap1.angles.N = len(angles)
snap1.bonds.group[:] = bonds[:]
snap1.bonds.typeid[:len(bondid)] = bondid[:]

snap1.angles.typeid[:] = angleid[:]
snap1.angles.group[:] = angles[:]
print(len(dihedral))
snap1.dihedrals.N = len(dihedral)
snap1.dihedrals.group[:] = dihedral
snap1.dihedrals.typeid[:] = dihedraltypes

snap1.pairs.N = len(constraint_pair1)
snap1.pairs.typeid[:] = constraint_type
snap1.pairs.types[:] = ['const']
snap1.pairs.group[:] = constraint_pair1

sim.state.set_snapshot(snap1)
###############################################################################
### Filtering integration in hoomd
###############################################################################
rigid_centers_and_free_filter = hoomd.filter.Rigid(("center", "free"))
nvt = hoomd.md.methods.Langevin(kT = kbT, filter = rigid_centers_and_free_filter)
mttk = hoomd.md.methods.thermostats.MTTK(kT=1.0,tau=dt*100)
npt = hoomd.md.methods.ConstantPressure(filter=rigid_centers_and_free_filter , thermostat=mttk,
tauS = dt*1000, S=[0.001,0.001,0.001,0,0,0], couple="xyz",box_dof=[True,True,True,False,False,False],gamma=dof)

integrator.methods.append(npt)
sim.operations.integrator = integrator
filename="EquilibriumSlab_stiff"+str(k_stiff)+"_PBCyzHar_FeringGel"+"_nbeads_"+str(int(no_beads))+"_kT_"+str(kbT)+".gsd"

gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(30000000+400000)), filename=filename,mode='wb',dynamic=['momentum','property'])
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1), filename=filename,mode='wb',dynamic=['momentum','property'])
sim.operations += gsd_oper


thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=rigid_centers_and_free_filter)

sim.operations.computes.append(thermodynamic_properties)

sim.state.thermalize_particle_momenta(filter=rigid_centers_and_free_filter, kT=kbT)
logg1= hoomd.logging.Logger(categories=['scalar', 'string','scalar'])
logg1.add(sim, quantities=['timestep', 'tps'])
logg1.add(thermodynamic_properties,quantities=['kinetic_temperature'])

status = Status(sim)
logg1[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(500000),
                          logger=logg1)
sim.operations.writers.append(table)
sim.run(30000000)

print("Finished zero pressure compression")
sys.stdout.flush()
integrator.methods.remove(npt)
integrator.methods.append(nvt)
integrator.forces.remove(Harmonic)
integrator.forces.append(fenewca)

ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=200000)
initial_box = sim.state.box
print(initial_box)
final_box = hoomd.Box(107,53,26,0,0,0)
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box,
                                    box2=final_box,
                                    variant=ramp,
                                    trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
sim.run(200000)
sim.operations.updaters.remove(box_resize)
sim.run(200000)
