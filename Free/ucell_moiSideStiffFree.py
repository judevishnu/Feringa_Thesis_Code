import hoomd
import numpy
import math
import random
import time
import freud
import gsd.hoomd
import sys
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime


#############################
seed3 = random.randint(1,65535) ##Random seed
print(seed3)
nlat=[8,4,2]
#nlat=[8,4,2]

#nlat=[8,4,2]
dof=0
a = 1
alat = float(sys.argv[1])
nsidebead = int(sys.argv[2])
k_stiff = float(sys.argv[3])
rcut = math.pow(2/a,1/6.)
Lx=(alat)*(nlat[0])
Ly=(alat)*nlat[1]
Lz=(alat)*(nlat[2])
Lz_1=(alat+1.2)*(nlat[2])

Lxby2=Lx/2.
Lyby2=Ly/2.
Lzby2=Lz/2.
rmax = (math.sqrt(3)/4)*alat
print(rmax)
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
r0=1.5
delta=0.0
rm=0.957
box9=freud.box.Box(1,1,1,0,0)
print(box9)
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
#############Diamond lattice#####################
def initialize_coordinates():
    basis =[[0.0,0.0,0.0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0,0.5,0.5],
            [0.25,0.25,0.25],[0.75,0.75,0.25],[0.75,0.25,0.75],[0.25,0.75,0.75]]
    for i in range(0,nlat[0]):
        for j in range(0,nlat[1]):
            for k in range(0,nlat[2]):
                for m in range(8):
                    x = (basis[m][0]+i)*alat
                    y = (basis[m][1]+j)*alat
                    z = (basis[m][2]+k)*alat
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

#################################################
###Setting up cross-linkers which are rigid bodies
#################################################
# initialize_coordinates()
# coordinates=numpy.asarray(coords)
uc=freud.data.UnitCell(box9, basis_positions=[[0.0,0.0,0.0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0,0.5,0.5],
        [0.25,0.25,0.25],[0.75,0.75,0.25],[0.75,0.25,0.75],[0.25,0.75,0.75]])
#n_repeats = (8,4,2)
n_repeats = (nlat[0],nlat[1],nlat[2])
#n_repeats = (1,1,1)
system = uc.generate_system(n_repeats, scale=alat)
# Lx=system[0].Lx
# Ly=system[0].Ly
# Lz=system[0].Lz
# Lxby2=Lx/2.
# Lyby2=Ly/2.
# Lzby2=Lz/2.
coordinates=system[1]
print(coordinates)
#with gsd.hoomd.open(name = 't.gsd',mode='wb+') as f:
#    s = gsd.hoomd.Snapshot()

#    s.particles.N=len(coordinates)
#    s.particles.position=coordinates
#    s.configuration.box=[system[0].Lx,system[0].Ly,system[0].Lz,0,0,0]
#    f.append(s)
# exit()

############Central particls position and moment of inertia#############
#I = ((2.0/5.0)*(0.5*0.5)+2*(0.5)**2)/math.sqrt(3) ##moment of inertia of rigid bodies
I = Calculate_MOI() ##moment of inertia of rigid bodies
I = numpy.array([0,0,I[2]])

pos = numpy.empty((0,3),dtype=float)
typearr = numpy.empty(0,dtype=int)
orientation=numpy.empty((0,4),dtype=float)
mass=numpy.empty(0,dtype=float)
moment_inertia=numpy.empty((0,3),dtype=float)
bonds=numpy.empty((0,2),dtype=int)
#bonds1=numpy.empty((0,2),dtype=int)
angles =numpy.empty((0,3),dtype=int)
bondid = numpy.empty(0,dtype=int)
angleid = numpy.empty(0,dtype=int)
diameter=numpy.empty(0,dtype=float)
positive_tau=numpy.empty(0,dtype=int)
body=numpy.empty(0,dtype=int)
print(len(coordinates))
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
N = len(pos)
################################################################################
#############           Dihedral ###############################################
################################################################################
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
    #dihedral = numpy.append(dihedral,[[x5,x6,x7,x8]],axis=0)
    #dihedraltypes = numpy.append(dihedraltypes,0)
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
print("num dihedral:",len(dihedral))
constraint_type = numpy.zeros(len(constraint_pair1))
############################################################
#### Creating snapshot to read the system ############
#################################################
type_= ['R','X','A','C','B','S']
snap = gsd.hoomd.Frame()
snap.particles.N = N
snap.particles.position = pos[0:N]

snap.particles.types = type_
snap.particles.typeid = typearr[:]
snap.dihedrals.types = ['ditype']
snap.bonds.types= ['R-X','poly']
snap.angles.types=['A-R-A','A-R-X','pangles']
snap.particles.diameter = diameter[:]
snap.particles.orientation = orientation[:]
snap.particles.moment_inertia = moment_inertia[:]
snap.configuration.box = [1.*Lx, 1.*Ly, 1.*Lz_1, 0, 0, 0]
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

wca.params[('S','R')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','X')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','A')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','C')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','B')] = {'sigma':sigma,'epsilon': eps}
wca.params[('S','S')] = {'sigma':sigma,'epsilon': eps}

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
fenewca.params['R-X'] = dict(k=k,r0=0.85*r0,epsilon=eps,sigma=0.85*sigma,delta=delta)
fenewca.params['poly'] = dict(k=k,r0=r0,epsilon=eps,sigma=sigma,delta=delta)

Harmonic = hoomd.md.bond.Harmonic()

Harmonic.params['R-X'] =  dict(k=500,r0=rm)
Harmonic.params['poly'] =  dict(k=500,r0=rm)



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
################################################################################
##### Adding forces to calculate during integration
################################################################################
integrator.forces.append(wca)
# integrator.forces.append(fenewca)
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
box = freud.box.Box(Lx =Lx, Ly= Ly, Lz = Lz_1)  #periodic
box4 = freud.box.Box(Lx =Lx, Ly= Ly, Lz =10*Lz)  #aperiodic in Lz
box2 = freud.box.Box(Lx =10*Lx, Ly= 10*Ly,Lz = 10*Lz) #non-periodic
#box1 = freud.box.Box(Lx =1.2*Lx, Ly= 1.2*Ly,Lz = 1.2*Lz) # this is for neighbor finding
box1 = freud.box.Box(Lx =Lx, Ly= Ly,Lz = Lz) # this is for neighbor finding
coordinates1 = numpy.copy(coordinates)

aq = freud.locality.AABBQuery(box2,coordinates1)
query_result = aq.query(coordinates1, dict(r_max=rmax+0.25,exclude_ii=True))
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
print(len(store_bonds))
#exit()

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
counter_poly = 0
for x in store_bonds1:
    dist = numpy.linalg.norm(box.wrap(position[x[1]]-position[x[0]]))
    dr = box.wrap(position[x[1]]-position[x[0]])

    division = int(dist/dia2)
    const = dr/division
    b0 = dist/division
    no_beads = division-1
    bonds=numpy.append(bonds,[[x[0],len(position)]],axis=0)
    bondid = numpy.append(bondid,1)

    for k in range(1,no_beads+1):
        pos = box.wrap(k*const+position[x[0]])
        if k == no_beads:
            bonds = numpy.append(bonds,[[len(position),x[1]]],axis=0)
            bondid = numpy.append(bondid,1)



        else:
            bonds=numpy.append(bonds,[[len(position),len(position)+1]],axis=0)
            bondid = numpy.append(bondid,1)
            if k<=no_beads-2:
                angles = numpy.append(angles,[[len(position),len(position)+1,len(position)+2]],axis=0)
                angleid = numpy.append(angleid,2,axis=None)


        position=numpy.append(position,[pos],axis=0)
        typeid = numpy.append(typeid,4)
    counter_poly = 0
################################################################################
# for k in range(0,counter_poly):
#     for n in range(0,no_beads)
#         angles = numpy.append(angles,[[k,k+1,k+2]],axis=0)
#         angleid = numpy.append(angleid,1)
# exit()
################################################################################
############# Side chain intialization ########################################
################################################################################
theta = 1.92
Rx = numpy.asarray([[1,0,0],[0,numpy.cos(theta),-numpy.sin(theta)],[0,numpy.sin(theta),numpy.cos(theta)]])
temp_bonds = numpy.copy(store_bonds1)
temp_bonds =numpy.sort(store_bonds1,axis=1)
bond_vectors = numpy.empty((0,3),dtype=float)
other_vectors = numpy.empty((0,3),dtype=float)
bond_forside = numpy.empty((0,2),dtype=int)
#print("I am before this",len(temp_bonds),len(typeid))
for i in range(len(temp_bonds)):

    id1=temp_bonds[i,0]
    id2=temp_bonds[i,1]
    #print(typeid[id1],typeid[id2])
    if((typeid[id2]==2 and typeid[id1]==3)or(typeid[id1]==2 and typeid[id2]==3)):
        #print(i)
        bond_forside=numpy.append(bond_forside,[numpy.asarray([id1,id2])],axis=0)
        dist = numpy.linalg.norm(box.wrap(position[id1]-position[id2]))
        dr = box.wrap(position[id1]-position[id2])
        bond_vectors= numpy.append(bond_vectors,[numpy.asarray([dr[0]/dist,dr[1]/dist,dr[2]/dist])],axis=0)
        othervectemp = numpy.dot(numpy.asarray([dr[0]/dist,dr[1]/dist,dr[2]/dist]),Rx)
        otherdist= numpy.linalg.norm(box.wrap(othervectemp))
        other_vectors= numpy.append(other_vectors,[numpy.asarray([othervectemp[0]/otherdist,othervectemp[1]/otherdist,othervectemp[2]/otherdist])],axis=0)

print("I am before this 1")
#if nsidebeads==0:

for i in range(len(bond_forside)):
    x=bond_forside[i]
    dist = numpy.linalg.norm(box.wrap(position[x[1]]-position[x[0]]))
    dr = box.wrap(position[x[1]]-position[x[0]])
    #print(dist)
    division = int(dist/dia2)
    const = dr/division
    b0 = dist/division
    no_beads = division-1

    if int(nsidebead)==0:
        break
    #print(no_beads)
    #bonds=numpy.append(bonds,[[x[0],len(position)]],axis=0)
    #system.bonds.add('polymer',x[0],len(position))
    for k in range(1,no_beads+1):
        pos = box.wrap(k*const+position[x[0]])
        #pos = k*const+position[x[0]]
        basebeadid=numpy.where((position[:,0]==pos[0])&(position[:,1]==pos[1])&(position[:,2]==pos[2]))[0][0]



        posit = box.wrap(other_vectors[i]*1+position[basebeadid])
        #posit = other_vectors[i]*1+position[basebeadid]

        bonds = numpy.append(bonds,[[basebeadid,len(position)]],axis=0)
        bondid = numpy.append(bondid,1)
        position=numpy.append(position,[posit],axis=0)
        typeid = numpy.append(typeid,5)
        # system.particles.add('D')
        # system.particles[len(position)-1].position=(posit[0],posit[1],posit[2])

        for m in range(2,nsidebead+1):

                posit = box.wrap(other_vectors[i]*m+position[basebeadid])
                #posit = other_vectors[i]*m+position[basebeadid]
                bonds = numpy.append(bonds,[[len(position)-1,len(position)]],axis=0)
                bondid = numpy.append(bondid,1)
                position=numpy.append(position,[posit],axis=0)
                typeid = numpy.append(typeid,5)
                # system.particles.add('D')
                # system.particles[len(position)-1].position=(posit[0],posit[1],posit[2])
print("I am starting simulations")
print("Len angles",len(angles),len(angleid))
################################################################################
#### Adding particles and bonds to snap and giving back to sim state
################################################################################

snap1.particles.N = len(position)
snap1.particles.position[:] = position[:]
snap1.particles.typeid[:] =typeid[:]
snap1.bonds.N = len(bonds)
snap1.angles.N = len(angles)
snap1.bonds.group[:] = bonds[:]
snap1.bonds.typeid[:] = bondid[:]
snap1.angles.typeid[:] = angleid[:]
snap1.angles.group[:] = angles[:]

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
#nvt = hoomd.md.methods.NVT(kT = kbT, tau = 1., filter = rigid_centers_and_free_filter)
nvt = hoomd.md.methods.Langevin(kT = kbT, filter = rigid_centers_and_free_filter)
mttk = hoomd.md.methods.thermostats.MTTK(kT=1.0,tau=dt*100)
npt = hoomd.md.methods.ConstantPressure(filter=rigid_centers_and_free_filter , thermostat=mttk,
tauS = dt*1000, S=[0.001,0.001,0.001,0,0,0], couple="yz",box_dof=[True,True,True,False,False,False],gamma=dof)

sim.operations.integrator = integrator
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter= rigid_centers_and_free_filter)
logger = hoomd.logging.Logger(['particle','constraint'])
logg1= hoomd.logging.Logger(categories=['scalar', 'string','scalar'])
logg1.add(sim, quantities=['timestep', 'tps'])
logg1.add(thermodynamic_properties,quantities=['kinetic_temperature'])

status = Status(sim)
logg1[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(100000),
                          logger=logg1)
sim.operations.writers.append(table)

#filename="initialSidechain"+"_ucellHarMOI_FeringGel"+"_nbeads_"+str(no_beads)+"_nside_"+str(nsidebead)+"_kT_"+str(kbT)+".gsd"
#filename="initiallattice1.gsd"

#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1020000), filename=filename,mode='wb',dynamic=['momentum','property'])

filename1="initialSidechain_stiff"+str(k_stiff)+"_nonPBCFree_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#filename="initiallattice1.gsd"

gsd_oper1 = hoomd.write.GSD(trigger=hoomd.trigger.On(3000000+200000), filename=filename1,mode='wb',dynamic=['momentum','property'])


#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1000), filename=filename,mode='wb',dynamic=['momentum','property'])

#sim.operations += gsd_oper
sim.operations += gsd_oper1

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=rigid_centers_and_free_filter)

sim.operations.computes.append(thermodynamic_properties)

sim.state.thermalize_particle_momenta(filter=rigid_centers_and_free_filter, kT=kbT)

integrator.methods.append(npt)
sim.run(3000000)



integrator.forces.remove(Harmonic)
integrator.forces.append(fenewca)

#integrator.methods.append(nvt)
integrator.methods.remove(npt)
integrator.methods.append(nvt)

sim.run(200000)
