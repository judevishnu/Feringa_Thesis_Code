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
dof =float(sys.argv[11])
cforce =float(sys.argv[10])
k = int(sys.argv[9])
#k1=5000
k1=int(sys.argv[8])
kang = 1000
r0=1.5*sigma
delta=0.0
kdihed = int(sys.argv[6])
no_beads = int(sys.argv[1])
Torque = float(sys.argv[2])
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

class anglelog():

    def __init__(self, trapobj):
        self.obj = trapobj

    @property
    def angles(self):
        return numpy.asarray(self.obj.get_angles('ditype'))
       

################################################################################
################## Class for calculating angles ################################
################################################################################
class HDF5Writer_cpu(hoomd.custom.Action):
    def __init__(self, filename, mode,postau,negtau,m_ids,box1,freq,freq1):
        self.filename = filename
        if mode not in {'w', 'w-', 'x', 'a', 'r+'}:
            raise ValueError("mode must be writtable")
        self.file = h5py.File(filename, mode)
        self.write_metadata()
        self.postau = postau
        self.negtau = negtau
        self.box1 = box1
        self.m_ids = m_ids

        self.diff = numpy.zeros((len(self.postau),1),dtype=float)
        self.old_angle =numpy.zeros((len(self.postau),1),dtype=float)
        self.period = freq
        self.period1 = freq1

        frames = list(self.file.keys())
        if frames:
            self._cur_frame = max(map(int, frames)) + 1
        else:
            self._cur_frame = 1

    def write_metadata(self):
        """Write the file metadata that defines the type of hdf5 file"""
        if 'app' in self.file.attrs:
            if self.file.attrs.app != 'hoomd-v3':
                raise RuntimeError(
                    'HDF5 file metadata "app" is not "hoomd-v3".')
        else:
            self.file.attrs.app = 'hoomd-v3'

        if 'version' not in self.file.attrs:
            self.file.attrs.version = '1.0'

    def getDifference(self):
        self.diff = numpy.fmod(self.diff,2*numpy.pi)
        self.diff = numpy.where(self.diff<-numpy.pi,self.diff+numpy.pi*2,self.diff)
        self.diff = numpy.where(self.diff>=numpy.pi,self.diff-numpy.pi*2,self.diff)

    def act(self, timestep):
        """Write out a new frame to the trajectory."""

        with sim.state.gpu_local_snapshot as data:

            new_frame = self.file.create_group(str(self._cur_frame))

            particles = getattr(data, 'particles')
            Boxsim = getattr(data,'global_box')
            self.box1 = freud.box.Box(Boxsim.Lx,Boxsim.Ly,Boxsim.Lz)
            pos = cp.asnumpy(getattr(particles,'position'))
            image = cp.asnumpy(getattr(particles,'image'))
            typeid = cp.asnumpy(getattr(particles,'typeid'))
            rtag = cp.asnumpy(getattr(particles,'rtag'))

            # num_body = len(self.postau)*2 just for this sim

            position=self.box1.unwrap(pos,image)

            central_pos1 = numpy.take(pos,rtag[self.postau],axis=0)
            central_pos2 = numpy.take(pos,rtag[self.negtau],axis=0)
            side_particle1 = rtag[self.postau*2+num_body]
            side_particle2 = rtag[self.negtau*2+num_body]
            side_particle12 = rtag[self.postau*2+num_body+1]
            side_particle22 = rtag[self.negtau*2+num_body+1]
            #tmp_mids = (self.postau+self.negtau-1)/4

            #print(self.postau)
            #print(self.negtau)
            #print(len(tmp_mids[self.m_ids]))
            #exit()
            side_pos1 = numpy.take(pos,side_particle1,axis=0)
            side_pos2 = numpy.take(pos,side_particle2,axis=0)

            side_pos12 = numpy.take(pos,side_particle12,axis=0)
            side_pos22 = numpy.take(pos,side_particle22,axis=0)
            new_vec1 =   self.box1.wrap(central_pos1-side_pos1)

            #print(new_vec1)
            new_vec2 =  self.box1.wrap(central_pos2-central_pos1)
            #new_vec3 = (central_pos1-central_pos2)
            new_vec4 =  self.box1.wrap(side_pos2-central_pos2)
            cross1 = numpy.cross(new_vec1,new_vec2)
            cross2= numpy.cross(new_vec2,new_vec4)
            # print(numpy.linalg.norm(new_vec1),numpy.linalg.norm(new_vec4))
            #print(numpy.dot(new_vec4.reshape(-1),new_vec1.reshape(-1)))
            #numpy.repeat(numpy.linalg.norm(new_vec4),3)
            #DOT = numpy.sum(new_vec1[:]*new_vec4,axis=1)
            # print(DOT)

            #CROSS = numpy.cross(new_vec1,new_vec4)
            # print(CROSS)
            tmp_dphi_i= numpy.arctan2(numpy.sum(new_vec2*(numpy.cross(cross1,cross2)),axis=1),numpy.linalg.norm(new_vec2,axis=1)*(numpy.sum(cross1*cross2,axis=1)))

            # print(tmp_dphi_i)
            tmp_dphi_i = tmp_dphi_i.reshape((len(self.postau),1))



            if self._cur_frame %self.period1 ==1:
                self.old_angle = tmp_dphi_i
                print("Frame :",self._cur_frame,"Remainder :",self._cur_frame %self.period1)



            self.diff = tmp_dphi_i - self.old_angle
            self.getDifference()
            #diff = numpy.take(self.diff,self.m_ids,axis=0)
            diff = self.diff
            # print(diff)

            dataset = new_frame.create_dataset('angles',diff.shape,
                                            dtype='f8')
            dataset[:] = diff
            new_frame.attrs['timestep'] = timestep


            self.old_angle = tmp_dphi_i
            self._cur_frame += 1


    def __del__(self):
            self.file.close()
################################################################################
#################   Class Updater to update constraint distance ################
################################################################################
class UpdateConstraint_distance():
    def __init__(self,sim,constraint_pair):
        self._state = sim.state
        self._id1 = constraint_pair[:,0]
        self._id2 = constraint_pair[:,1]
        self._dist = numpy.zeros(len(constraint_pair)) #dummy values


    def __call__(self):
        #data = self._state.get_snapshot()
        with self._state.gpu_local_snapshot as data:
            particles = getattr(data,'particles')
            pos = cp.asnumpy(getattr(particles,'position'))
            rtag =cp.asnumpy(getattr(particles,'rtag'))
            rtag1 =numpy.take(rtag,self._id1,axis=0)
            rtag2 =numpy.take(rtag,self._id2,axis=0)
            #image = getattr(particles,'image')
            temp_pos1=numpy.take(pos,rtag1,axis=0)
            temp_pos2=numpy.take(pos,rtag2,axis=0)
            self._dist=numpy.linalg.norm(box1.wrap(temp_pos1 - temp_pos2),axis=1)
            #print(self._dist)

        return self._dist

################################################################################
#################Class to   Undo constraint distance ###########################
################################################################################
class UndoConstraint_distance():
    def __init__(self,Har,constraint_pair,bond_name):
        self._id1 = constraint_pair[:,0]
        self._id2 = constraint_pair[:,1]
        self.Har = Har
        self.bond_name = bond_name


    def __call__(self):
        for i in range(0,len(self.bond_name)):
            if i==0:
                self.Har.params[self.bond_name[i]] = dict(k=k,r0=r0)
            else :
                self.Har.params[self.bond_name[i]] = dict(k=0,r0=r0)

###############################################################################
############### UpdateConstraint_distance caller ###############################
###############################################################################
def call_constraint():
    global sim,Harmonic,constant_pair,bond_name
    s=UpdateConstraint_distance(sim,Harmonic,constraint_pair,bond_name)
    s()
################################################################################
################ undo constraint ##############################################
###############################################################################
def remove_forcesfromfire():
    global nve,nvtLang,fire,integrator,Har,Harmonic,wca,harmonic,nve,nvtLang,sim
    global h5_writer
    # fire.forces.remove(wca)
    # fire.forces.remove(harmonic)
    #integrator.forces.remove(Harmonic)
    # integrator.methods.remove(nve)


    #integrator.forces.append(Har)
    #integrator.forces.append(const_torque)
    # integrator.forces.append(Harmonic)

    integrator.methods.remove(nve)

    integrator.methods.append(nvtLang)




    #sim.operations.integrator = integrator
    #sim.operations.writers.remove(h5_writer)

def reset_integratorforces():
    global nve,nvtLang, integrator,Har,Harmonic,wca,harmonic,nve,nvtLang,sim
    integrator.forces.remove(Harmonic)
    integrator.forces.append(const_torque)
    integrator.forces.append(Har)

    integrator.methods.remove(nvtLang)
    integrator.methods.append(nve)
    #di_har.params['ditype'] = dict(k =100,d=-1,n=1,phi0 = 0)

    #integrator.forces.append(di_har)

    # sim.operations.integrator = None
    # sim.operations.integrator = integrator
    #sim.operations.writers.append(h5_writer)
################################################################################
##########functions for enabling and disbaling writing angle to h5file #########
################################################################################
def set_h5(i):
    global h5_writer,sim,inteval
    if (not i% interval):
        sim.operations.writers.append(h5_writer)


#def reset_h5(i):
#    global h5_writer,sim,interval
#    if (not i% interval):
#        sim.operations.writers.remove(h5_writer)
#        if i>400:
#            interval = 20
#        else:
#            if i>=300 and i<1000
#            else:
#                if i >=500 and i<1000:
#                    interval = 40
#                else:
#                    if i>= 1000 and i<4000:
#                        interval = 100
#                    else:
#                        if i>=4000 and i<9000:
#                            interval = 200


def reset_h5(i):
    global h5_writer,sim,interval
    if (not i% interval):
        sim.operations.writers.remove(h5_writer)
        if i>600:
            interval = 1
        else:
            if i>=700 and i<1000:
                interval = 20
            else:
                if i >=1000 and i<1500:
                    interval = 40
                else:
                    if i>= 1500 and i<4000:
                        interval = 100
                    else:
                        if i>=4000 and i<9000:
                            interval = 200


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
filename="/lustre/project/nhr-trr146/jvishnu/Feringa/Piston/zerolatpress/PulledFinalGel"+"_PBCyzHar"+str(sample)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(cforce)+"_kT_"+str(kbT)+".gsd"


#filename=".../../_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".gsd"

#filename="../../PulledFinalGel"+"_PBCyzHar"+str(sample)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(cforce)+"_kT_"+str(kbT)+".gsd"
#filename="../initial"+"_ucellHarsing_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#filename="initial"+"_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
traj = gsd.hoomd.open(name= filename,mode='r')
snap = gsd.hoomd.Frame()

bonds  = numpy.copy(traj[len(traj)-1].bonds.group)
bondtypeid  = numpy.copy(traj[len(traj)-1].bonds.typeid)
angles = numpy.copy(traj[len(traj)-1].angles.group)
dihedrals = numpy.copy(traj[len(traj)-1].dihedrals.group)
print(len(dihedrals))
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
#mono_ids = numpy.where(typeid==4)[0]
#print(mono_ids)
#bnd= numpy.empty((0,2),dtype=int)
#for x in mono_ids:
#    y=numpy.where(bonds1[:]==x)
#    bnd =numpy.append(bnd,[bonds1[y[0][0]]],axis=0)
#    bnd =numpy.append(bnd,[bonds1[y[0][1]]],axis=0)

    # print(x,bonds1[y[0][0]],bonds1[y[0][1]])
    # print(bonds1[y[0][0]],bonds1[y[1][1]])
#_,idx=numpy.unique(bnd,axis=0,return_index=True)
#bnd=bnd[numpy.sort(idx)]
# print(len(bnd),len(mono_ids))
#num_poly = int(len(mono_ids)/(int(no_beads)))
#mono_ids = mono_ids.reshape((num_poly,int(no_beads),1))
#bnd =bnd.reshape((num_poly,1+no_beads,2))
#bnd_1 = bnd[:,0::no_beads]

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
molfile="/lustre/project/nhr-trr146/jvishnu/Feringa/Piston/zerolatpress/Molecularids"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".txt"

#molfile="../Molecularids"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".txt"

arraymol = numpy.asarray((dihedrals[::2,1].T,dihedrals[::2,2].T)).T#numpy.loadtxt(molfile)

posttau_ids = arraymol[:,0].astype(int)
negttau_ids = arraymol[:,1].astype(int)
n_ids = numpy.copy(negttau_ids)
p_ids = numpy.copy(posttau_ids)
n_ids = numpy.sort(n_ids)
p_ids = numpy.sort(p_ids)

posttau_ids = cp.asarray(posttau_ids)
negttau_ids = cp.asarray(negttau_ids)

print(n_ids.shape,p_ids.shape)
#exit()

molfile="/lustre/project/nhr-trr146/jvishnu/Feringa/Piston/zerolatpress/Molecularids"+str(4+int(sample))+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".txt"

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
# constraint_pair = numpy.empty((0,2),dtype=int)
# pair0 = num_body + p_ids*2
# pair1 = num_body + n_ids*2+1
# pair2 = num_body + p_ids*2+1
# pair3 = num_body + n_ids*2
# constraint_name = []
# bond_name = ['R-X']
# temp = len(bonds)
# for i in range(len(pair0)):
#     constraint_pair = numpy.append(constraint_pair,[[pair0[i],pair1[i]]],axis=0)
#     constraint_name.append(str(i)+str(1))
#     bond_name.append(str(i)+str(1))
#
#     constraint_pair = numpy.append(constraint_pair,[[pair0[i],pair3[i]]],axis=0)
#     constraint_name.append(str(i)+str(2))
#     bond_name.append(str(i)+str(2))
#
#     constraint_pair = numpy.append(constraint_pair,[[pair2[i],pair3[i]]],axis=0)
#     constraint_name.append(str(i)+str(3))
#     bond_name.append(str(i)+str(3))
#
#     constraint_pair = numpy.append(constraint_pair,[[pair1[i],pair2[i]]],axis=0)
#     constraint_name.append(str(i)+str(4))
#     bond_name.append(str(i)+str(4))
#
# bonds = numpy.append(bonds,constraint_pair,axis=0)
# constraintype=numpy.arange(1,len(constraint_name)+1,1)
# bondtypeid = numpy.append(bondtypeid,constraintype)
################################################################################
#dicts ={}

#molecular_ids = numpy.empty(0,dtype=int)
#for i in range(0,num_body-1,2):
#    x1 = 2*i+num_body
#    x2 = 2*i + num_body+1
#    x3 = 2*(i+1) + num_body
#    x4 = 2*(i+1) + num_body+1
#    dicts[x1] = int(i/2)
#    dicts[x2] = int(i/2)
#    dicts[x3] = int(i/2)
#    dicts[x4] = int(i/2)
#for i in range(0,num_body-1,2):

#    x1 = 2*i+num_body
#    x2 = 2*i + num_body+1
#    x3 = 2*(i+1) + num_body
#    x4 = 2*(i+1) + num_body+1
#    p1 = len(numpy.where(x1==bnd_1)[0])
#    p2 = len(numpy.where(x2==bnd_1)[0])
#    p3 = len(numpy.where(x3==bnd_1)[0])
#    p4 = len(numpy.where(x4==bnd_1)[0])
    #print(p1)
    #print(p2)
    #print(p3)
    #print(p4)
#    sum_p = p1+p2+p3+p4
#    if sum_p==4:
#        molecular_ids = numpy.append(molecular_ids,int(i/2))

#print("End ",print(len(molecular_ids)))


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

        #findx1=numpy.where(x1==bnd_1)
        #findx2=numpy.where(x2==bnd_1)
        #findx3=numpy.where(x3==bnd_1)
        #findx4=numpy.where(x4==bnd_1)
        #print(len(findx1[0]),findx1[0])
        #if sum_p ==5 or sum_p==6 :
        #    if len(findx1[0])!=0:
        #        xf1 = bnd_1[findx1[0],1-findx1[1],1-findx1[2]]
        #        print("xf1",x1,xf1,dicts[xf1[0]])
        #        molecular_ids = numpy.append(molecular_ids,[dicts[xf1[0]]],axis=0)

        #    if len(findx2[0])!=0:
        #        xf2 = bnd_1[findx2[0],1-findx2[1],1-findx2[2]]
        #        print("xf2",x2,xf2,dicts[xf2[0]])
        #        molecular_ids = numpy.append(molecular_ids,[dicts[xf2[0]]],axis=0)
        #    if len(findx3[0])!=0:
        #        xf3 = bnd_1[findx3[0],1-findx3[1],1-findx3[2]]
        #        print("xf3",x3,xf3,dicts[xf3[0]])
        #        molecular_ids = numpy.append(molecular_ids,[dicts[xf3[0]]],axis=0)
        #    if len(findx4[0])!=0:
        #        xf4 = bnd_1[findx4[0],1-findx4[1],1-findx4[2]]
        #        print("xf4",x4,xf4,dicts[xf4[0]])
        #        molecular_ids = numpy.append(molecular_ids,[dicts[xf4[0]]],axis=0)

molecular_ids = numpy.unique(molecular_ids)
body_id_index = numpy.arange(0,len(p_ids),1)
molecular_ids = numpy.setdiff1d(body_id_index,molecular_ids)
constraint_type1 = numpy.zeros(len(constraint_pair1))
print(molecular_ids)
dist_pair =numpy.linalg.norm(box1.wrap(position[constraint_pair[:,0]] - position[constraint_pair[:,1]]),axis=1)
"""
filename0="Storebonds_nummolecule"+str(int(num_body/2))+".csv"
b=listofdictreader(filename0)
#print(b)
#for key, val in b.items():

molecular_ids1 = numpy.empty(0,dtype=int)
for key in molecular_ids:
    c=b[key]
    sum_n1=0
    for v in range(len(c)):
        print(b[c[v]])
        n1=len(b[c[v]])
        sum_n1+=n1
    if sum_n1==12 or sum_n1==16:
        molecular_ids1=numpy.append(molecular_ids1,key)

#print(molecular_ids1)
print(len(molecular_ids1),len(molecular_ids))
"""


#selfile="Selectedids"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".txt"
#arraysel = numpy.asarray((p_ids,n_ids)).T

#with open(selfile,'w+') as f11:
#    numpy.savetxt(f11,molecular_ids.T)

#exit()
#selfile="/lustre/project/nhr-trr146/jvishnu/Feringa/Piston/zerolatpress/Selectedids"+str(int(sample))+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".txt"

#selfile="../Selectedids"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".txt"


#molecular_ids = numpy.loadtxt(selfile).astype(int)
#print(molecular_ids.shape)

################################################################################
####################### Dihedral ###############################################
################################################################################
dihedral = numpy.empty((0,4),dtype=int)
dihedraltypes = numpy.empty(0,dtype=int)
improper = numpy.empty((0,4),dtype=int)
impropertypes = numpy.empty(0,dtype=int)
trapids = numpy.empty(0,dtype=int)
#postrap = numpy.empty((0,3),dtype=int)

for i in range(0,int(num_body/2)):
    x1 = 2*p_ids[i]+num_body
    x2 = p_ids[i]
    x3 = n_ids[i]
    x4 = 2*(n_ids[i])+num_body

    dihedral = numpy.append(dihedral,[[x1,x2,x3,x4]],axis=0)
    dihedraltypes = numpy.append(dihedraltypes,0)

for i in range(0,num_body-1,2):
    x1 = 2*i+num_body
    x2 = i
    x3 = i+1
    x4 = 2*(i+1)+num_body

    #x1 = 2*p_ids[i]+num_body
    #x2 = p_ids[i]
    #x3 = n_ids[i]
    #x4 = 2*(n_ids[i])+num_body


    x5 = 2*i+num_body+1
    x6 = i
    x7 = i+1
    x8 = 2*(i+1)+num_body+1
    #trapids = numpy.append(trapids,[x1,x2,x3,x4,x5,x8])

    #if ((i==10) | (i==12)):
    #    print(i,"Iam here")


    trapids = numpy.append(trapids,[x1,x2,x3,x4,x5,x8])

    #dihedral = numpy.append(dihedral,[[x1,x2,x3,x4]],axis=0)
    #dihedral = numpy.append(dihedral,[[x1,x2,x4,x3]],axis=0)
        #dihedral = numpy.append(dihedral,[[x1,x4,x3,x2]],axis=0)
    #dihedral = numpy.append(dihedral,[[x5,x6,x7,x8]],axis=0)
    #dihedral = numpy.append(dihedral,[[x5,x6,x8,x7]],axis=0)
        #dihedral = numpy.append(dihedral,[[x5,x8,x7,x6]],axis=0)
    #dihedraltypes = numpy.append(dihedraltypes,0)
    #dihedraltypes = numpy.append(dihedraltypes,1)
        #dihedraltypes = numpy.append(dihedraltypes,2)
    #dihedraltypes = numpy.append(dihedraltypes,0)
    #dihedraltypes = numpy.append(dihedraltypes,1)
        #dihedraltypes = numpy.append(dihedraltypes,2)
    #else:
    improper = numpy.append(improper,[[x1,x2,x3,x4]],axis=0)
    improper = numpy.append(improper,[[x5,x6,x7,x8]],axis=0)
    impropertypes = numpy.append(impropertypes,0)
    impropertypes = numpy.append(impropertypes,0)

postrap = numpy.take(position,trapids,axis=0)
print(dihedral)
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
#Lx = 2.5*(maxx-minx)
#Ly = 2.5*(maxy-miny)
#Lz = 2.5*(maxz-minz)

bondtypeid[len(newbonds):]=1
snap.particles.N = len(position)
snap.particles.position = position[:]
#snap.particles.position = real_pos[:]
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

snap.impropers.N = len(improper)
snap.impropers.group = improper
snap.impropers.typeid = impropertypes
snap.impropers.types = ['imp']



snap.pairs.N = len(constraint_pair1)
snap.pairs.typeid = constraint_type1
snap.pairs.types = ['const']
snap.pairs.group = constraint_pair1

#sim.create_state_from_snapshot(snap)
sim.timestep = 0

sim.create_state_from_gsd(filename,frame=-1)
assert sim.timestep == 0
newsnap=sim.state.get_snapshot()
newsnap.dihedrals.N = len(dihedral[:])
newsnap.dihedrals.typeid[:] = dihedraltypes
newsnap.dihedrals.group[:] = dihedral[:]
sim.state.set_snapshot(newsnap)
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

wca.params[('R','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('C','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('B','D')] = {'sigma':sigma,'epsilon': eps}
wca.params[('D','D')] = {'sigma':sigma,'epsilon': eps}

wca.params[('R','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('A','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('C','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('B','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('D','E')] = {'sigma':sigma,'epsilon': eps}
wca.params[('E','E')] = {'sigma':sigma,'epsilon': eps}


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
fenewca.params['poly'] = dict(k=k,r0=r0,epsilon=eps,sigma=sigma,delta=0)
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

#harmonic.params['pangles'] = dict(k=k_stiff,t0=math.pi)
################################################################################
################# Trapping potential ###########################################
################################################################################

trap_sinaction = hoomd.md.dihedral.TorsionalDihedral()
trap_sinaction.params['ditype'] = dict(k =600,d=-1,n=2,chk = -1)

loggerTors = hoomd.logging.Logger(categories=['dihedral', 'dihedral','dihedral','particle'])
#loggerTors=hoomd.logging.Logger(categories['dihedral','dihedral','dihedral','dihedral'])
loggerTors.add(trap_sinaction, quantities=['energies', 'forces','torques','virials'])
#exit()
############################constrain distance##################################
Constraint=hoomd.md.special_pair.LJ()
Constraint.params['const'] = dict(epsilon=0, sigma=1.0)
Constraint.r_cut['const'] = math.pow(2,1./6)

constant = hoomd.md.force.Constant(
    filter=hoomd.filter.All()
    )
constant.constant_force['D'] = (cforce,0,0)
constant.constant_force['E'] = (-cforce,0,0)

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


integrator.methods.append(npt)

sim.operations.integrator = integrator
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(rotperiod),
#                        filename="trajN"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                        mode='wb',dynamic=['momentum','property'])

#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(20000),
#                         filename="trajExtendnew"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".gsd",
#                         mode='wb',dynamic=['momentum','property'])
gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(rotperiod)),filename="traj"+str(4+int(sample))+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".gsd",mode='wb',dynamic=['momentum','property'])

#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic((rotperiod)),
#                          filename="trajExtendnewer"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                          mode='wb',dynamic=['momentum','property'])

# gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(100),
#                          filename="traj"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                          mode='wb',dynamic=['momentum','property'])



file="Angle"+str(4+int(sample))+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".h5"
#h5_writer = hoomd.write.CustomWriter(action=HDF5Writer_cpu(file,
#            'w',p_ids ,n_ids,molecular_ids,box1,freq,freq1),trigger=period_angle)


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
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=int(rotperiod)),
                          logger=logger)
Anglelog=anglelog(trap_sinaction)

sim.operations.writers.append(table)
sim.operations.computes.append(thermodynamic_properties)
###############################################################################

fenewca.params['R-X'] = dict(k=1000,r0=r0*0.85,epsilon=eps,sigma=0.85*sigma,delta=0)
#fenewca.params['poly'] = dict(k=1000,r0=r0,epsilon=eps,sigma=sigma,delta=0)

wca.params[('R','B')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','B')] = {'sigma':sigma,'epsilon': eps}

integrator.forces.append(constant)
filename1="/lustre/project/nhr-trr146/jvishnu/Feringa/Piston/zerolatpress/NewPulledFinalGel"+"_PBCyzHar"+str(int(sample)+4)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(cforce)+"_kT_"+str(kbT)+".gsd"


gsd_oper1 = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(langperiod)),filename=filename1,mode='wb',dynamic=['momentum','property'])

sim.operations += gsd_oper1
sim.run(langperiod)
sim.operations -= gsd_oper1
integrator.forces.append(trap_sinaction)
###############################################################################
sim.operations += gsd_oper
#gsd_oper.logger = log1
gsd_oper.logger = loggerTors

#anglelog = hoomd.logging.log(trap_sinaction.get_angles('ditype'),category='sequence',requires_run=True))
anglelog_ = hoomd.logging.Logger(hoomd.write.HDF5Log.accepted_categories)
anglelog_[('anglelog','angles')] = (Anglelog,'angles','sequence')
hdf5_log = hoomd.write.HDF5Log(
    trigger=hoomd.trigger.Periodic(int(rotperiod)),
    filename=file,
    logger=anglelog_,mode='w')
sim.operations.writers.append(hdf5_log)

#anglelogger = hoomd.logging.Logger(categories=['sequence'])
#anglelogger[('anglelog','get_angles')]=(Anglelog,'
for i in range(220):

    trap_sinaction.params['ditype'] = dict(k =600,d=-1,n=2,chk = -1)
    trap_sinaction.set_angles('ditype',refang_bool = True)
    print(i ,"of step ",220 )
    di_har.params['ditype'] = dict(k =kdihed,d=1,n=1,phi0 = 0)
    sim.run(rotperiod)
    
    #print(trap_sinaction.get_angles('ditype'))
    trap_sinaction.params['ditype'] = dict(k =600,d=-1,n=2,chk = 0)
    trap_sinaction.set_angles('ditype',refang_bool = True)
    
    integrator.forces.remove(di_har)
   
    sim.run(relaxationp)
    #print(trap_sinaction.get_angles('ditype'))
    
    integrator.forces.append(di_har)
    


    

    trap_sinaction.params['ditype'] = dict(k =600,d=-1,n=2,chk = -1)
    trap_sinaction.set_angles('ditype',refang_bool = True)


    di_har.params['ditype'] = dict(k =kdihed,d=-1,n=1,phi0 =0)
   
    sim.run(rotperiod)
    #print(trap_sinaction.get_angles('ditype'))

    integrator.forces.remove(di_har)
    

    trap_sinaction.params['ditype'] = dict(k =600,d=-1,n=2,chk = 0)

    trap_sinaction.set_angles('ditype',refang_bool = True)

    
    sim.run(relaxationp)
    #print(trap_sinaction.get_angles('ditype'))
    

    integrator.forces.append(di_har)


