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
################################################################################
########################## Initial values#######################################
################################################################################
seed3 = random.randint(1,65535) ##Random seed
print(seed3)
nlat=[4,2,1]
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
k = int(sys.argv[9])
#k1=5000
k1=int(sys.argv[8])
kang = 1000
r0=1.5*sigma
delta=0.0
cforce = float(sys.argv[10])
print(cforce)

kdihed = int(sys.argv[6])
no_beads = int(sys.argv[1])
Torque = float(sys.argv[2])
sample = sys.argv[3]
k_constraint =10000
rotperiod = int(sys.argv[4])
relaxationp = int(sys.argv[5])
langperiod = int(sys.argv[7])
dof = float(sys.argv[11])
interval = 1
period_angle = 20
freq = int((rotperiod+relaxationp)/period_angle)
#freq1= 4*int(rotperiod/period_angle)
freq1= int((rotperiod+relaxationp)/period_angle)
rm =.957

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

        #with sim.state.gpu_local_snapshot as data:
        with sim.state.cpu_local_snapshot as data:

            new_frame = self.file.create_group(str(self._cur_frame))

            particles = getattr(data, 'particles')
            Boxsim = getattr(data,'global_box')
            self.box1 = freud.box.Box(Boxsim.Lx,Boxsim.Ly,Boxsim.Lz)
            #pos = cp.asnumpy(getattr(particles,'position'))
            #image = cp.asnumpy(getattr(particles,'image'))
            #typeid = cp.asnumpy(getattr(particles,'typeid'))
            #rtag = cp.asnumpy(getattr(particles,'rtag'))

            pos = numpy.array(getattr(particles,'position'),copy=False)
            image = numpy.array(getattr(particles,'image'),copy=False)
            typeid = numpy.array(getattr(particles,'typeid'),copy=False)
            rtag = numpy.array(getattr(particles,'rtag'),copy=False)


            # num_body = len(self.postau)*2 just for this sim

            position=self.box1.unwrap(pos,image)

            central_pos1 = numpy.take(position,rtag[self.postau],axis=0)
            central_pos2 = numpy.take(position,rtag[self.negtau],axis=0)
            side_particle1 = rtag[self.postau*2+num_body]
            side_particle2 = rtag[self.negtau*2+num_body]
            side_particle12 = rtag[self.postau*2+num_body+1]
            side_particle22 = rtag[self.negtau*2+num_body+1]
            #tmp_mids = (self.postau+self.negtau-1)/4

            #print(self.postau)
            #print(self.negtau)
            #print(len(tmp_mids[self.m_ids]))
            #exit()
            side_pos1 = numpy.take(position,side_particle1,axis=0)
            side_pos2 = numpy.take(position,side_particle2,axis=0)

            side_pos12 = numpy.take(position,side_particle12,axis=0)
            side_pos22 = numpy.take(position,side_particle22,axis=0)

            new_vec1 = (-central_pos1+side_pos1)
            #print(new_vec1)
            new_vec2 = (central_pos2-central_pos1)
            new_vec3 = (central_pos1-central_pos2)
            new_vec4 = (side_pos2-central_pos2)
            # print(numpy.linalg.norm(new_vec1),numpy.linalg.norm(new_vec4))
            #print(numpy.dot(new_vec4.reshape(-1),new_vec1.reshape(-1)))
            #numpy.repeat(numpy.linalg.norm(new_vec4),3)
            DOT = numpy.sum(new_vec1[:]*new_vec4,axis=1)
            # print(DOT)

            CROSS = numpy.cross(new_vec1,new_vec4)
            # print(CROSS)

            tmp_dphi_i = numpy.arctan2(-CROSS[:,2],DOT)
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
def undo_constraint():
    global bond_name,constant_pair,Har
    s=UndoConstraint_distance(Har,constraint_pair,bond_name)
    s()
################################################################################
####################### Custom Torques #########################################
###############################################################################
class MyCustomGPUForce(hoomd.md.force.Custom):
    def __init__(self):
        super().__init__(aniso=True)
    def set_forces(self,timestep):
        with sim.state.gpu_local_snapshot as data:
                   idx1 = cp.array(data.particles.rtag[posttau_ids[:]],copy=False)
                   idx2 = cp.array(data.particles.rtag[negttau_ids[:]],copy=False)

                   with self.gpu_local_force_arrays as arrays:
                       arrays.torque[idx1[:]]=cp.array([0.,0.,Torque])
                       arrays.torque[idx2[:]]=cp.array([0.,0.,-Torque])
                       #arrays.torque[idx2[:]]=cp.array([0.,0.,0])
###############################################################################
########################## Set Harmonic Trap ###################################
################################################################################
"""
class Trap(hoomd.md.force.Custom):

    def __init__(self, k,r0,trapid):
        self._k = k
        self._r0 = r0
        self.trapid=trapid
        super().__init__(aniso=True)


    @property
    def trapforce(self):
        return self._r0

    @trapforce.setter
    def trapforce(self,r0):

        with sim.state.gpu_local_snapshot as data:
            particles = getattr(data,'particles')
            pos = cp.asnumpy(getattr(particles,'position'))
            image = cp.asnumpy(getattr(particles,'image'))
            rtag =cp.asnumpy(getattr(particles,'rtag'))
            tid1 = rtag[self.trapid[:]]


            tid1pos = pos[tid1]
            tid1image = image[tid1]

            tid1rpos = box1.unwrap(tid1pos,tid1image)
            self._r0  = tid1rpos
            #print(self._r0)

        print("Setting r0")

    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm

    def set_forces(self, timestep):
        # part_tag = rng.integers(self._state.N_particles)
        # direction = self._get_direction()
        # energy = self.energy(timestep)

        with sim.state.gpu_local_snapshot as data:
            particles = getattr(data,'particles')
            pos = cp.asnumpy(getattr(particles,'position'))
            image = cp.asnumpy(getattr(particles,'image'))
            rtag =cp.asnumpy(getattr(particles,'rtag'))
            tid1 = rtag[self.trapid[:]]


            tid1pos = pos[tid1]
            tid1image = image[tid1]
            tid1rpos = box1.unwrap(tid1pos,tid1image)

            dr = tid1rpos - self._r0
            #drmag = numpy.repeat(numpy.linalg.norm(dr),3).reshape(-1,3)
            f_array = -self._k*(dr)
            #print(f_array.shape)
            with self.gpu_local_force_arrays as arrays:
                arrays.force[tid1[:]]=f_array

"""
#############################################################################
##########Trap############################################################
###########################################################################
class Trap(hoomd.md.force.Custom):

    def __init__(self, k,ptau,ntau,numbody,Lx,Ly,Lz):
        self._k = k

        self.ptau=ptau
        self.ntau=ntau
        self.nvecp = numpy.zeros((len(self.ptau),3))
        self.nvecn = numpy.zeros((len(self.ptau),3))
        self.numbody = numbody
        self.boxarr = numpy.array([Lx,Ly,Lz])

        self.box1=freud.box.Box(Lx,Ly,Lz)
        super().__init__(aniso=True)


    @property
    def trapvec(self):
        return self.nvecp,self.nvecn

    @trapvec.setter
    def trapvec(self,r0):

        with sim.state.cpu_local_snapshot as data:
            particles = getattr(data,'particles')
            Boxsim = getattr(data,'global_box')
            self.box1=freud.box.Box(Boxsim.Lx,Boxsim.Ly,Boxsim.Lz)

            pos = numpy.array(getattr(particles,'position'),copy=False)
            image = numpy.array(getattr(particles,'image'),copy=False)
            rtag =numpy.array(getattr(particles,'rtag'),copy=False)
            ptau = rtag[self.ptau]
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]
            nvecp = self.box1.wrap(pos[sptau] -  pos[ptau])
            nvecn = self.box1.wrap (pos[sntau]-  pos[ntau] )
            nvecmagp = numpy.repeat(numpy.linalg.norm(nvecp,axis=1),3).reshape(-1,3)
            nvecmagn = numpy.repeat(numpy.linalg.norm(nvecn,axis=1),3).reshape(-1,3)
            self.nvecn=nvecn/nvecmagn
            self.nvecp=nvecp/nvecmagp




    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm

    def set_forces(self, timestep):
        # part_tag = rng.integers(self._state.N_particles)
        # direction = self._get_direction()
        # energy = self.energy(timestep)

        with sim.state.cpu_local_snapshot as data:
            particles = getattr(data,'particles')
            Boxsim = getattr(data,'global_box')
            self.box1=freud.box.Box(Boxsim.Lx,Boxsim.Ly,Boxsim.Lz)

            pos = numpy.array(getattr(particles,'position'),copy=False)
            image = numpy.array(getattr(particles,'image'),copy=False)
            rtag =numpy.array(getattr(particles,'rtag'),copy=False)
            ptau = rtag[self.ptau]
            #print(ptau)
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]
            nvecp = self.box1.wrap(pos[sptau] -  pos[ptau] )
            nvecn =  self.box1.wrap(pos[sntau] -  pos[ntau] )
            nvecmagp = numpy.repeat(numpy.linalg.norm(nvecp,axis=1),3).reshape(-1,3)
            nvecmagn = numpy.repeat(numpy.linalg.norm(nvecn,axis=1),3).reshape(-1,3)
            nvecp=nvecp/nvecmagp
            nvecn=nvecn/nvecmagn
            ndirp = numpy.cross(nvecp,self.nvecp)
            ndirn = numpy.cross(nvecn,self.nvecn)
            dotp = numpy.repeat(numpy.sum(self.nvecp*nvecp,axis=1),3).reshape(-1,3)
            dotn = numpy.repeat(numpy.sum(self.nvecn*nvecn,axis=1),3).reshape(-1,3)
            magfp =-2.0*self._k*dotp
            magfn =-2.0*self._k*dotn

            torqp = numpy.zeros((len(ptau),3),dtype=float)
            #torqp[:] = numpy.asarray([0,0,Torque])
            torqn = numpy.zeros((len(ntau),3),dtype=float)
            #torqn[:] = numpy.asarray([0,0,-Torque])
            #torqpind = numpy.where(ndirp[:,2]>0)[0]
            #torqnind = numpy.where(ndirn[:,2]<0)[0]
            torqp[:] = -magfp[:]*ndirp[:]
            torqn[:] = -magfn[:]*ndirn[:]

            #angp = numpy.arctan2(nvecp[:,1],nvecp[:,0])-numpy.arctan2(self.nvecp[:,1],self.nvecp[:,0])
            #angn = numpy.arctan2(nvecn[:,1],nvecn[:,0])-numpy.arctan2(self.nvecn[:,1],self.nvecn[:,0])
            #angp = numpy.where(angp<0,angp=angp+2*numpy.py,angp)
            #angn = numpy.where(angn<0,angn=angn+2*numpy.py,angn)
            #findp = numpy.where(numpy.fabs(angp)>numpy.pi)[0]
            #findn = numpy.where(numpy.fabs(angn)>numpy.pi)[0]
            #torqp[findp] = -torqp[findp]
            #torqn[findn] = -torqn[findn]

            #print(torqp)
            #if ndirp[:,2]>0:
            #    print(magfp[torqpind]*ndirp[torqpind])

            #print(torqp,ndirp,torqn,ndirn,torqpind,torqnind)

            with self.cpu_local_force_arrays as arrays:
                #mT1 =  cp.array([[0.,0.,Torque]])
                #mT2 =  cp.array([[0.,0.,-Torque]])
                #mT1 = cp.repeat(mT1,len(nvecp),axis=0)
                #mT2 = cp.repeat(mT2,len(nvecn),axis=0)
                #print(mT1)
                #print(mT2)
                #print(torqp,torqn)
                arrays.torque[ptau[:]]=torqp

                arrays.torque[ntau[:]]=torqn


#########################################################################
##### cpu Trapsin
#########################################################################
"""
class TrapSin(hoomd.md.force.Custom):

    def __init__(self, k,ptau,ntau,numbody,Lx,Ly,Lz):
        self._k = k

        self.ptau=ptau
        self.ntau=ntau
        self.nvecp = numpy.zeros((len(self.ptau),3))
        self.nvecn = numpy.zeros((len(self.ptau),3))
        self.nvecp1 = numpy.zeros((len(self.ptau),3))
        self.nvecn1 = numpy.zeros((len(self.ptau),3))
        self.anglevec = numpy.zeros(len(self.ptau))
        self.diff = numpy.zeros(len(self.ptau))
        self.angle = numpy.zeros(len(self.ptau))
        self.oldangle = numpy.zeros(len(self.ptau))
        self.mask = numpy.repeat(False,len(self.ptau))

        self.numbody = numbody
        self.boxarr = numpy.array([Lx,Ly,Lz])
        super().__init__(aniso=True)

    def getMask(self):
        mask=numpy.isclose(self.angle,numpy.pi,atol=0.3,rtol=1e-1)
        maskloc=numpy.where(mask==True)[0]
        self.mask[maskloc]=True
        return mask

    def normalize(self,nvecp,nvecn):
        nvecmagp = numpy.repeat(numpy.linalg.norm(nvecp,axis=1),3).reshape(-1,3)
        nvecmagn = numpy.repeat(numpy.linalg.norm(nvecn,axis=1),3).reshape(-1,3)
        return  nvecp/nvecmagp,nvecn/nvecmagn

    @property
    def trapvec(self):
        return self.nvecp,self.nvecn,self.mask,self.anglevec,self.oldangle,self.diff,self.angle

    def getDiff(self):
        self.diff = numpy.fmod(self.diff,2*numpy.pi)
        self.diff = numpy.where(self.diff<-numpy.pi,self.diff+numpy.pi*2,self.diff)
        self.diff = numpy.where(self.diff>=numpy.pi,self.diff-numpy.pi*2,self.diff)


    @trapvec.setter
    def trapvec(self,r0):

        self.mask = numpy.repeat(False,len(self.ptau))




        with sim.state.cpu_local_snapshot as data:
            particles = getattr(data,'particles')
            pos = numpy.array(getattr(particles,'position'))
            image = numpy.array(getattr(particles,'image'))
            rtag =numpy.array(getattr(particles,'rtag'))
            ptau = rtag[self.ptau]
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]
            sptau1= rtag[self.ptau*2+self.numbody+1]
            sntau1= rtag[self.ntau*2+self.numbody+1]

            nvecp = box1.wrap(pos[sptau] -  pos[ptau])
            nvecn = box1.wrap (pos[sntau]-  pos[ntau] )

            nvecp1 = box1.wrap(pos[sptau1] -  pos[ptau])
            nvecn1 = box1.wrap (pos[sntau1]-  pos[ntau] )

            nvecp,nvecn =self.normalize(nvecp,nvecn)
            nvecp1,nvecn1 =self.normalize(nvecp1,nvecn1)
            self.nvecn=nvecn
            self.nvecp=nvecp
            self.nvecn1=nvecn1
            self.nvecp1=nvecp1

            self.anglevec = numpy.arctan2(nvecp[:,1],nvecp[:,0])-numpy.arctan2(nvecn[:,1],nvecn[:,0])
            self.oldangle = self.anglevec
            self.diff = self.anglevec -self.oldangle
            self.getDiff()
            self.angle = self.diff






    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm

    def set_forces(self, timestep):
        # part_tag = rng.integers(self._state.N_particles)
        # direction = self._get_direction()
        # energy = self.energy(timestep)

        with sim.state.cpu_local_snapshot as data:

            particles = getattr(data,'particles')
            #pos = cp.asnumpy(getattr(particles,'position'))
            pos = numpy.array(getattr(particles,'position'))
            image = numpy.array(getattr(particles,'image'))
            rtag =numpy.array(getattr(particles,'rtag'))
            ptau = rtag[self.ptau]
            #print(ptau)
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]
            sptau1= rtag[self.ptau*2+self.numbody+1]
            sntau1= rtag[self.ntau*2+self.numbody+1]


            nvecp = box1.wrap(pos[sptau] -  pos[ptau] )
            nvecn =  box1.wrap(pos[sntau] -  pos[ntau] )
            nvecp1 = box1.wrap(pos[sptau1] -  pos[ptau] )
            nvecn1 =  box1.wrap(pos[sntau1] -  pos[ntau] )


            nvecp,nvecn =self.normalize(nvecp,nvecn)
            nvecp1,nvecn1 =self.normalize(nvecp1,nvecn1)

            ndirp = numpy.cross(nvecp,self.nvecp)
            ndirn = numpy.cross(nvecn,self.nvecn)
            ndirp1 = numpy.cross(nvecp1,self.nvecp1)
            ndirn1 = numpy.cross(nvecn1,self.nvecn1)


            dotp = numpy.repeat(numpy.sum(self.nvecp*nvecp,axis=1),3).reshape(-1,3)
            dotn = numpy.repeat(numpy.sum(self.nvecn*nvecn,axis=1),3).reshape(-1,3)
            magfp =-2.0*self._k*dotp
            magfn =-2.0*self._k*dotn

            dotp1 = numpy.repeat(numpy.sum(self.nvecp1*nvecp1,axis=1),3).reshape(-1,3)
            dotn1 = numpy.repeat(numpy.sum(self.nvecn1*nvecn1,axis=1),3).reshape(-1,3)
            magfp1 =-2.0*self._k*dotp1
            magfn1 =-2.0*self._k*dotn1


            torqp = numpy.zeros((len(ptau),3),dtype=float)
            torqp1 = numpy.zeros((len(ptau),3),dtype=float)
            #torqp[:] = numpy.asarray([0,0,Torque])
            torqn = numpy.zeros((len(ntau),3),dtype=float)
            torqn1 = numpy.zeros((len(ntau),3),dtype=float)
            #torqn[:] = numpy.asarray([0,0,-Torque])
            torqpind = numpy.where(ndirp[:,2]>0)[0]
            torqpind1 = numpy.where(ndirp1[:,2]>0)[0]
            utorqpind =  numpy.where(ndirp[:,2]<0)[0]

            torqnind = numpy.where(ndirn[:,2]<0)[0]
            torqnind1 = numpy.where(ndirn1[:,2]<0)[0]
            utorqnind =   numpy.where(ndirn[:,2]>0)[0]


            torqp[torqpind] = -magfp[torqpind]*ndirp[torqpind]
            torqn[torqnind] = -magfn[torqnind]*ndirn[torqnind]
            torqp1[torqpind1] = -magfp1[torqpind1]*ndirp1[torqpind1]
            torqn1[torqnind1] = -magfn1[torqnind1]*ndirn1[torqnind1]

            #torqp[utorqpind] = magfp[utorqpind]*ndirp[utorqpind]
            #torqn[utorqnind] = magfn[utorqnind]*ndirn[utorqnind]



            torqp[torqnind] += magfn[torqnind]*ndirn[torqnind]
            torqn[torqpind] += magfp[torqpind]*ndirp[torqpind]
            torqp1[torqnind1] += magfn1[torqnind1]*ndirn1[torqnind1]
            torqn1[torqpind1] += magfp1[torqpind1]*ndirp1[torqpind1]



            torqp[self.mask]=-magfp[self.mask]*ndirp[self.mask]*6
            torqn[self.mask]= -magfn[self.mask]*ndirn[self.mask]*6
            torqp1[self.mask]=-magfp1[self.mask]*ndirp1[self.mask]*6
            torqn1[self.mask]= -magfn1[self.mask]*ndirn1[self.mask]*6




            self.anglevec = numpy.arctan2(nvecp[:,1],nvecp[:,0])-numpy.arctan2(nvecn[:,1],nvecn[:,0])
            self.diff = self.anglevec -self.oldangle
            self.getDiff()
            self.angle += self.diff
            #print(self.angle)
            self.oldangle = self.anglevec
            #self.anglevecn = numpy.arctan2(nvecn[:,1],nvecn[:,0])-numpy.arctan2(self.nvecn[:,1],self.nvecn[:,0])
            mask=self.getMask()
            #print(mask)
            #print(self.mask)

            self.nvecp[mask]=nvecp[mask]
            self.nvecn[mask]=nvecn[mask]
            self.nvecp1[mask]=nvecp1[mask]
            self.nvecn1[mask]=nvecn1[mask]


            #torqp[self.mask]=-magfp[self.mask]*ndirp[self.mask]*6
            #torqn[self.mask]= -magfn[self.mask]*ndirn[self.mask]*6
            #torqp1[self.mask]=-magfp1[self.mask]*ndirp1[self.mask]*6
            #torqn1[self.mask]= -magfn1[self.mask]*ndirn1[self.mask]*6



            with self.cpu_local_force_arrays as arrays:
                #mT1 =  cp.array([[0.,0.,Torque]])
                mT1 =  numpy.array([[0.,0.,Torque]])
                #mT2 =  cp.array([[0.,0.,-Torque]])
                mT2 =  numpy.array([[0.,0.,-Torque]])
                #mT1 = cp.repeat(mT1,len(nvecp),axis=0)
                mT1 = numpy.repeat(mT1,len(nvecp),axis=0)
                #mT2 = cp.repeat(mT2,len(nvecn),axis=0)
                mT2 = numpy.repeat(mT2,len(nvecn),axis=0)
                #print(mT1)
                #print(mT2)
                #print(torqp,torqn)
                arrays.torque[ptau[:]]=torqp+torqp1+mT1
                #arrays.torque[ptau[:]]=cp.array(torqp)+cp.array(torqp1)+mT1

                arrays.torque[ntau[:]]=torqn+torqn1+mT2
                #arrays.torque[ntau[:]]=cp.array(torqn)+cp.array(torqn1)+mT2



"""
#########################################################################
##### cpu Trapsin
#########################################################################

#"""
class TrapSin(hoomd.md.force.Custom):

    def __init__(self, k,ptau,ntau,numbody,Lx,Ly,Lz):
        self._k = k

        self.ptau=ptau
        self.ntau=ntau
        self.nvecp = numpy.zeros((len(self.ptau),3))
        self.nvecn = numpy.zeros((len(self.ptau),3))
        self.nvecp1 = numpy.zeros((len(self.ptau),3))
        self.nvecn1 = numpy.zeros((len(self.ptau),3))
        self.anglevec = numpy.zeros(len(self.ptau))
        self.diff = numpy.zeros(len(self.ptau))
        self.angle = numpy.zeros(len(self.ptau))
        self.oldangle = numpy.zeros(len(self.ptau))
        self.mask = numpy.repeat(False,len(self.ptau))
        self.maskNegt = numpy.repeat(False,len(self.ptau))

        self.numbody = numbody
        self.boxarr = numpy.array([Lx,Ly,Lz])
        self.box1=freud.box.Box(Lx,Ly,Lz)
        self.step = 0
        super().__init__(aniso=True)

    def getMask(self):
        #mask=numpy.isclose(self.angle,numpy.pi,atol=0.3,rtol=1e-1)
        mask=((self.angle>=numpy.pi) & (self.angle<numpy.pi*3/2))
        #mask=(self.angle>=numpy.pi)
        maskloc=numpy.where(mask==True)[0]
        self.mask[maskloc]=True
        return mask

    def getMaskNegative(self):
        #mask=self.angle<=-0.55
        mask=self.angle<0
        #mask=self.angle<=-0.25

        maskloc=numpy.where(mask==True)[0]
        self.maskNegt[maskloc]=True
        return mask

    def getMaskZero(self):
        mask=numpy.isclose(self.angle,0,atol=0.8,rtol=1e-1)
        maskloc=numpy.where(mask==True)[0]
        #self.maskNegt[maskloc]=True
        return mask



    def normalize(self,nvecp,nvecn):
        nvecmagp = numpy.repeat(numpy.linalg.norm(nvecp,axis=1),3).reshape(-1,3)
        nvecmagn = numpy.repeat(numpy.linalg.norm(nvecn,axis=1),3).reshape(-1,3)
        return  nvecp/nvecmagp,nvecn/nvecmagn

    @property
    def trapvec(self):
        return self.nvecp,self.nvecn,self.mask,self.maskNegt,self.anglevec,self.oldangle,self.diff,self.angle

    def getDiff(self):
        self.diff = numpy.fmod(self.diff,2*numpy.pi)
        self.diff = numpy.where(self.diff<-numpy.pi,self.diff+numpy.pi*2,self.diff)
        self.diff = numpy.where(self.diff>=numpy.pi,self.diff-numpy.pi*2,self.diff)


    @trapvec.setter
    def trapvec(self,i):

        self.mask = numpy.repeat(False,len(self.ptau))
        self.maskNegt = numpy.repeat(False,len(self.ptau))
        self.step = i



        with sim.state.cpu_local_snapshot as data:
            particles = getattr(data,'particles')
            Boxsim = getattr(data,'global_box')

            self.box1=freud.box.Box(Boxsim.Lx,Boxsim.Ly,Boxsim.Lz)
            pos = numpy.array(getattr(particles,'position'),copy=False)
            image = numpy.array(getattr(particles,'image'),copy=False)
            rtag =numpy.array(getattr(particles,'rtag'),copy=False)
            ptau = rtag[self.ptau]
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]

            nvecp = self.box1.wrap(pos[sptau] -  pos[ptau])
            nvecn = self.box1.wrap (pos[sntau]-  pos[ntau] )


            nvecp,nvecn =self.normalize(nvecp,nvecn)

            self.nvecn=nvecn
            self.nvecp=nvecp

            self.anglevec = numpy.arctan2(nvecp[:,1],nvecp[:,0])-numpy.arctan2(nvecn[:,1],nvecn[:,0])
            self.oldangle = self.anglevec
            self.diff = self.anglevec -self.oldangle
            self.getDiff()
            self.angle = self.diff






    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm

    def set_forces(self, timestep):
        # part_tag = rng.integers(self._state.N_particles)
        # direction = self._get_direction()
        # energy = self.energy(timestep)

        with sim.state.cpu_local_snapshot as data:

            particles = getattr(data,'particles')
            #pos = cp.asnumpy(getattr(particles,'position'))
            Boxsim = getattr(data,'global_box')

            self.box1=freud.box.Box(Boxsim.Lx,Boxsim.Ly,Boxsim.Lz)

            pos = numpy.array(getattr(particles,'position'),copy=False)
            image = numpy.array(getattr(particles,'image'),copy=False)
            rtag =numpy.array(getattr(particles,'rtag'),copy=False)
            ptau = rtag[self.ptau]
            #print(ptau)
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]


            nvecp = self.box1.wrap(pos[sptau] -  pos[ptau] )
            nvecn =  self.box1.wrap(pos[sntau] -  pos[ntau] )


            nvecp,nvecn =self.normalize(nvecp,nvecn)


            mask=self.getMask()
            maskNegt=self.getMaskNegative()

            sinAngle= numpy.sin(self.angle)
            cosAngle= numpy.cos(self.angle)
            sinAngle1= numpy.sin(self.angle-numpy.pi)
            cosAngle1= numpy.cos(self.angle-numpy.pi)
            #direct = numpy.cross(nvecp,nvecn)


            tpNegt = numpy.zeros((len(ptau),3),dtype=float)
            tnNegt = numpy.zeros((len(ntau),3),dtype=float)

            potqpNegt = numpy.zeros(len(ptau),dtype=float)
            potqnNegt = numpy.zeros(len(ntau),dtype=float)

            tpNegt[maskNegt,2] =-2.0*self._k*sinAngle[maskNegt]*cosAngle[maskNegt]
            tnNegt[maskNegt,2] =2.0*self._k*sinAngle[maskNegt]*cosAngle[maskNegt]
            #tpNegt[maskNegt,2] =-2.0*self._k*direct[maskNegt,2]*cosAngle[maskNegt]*6
            #tnNegt[maskNegt,2] =2.0*self._k*direct[maskNegt,2]*cosAngle[maskNegt]*6


            potqpNegt[maskNegt]= self._k*numpy.square(sinAngle[maskNegt])
            potqnNegt[maskNegt]= self._k*numpy.square(sinAngle[maskNegt])


            exTp = numpy.zeros((len(ptau),3),dtype=float)
            exTn = numpy.zeros((len(ntau),3),dtype=float)
            #if (((timestep-langperiod)%rotperiod<10000 )&( self.step<100)):
            if (((timestep-langperiod)%rotperiod<rotperiod )):
                #print(timestep)
                maskZero =self.getMaskZero()
                exTp[maskZero,2] =Torque
                exTn[maskZero,2] = -Torque
                #exTp[:,2] =Torque
                #exTn[:,2] = -Torque


            #maskZero =self.getMaskZero()
            #exTp[maskZero,2] =Torque
            #exTn[maskZero,2] = -Torque



            torqp = numpy.zeros((len(ptau),3),dtype=float)
            torqn = numpy.zeros((len(ntau),3),dtype=float)

            potqp = numpy.zeros(len(ptau),dtype=float)
            potqn = numpy.zeros(len(ntau),dtype=float)




            torqp[self.mask,2]=-2*self._k*cosAngle1[self.mask]*sinAngle1[self.mask]
            torqn[self.mask,2]= 2*self._k*cosAngle1[self.mask]*sinAngle1[self.mask]
            #torqp[self.mask,2]=-2*self._k*cosAngle1[self.mask]*direct[self.mask,2]*6
            #torqn[self.mask,2]= 2*self._k*cosAngle1[self.mask]*direct[self.mask,2]*6

            potqp[self.mask]= self._k*numpy.square(sinAngle1[self.mask])
            potqn[self.mask]= self._k*numpy.square(sinAngle1[self.mask])

            #torqp[self.mask,2]=-2*self._k*cosAngle[self.mask]*sinAngle[self.mask]*6
            #torqn[self.mask,2]= 2*self._k*cosAngle[self.mask]*sinAngle[self.mask]*6



            self.anglevec = numpy.arctan2(nvecp[:,1],nvecp[:,0])-numpy.arctan2(nvecn[:,1],nvecn[:,0])
            self.diff = self.anglevec -self.oldangle
            self.getDiff()
            self.angle += self.diff

            self.oldangle = self.anglevec







            with self.cpu_local_force_arrays as arrays:
                #mT1 =  cp.array([[0.,0.,Torque]])
                mT1 =  numpy.array([[0.,0.,Torque]])
                #mT2 =  cp.array([[0.,0.,-Torque]])
                mT2 =  numpy.array([[0.,0.,-Torque]])
                #mT1 = cp.repeat(mT1,len(nvecp),axis=0)
                mT1 = numpy.repeat(mT1,len(nvecp),axis=0)
                #mT2 = cp.repeat(mT2,len(nvecn),axis=0)
                mT2 = numpy.repeat(mT2,len(nvecn),axis=0)
                #print(mT1)
                #print(mT2)
                #print(torqp,torqn)
                arrays.torque[ptau[:]]=torqp+tpNegt + exTp
                arrays.potential_energy[ptau[:]]=potqp+potqpNegt
                #arrays.torque[ptau[:]]=cp.array(torqp)+cp.array(torqp1)+mT1

                arrays.torque[ntau[:]]=torqn+tnNegt+exTn
                arrays.potential_energy[ntau[:]]=potqn+potqnNegt
                #arrays.torque[ntau[:]]=cp.array(torqn)+cp.array(torqn1)+mT2



#"""

#########################################################################
##### gpu Trapsin #####################################################
#########################################################################
"""
class TrapSin(hoomd.md.force.Custom):

    def __init__(self, k,ptau,ntau,numbody,Lx,Ly,Lz):
        self._k = k

        self.ptau=ptau
        self.ntau=ntau
        self.nvecp = cp.zeros((len(self.ptau),3))
        self.nvecn = cp.zeros((len(self.ptau),3))
        self.nvecp1 = cp.zeros((len(self.ptau),3))
        self.nvecn1 = cp.zeros((len(self.ptau),3))
        self.anglevec = cp.zeros(len(self.ptau))


        self.numbody = numbody
        self.boxarr = cp.array([Lx,Ly,Lz])
        super().__init__(aniso=True)

    def getMask(self):
        self.anglevec =cp.fmod(self.anglevec,2*cp.pi)
        self.anglevec = cp.where(self.anglevec<-cp.pi,self.anglevec+cp.pi*2,self.anglevec)
        self.anglevec = cp.where(self.anglevec>=cp.pi,self.anglevec-cp.pi*2,self.anglevec)
        mask=cp.isclose(cp.fabs(self.anglevec),cp.pi,rtol=1e-3)

        return mask


    @property
    def trapvec(self):
        return self.nvecp,self.nvecn

    def boxwrap(self,pos,image,id1,id2):
        diffr=pos[id1]+self.boxarr*image[id1]- pos[id2]+self.boxarr*image[id2]
        return diffr



    @trapvec.setter
    def trapvec(self,r0):

        with sim.state.gpu_local_snapshot as data:
            particles = getattr(data,'particles')
            pos = cp.array(getattr(particles,'position'))
            image = cp.array(getattr(particles,'image'))
            rtag =cp.array(getattr(particles,'rtag'))
            ptau = rtag[self.ptau]
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]
            sptau1= rtag[self.ptau*2+self.numbody+1]
            sntau1= rtag[self.ntau*2+self.numbody+1]

            nvecp = self.boxwrap(pos,image,sptau,ptau)
            nvecn = self.boxwrap (pos,image,sntau,ntau )
            nvecmagp = cp.repeat(cp.linalg.norm(nvecp,axis=1),3).reshape(-1,3)
            nvecmagn = cp.repeat(cp.linalg.norm(nvecn,axis=1),3).reshape(-1,3)

            nvecp1 = self.boxwrap(pos,image,sptau1, ptau)
            nvecn1 = self.boxwrap (pos,image,sntau1,ntau )
            nvecmagp1 = cp.repeat(cp.linalg.norm(nvecp1,axis=1),3).reshape(-1,3)
            nvecmagn1 = cp.repeat(cp.linalg.norm(nvecn1,axis=1),3).reshape(-1,3)


            self.nvecn=nvecn/nvecmagn
            self.nvecp=nvecp/nvecmagp
            self.nvecn1=nvecn1/nvecmagn1
            self.nvecp1=nvecp1/nvecmagp1




    def attach(self, simulation):
        self._state = simulation.state
        self._comm = simulation.device.communicator

    def detach(self):
        del self._state
        del self._comm


    def set_forces(self, timestep):
        # part_tag = rng.integers(self._state.N_particles)
        # direction = self._get_direction()
        # energy = self.energy(timestep)

        with sim.state.gpu_local_snapshot as data:
            start  = time.time()
            particles = getattr(data,'particles')
            pos = cp.array(getattr(particles,'position'),copy=False)
            image = cp.array(getattr(particles,'image'),copy=False)
            rtag =cp.array(getattr(particles,'rtag'),copy=False)
            ptau = rtag[self.ptau]
            #print(ptau)
            ntau = rtag[self.ntau]
            sptau= rtag[self.ptau*2+self.numbody]
            sntau= rtag[self.ntau*2+self.numbody]
            sptau1= rtag[self.ptau*2+self.numbody+1]
            sntau1= rtag[self.ntau*2+self.numbody+1]


            nvecp = self.boxwrap(pos,image,sptau ,ptau )
            nvecn =  self.boxwrap(pos,image,sntau,ntau )
            nvecp1 = self.boxwrap(pos,image,sptau1,ptau )
            nvecn1 =  self.boxwrap(pos,image,sntau1 ,  ntau )

            nvecmagp = cp.repeat(cp.linalg.norm(nvecp,axis=1),3).reshape(-1,3)
            nvecmagn = cp.repeat(cp.linalg.norm(nvecn,axis=1),3).reshape(-1,3)
            nvecmagp1 = cp.repeat(cp.linalg.norm(nvecp1,axis=1),3).reshape(-1,3)
            nvecmagn1 = cp.repeat(cp.linalg.norm(nvecn1,axis=1),3).reshape(-1,3)

            nvecp=nvecp/nvecmagp
            nvecn=nvecn/nvecmagn
            nvecp1=nvecp1/nvecmagp1
            nvecn1=nvecn1/nvecmagn1


            ndirp = cp.cross(nvecp,self.nvecp)
            ndirn = cp.cross(nvecn,self.nvecn)
            ndirp1 = cp.cross(nvecp1,self.nvecp1)
            ndirn1 = cp.cross(nvecn1,self.nvecn1)


            dotp = cp.repeat(cp.sum(self.nvecp*nvecp,axis=1),3).reshape(-1,3)
            dotn = cp.repeat(cp.sum(self.nvecn*nvecn,axis=1),3).reshape(-1,3)
            magfp =-2.0*self._k*dotp
            magfn =-2.0*self._k*dotn

            dotp1 = cp.repeat(cp.sum(self.nvecp1*nvecp1,axis=1),3).reshape(-1,3)
            dotn1 = cp.repeat(cp.sum(self.nvecn1*nvecn1,axis=1),3).reshape(-1,3)
            magfp1 =-2.0*self._k*dotp1
            magfn1 =-2.0*self._k*dotn1


            torqp = cp.zeros((len(ptau),3),dtype=float)
            torqp1 = cp.zeros((len(ptau),3),dtype=float)
            #torqp[:] = numpy.asarray([0,0,Torque])
            torqn = cp.zeros((len(ntau),3),dtype=float)
            torqn1 = cp.zeros((len(ntau),3),dtype=float)
            #torqn[:] = numpy.asarray([0,0,-Torque])
            torqpind = cp.where(ndirp[:,2]>0)[0]
            torqpind1 = cp.where(ndirp1[:,2]>0)[0]


            torqnind = cp.where(ndirn[:,2]<0)[0]
            torqnind1 = cp.where(ndirn1[:,2]<0)[0]



            torqp[torqpind] = -magfp[torqpind]*ndirp[torqpind]
            torqn[torqnind] = -magfn[torqnind]*ndirn[torqnind]
            torqp1[torqpind1] = -magfp1[torqpind1]*ndirp1[torqpind1]
            torqn1[torqnind1] = -magfn1[torqnind1]*ndirn1[torqnind1]




            torqp[torqnind] += magfn[torqnind]*ndirn[torqnind]
            torqn[torqpind] += magfp[torqpind]*ndirp[torqpind]
            torqp1[torqnind1] += magfn1[torqnind1]*ndirn1[torqnind1]
            torqn1[torqpind1] += magfp1[torqpind1]*ndirp1[torqpind1]

            self.anglevec = cp.arctan2(nvecp[:,1],nvecp[:,0])-cp.arctan2(nvecn[:,1],nvecn[:,0])
            mask=self.getMask()

            self.nvecp[mask]=nvecp[mask]
            self.nvecn[mask]=nvecn[mask]
            end = time.time()
            print(end-start)
            with self.gpu_local_force_arrays as arrays:
                mT1 =  cp.array([[0.,0.,Torque]])
                mT2 =  cp.array([[0.,0.,-Torque]])
                mT1 = cp.repeat(mT1,len(nvecp),axis=0)
                mT2 = cp.repeat(mT2,len(nvecn),axis=0)
                #print(mT1)
                #print(mT2)
                #print(torqp,torqn)
                arrays.torque[ptau[:]]=torqp+torqp1+mT1

                arrays.torque[ntau[:]]=torqn+torqn1+mT2



"""



###############################################################################
####################### Set Torque to zero #####################################
###############################################################################
class Set_Torque_Zero(hoomd.md.force.Custom):
    def __init__(self):
        super().__init__(aniso=True)
    def set_forces(self,timestep):
        with sim.state.gpu_local_snapshot as data:
                   idx1 = cp.array(data.particles.rtag[posttau_ids[:]],copy=False)
                   idx2 = cp.array(data.particles.rtag[negttau_ids[:]],copy=False)

                   with self.gpu_local_force_arrays as arrays:
                       arrays.torque[idx1[:]]=cp.array([0.,0.,0])
                       arrays.torque[idx2[:]]=cp.array([0.,0.,0])
################################################################################
####### function for setting and resetting constrain harmonic bonds ############
###############################################################################
def set_Harmonic():
    global bond_name,Har
    for i in range(0,len(bond_name)):
        if i==0:
            Har.params[bond_name[i]] = dict(k=k,r0=r0)


        else :
            Har.params[bond_name[i]] = dict(k=0,r0=r0)


################################################################################
############ functions for adding and removing forces ##########################
################################################################################
def add_forcestofire():
    global nve,nvtLang,fire,integrator,Har,Harmonic,wca,harmonic,nve,nvtLang,sim

    # integrator.forces.remove(wca)
    # integrator.forces.remove(harmonic)
    integrator.forces.remove(Har)
    integrator.forces.append(Harmonic)
    integrator.forces.remove(const_torque)
    #integrator.forces.remove(di_har)
    #integrator.methods.remove(nve)
    #integrator.methods.append(nvtLang)
    # fire.forces.append(wca)
    # fire.forces.append(harmonic)
    # fire.forces.append(Harmonic)
    # fire.methods.append(nve)
    # sim.operations.integrator = fire


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


def reset_h5(i):
    global h5_writer,sim,interval
    if (not i% interval):
        sim.operations.writers.remove(h5_writer)
        if i>400:
            interval = 20
        else:
            if i >=300 and i<1000:
                interval = 40
            else:
                if i>= 1000 and i<4000:
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
filename="./GelGraft_PBCyzHar"+str(sample)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"

#filename="traj"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(-0.3)+"_dof_"+str(dof)+".gsd"
#filename = "FreeGraftGel"+"_PBCyzHar"+str(sample)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(-0.3)+"_kT_"+str(kbT)+"_dof_"+str(dof)+".gsd"

#filename="../initial"+"_ucellHarsing_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#filename="initial"+"_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"

#filename="../PulledFinalGel"+"_PBCyzHar"+str(sample)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(cforce)+"_kT_"+str(kbT)+".gsd"
with gsd.hoomd.open(filename,mode='rb+') as traj:
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
    EDtype= ((typeid ==5)|(typeid==6))
    EDpos = position[EDtype]
    EDimage = image[EDtype]




box1 = freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
snap1 = gsd.hoomd.Snapshot()
snap1.particles.N = len(position)
snap1.particles.position = position[:]
snap1.particles.types= types
snap1.particles.body = body[:]
snap1.particles.mass = [1]*len(position)

snap1.particles.typeid =typeid[:]
snap1.particles.orientation = orientation[:len(orientation)]
snap1.particles.moment_inertia = moment_inertia
snap1.particles.velocity = vel
snap1.particles.image = image[:]
snap1.particles.angmom = angmom[:]

snap1.bonds.N = len(bonds)

snap1.angles.N = len(angles)
snap1.bonds.group = bonds[:]
snap1.bonds.types = bondtype
snap1.bonds.typeid = bondtypeid[:]

snap1.angles.types = angletype
snap1.angles.typeid = angletypeid[:]
snap1.angles.group = angles[:]

snap1.configuration.box = [Lx, Ly, Lz, 0, 0, 0]
snap1.particles.diameter = diameter[:]

snap1.dihedrals.N = len(dihedrals)
snap1.dihedrals.group = dihedrals
snap1.dihedrals.types = dihedralstype
snap1.dihedrals.typeid = dihedralstypeid

snap1.pairs.N = len(spairs)
snap1.pairs.types = spairtype
snap1.pairs.typeid = spairtypeid
snap1.pairs.group = spairs






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
     posttau_ids = numpy.append(posttau_ids,random.choice(temp1),axis=None)

id_seq = numpy.arange(0,len(bodyid),1)
negttau_ids = numpy.setdiff1d(id_seq,posttau_ids)

n_ids = numpy.copy(negttau_ids)
p_ids = numpy.copy(posttau_ids)
n_ids = numpy.sort(n_ids)
p_ids = numpy.sort(p_ids)

posttau_ids = cp.asarray(posttau_ids)
negttau_ids = cp.asarray(negttau_ids)

print(p_ids)
print(n_ids)
print(n_ids.shape,p_ids.shape)
#exit()

molfile="Molecularids"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".txt"
arraymol = numpy.asarray((p_ids,n_ids)).T

with open(molfile,'w+') as f:
    numpy.savetxt(f,arraymol)

################################################################################
num_body = len(bodyid)


################################################################################
##################### Selecting molecular ids that are of importance############
####################Constraint pair and distance################################
################################################################################
molecular_ids = numpy.empty(0,dtype=int)
for i in range(0, num_body-1,2):
    x1 = 2*i+num_body
    x2 = 2*i + num_body+1
    x3 = 2*(i+1) + num_body
    x4 = 2*(i+1) + num_body+1
    p1 = len(numpy.where(x1==bonds)[0])
    p2 = len(numpy.where(x2==bonds)[0])
    p3 = len(numpy.where(x3==bonds)[0])
    p4 = len(numpy.where(x4==bonds)[0])
    sum_p = p1+p2+p3+p4


    if sum_p ==5 or sum_p==6 or sum_p ==7:
        molecular_ids = numpy.append(molecular_ids,[int(i/2)],axis=0)


molecular_ids = numpy.unique(molecular_ids)
body_id_index = numpy.arange(0,len(p_ids),1)
molecular_ids = numpy.setdiff1d(body_id_index,molecular_ids)

selfile="Selectedids"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".txt"

with open(selfile,'w+') as f11:
    numpy.savetxt(f11,molecular_ids.T)

#exit()

################################################################################
####################### Dihedral ###############################################
################################################################################
trapids = numpy.empty(0,dtype=int)

for i in range(0,num_body-1,2):
    x1 = 2*i+num_body
    x2 = i
    x3 = i+1
    x4 = 2*(i+1)+num_body

    x5 = 2*i+num_body+1
    x6 = i
    x7 = i+1
    x8 = 2*(i+1)+num_body+1


    trapids = numpy.append(trapids,[x1,x2,x3,x4,x5,x8])


postrap = numpy.take(position,trapids,axis=0)
print(trapids)
################################################################################
######### Reading last line of initial sim file and creating a state ###########
################################################################################
sim.create_state_from_snapshot(snap1)
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
################################################################################
#################            Constant Torque             #######################
################################################################################
const_torque=MyCustomGPUForce()
###############################################################################
############################# Zero Torque ####################################
zero_torque = Set_Torque_Zero()
################################################################################
################# Trapping potential ###########################################
################################################################################
trap_action =  Trap(800,p_ids,n_ids,num_body,Lx,Ly,Lz)
#trap_action =  Trap(120,p_ids,n_ids,num_body,Lx,Ly,Lz)



trap_sinaction =  TrapSin(800,p_ids,n_ids,num_body,Lx,Ly,Lz)
#trap_sinaction =  TrapSin(20,p_ids,n_ids,num_body,Lx,Ly,Lz)
#trap_sinaction =  Oneway(600,p_ids,n_ids,num_body,Lx,Ly,Lz)

#trap_sinaction=TrapSin(100,posttau_ids ,negttau_ids,num_body,Lx,Ly,Lz) #GPU
# trap_operation=hoomd.update.CustomUpdater(action=trap_action, trigger=1)
############################constrain distance##################################
Constraint=hoomd.md.special_pair.LJ()
Constraint.params['const'] = dict(epsilon=0, sigma=1.0)
Constraint.r_cut['const'] = math.pow(2,1./6)
######################Constant Force ###########################################
################################################################################
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
integrator.forces.append(fenewca)


################################################################################
###### Adding rigid body constraints
################################################################################
rigid = hoomd.md.constrain.Rigid(); ## Setting up constraints
rigid.body['R'] = {
    "constituent_types": ['A','A'],
    "positions":[(0.85,0,0),(-0.85,0,0)],
    "orientations":[(1,0,0,0),(1,0,0,0)] ,
    "charges":[0.,0] ,
    "diameters":[1.,1.] }

rigid.body['X'] = {
    "constituent_types": ['C','C'],
    "positions":[(0.85,0,0),(-0.85,0,0)],
    "orientations":[(1,0,0,0),(1,0,0,0)] ,
    "charges":[0.,0] ,
    "diameters":[1.,1.] }
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

npt = hoomd.md.methods.NPT(filter=rigid_centers_and_free_filter , tau=.05, kT=kbT,
tauS = 0.5, S=[0.0,0.0,0.0,0,0,0], couple="yz",box_dof=[False,True,True,False,False,False],gamma=dof)

npt0 = hoomd.md.methods.NPT(filter=rigid_centers_and_free_filter , tau=.05, kT=kbT,
tauS = 0.5, S=[0,0,0,0,0,0], couple="yz",box_dof=[False,True,True,False,False,False],gamma=dof,rescale_all=True)


#npt = hoomd.md.methods.NPT(filter=rigid_centers_and_free_filter , tau=.05*10, kT=kbT,
#tauS = 0.5*10, S=0, couple="xyz",rescale_all=True)


#integrator.methods.append(npt)
print(npt.translational_thermostat_dof)
#npt.barostat_dof=[1,0,0,1,0,1]
#npt.rotational_thermostat_dof=[1,1]
#npt.translational_thermostat_dof=[1,1]


#nvtLang1 = hoomd.md.methods.Langevin(kT = kbT1, filter = unions)
nvtLang1 = hoomd.md.methods.Langevin(kT = kbT1, filter = rigid_centers_and_free_filter )
nvtLang1.gamma.default = 1

nvtLang1.gamma_r.default = [1.0, 1.0, 1]


integrator.methods.append(npt0)

sim.operations.integrator = integrator
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(10),
#                         filename="traj"+str(sample)+"nonPBC_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                         mode='wb',dynamic=['momentum','property'])

#filename="EquilibriumSlab"+"_PBCyzHar_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(2*(rotperiod+relaxationp)),
                         filename="traj"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".gsd",
                         mode='wb',dynamic=['momentum','property'])
#gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(50),
#                         filename="traj"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+".gsd",
#                         mode='wb',dynamic=['momentum','property'])




#filename1="NewPulledFinalGel"+"_PBCyzHar"+str(sample)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(cforce)+"_kT_"+str(kbT)+"_dof_"+str(dof)+".gsd"
#gsd_oper1 = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(langperiod),
#                         filename=filename1,mode='wb',dynamic=['momentum','property'])

#filename1="FreeGraftGel"+"_PBCyzHar"+str(sample)+"_FeringGel"+"_nbeads_"+str(no_beads)+"_cforce_"+str(cforce)+"_kT_"+str(kbT)+"_dof_"+str(dof)+".gsd"
#gsd_oper1 = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(langperiod),
#                         filename=filename1,mode='wb',dynamic=['momentum','property'])




file="Angle"+str(sample)+"nonPBCWork_FeringGel_torque_control"+"_nve_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+"_torque_"+str(Torque)+"_kdih_"+str(kdihed)+"_k1_"+str(k1)+"_k_"+str(k)+"_cforce_"+str(cforce)+"_dof_"+str(dof)+".h5"
h5_writer = hoomd.write.CustomWriter(action=HDF5Writer_cpu(file,
            'w',p_ids ,n_ids,molecular_ids,box1,freq,freq1),trigger=period_angle)

###############################################################################
############ Adding logger  to operations.writers #############################
################################################################################
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter= rigid_centers_and_free_filter)

log1 = hoomd.logging.Logger(categories=['scalar','scalar'])
log1.add(thermodynamic_properties,quantities=['potential_energy','kinetic_energy'])
gsd_oper.log = log1



logger = hoomd.logging.Logger(categories=['scalar', 'string','scalar'])
logger.add(sim, quantities=['timestep', 'tps'])
logger.add(thermodynamic_properties,quantities=['kinetic_temperature'])

status = Status(sim)
logger[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=int(rotperiod)),
                          logger=logger)
sim.operations.writers.append(table)
sim.operations.computes.append(thermodynamic_properties)
###############################################################################
integrator.forces.append(zero_torque)
integrator.forces.append(Constraint)
#sim.run(langperiod)

integrator.forces.remove(zero_torque)
#integrator.methods.remove(npt)

#integrator.methods.append(nvtLang1)

#integrator.forces.remove(Harmonic)

fenewca.params['R-X'] = dict(k=k,r0=.85*r0,epsilon=eps,sigma=.85*sigma,delta=0)
fenewca.params['R-X'] = dict(k=k,r0=.85*r0,epsilon=eps,sigma=.85*sigma,delta=0)

wca.params[('R','B')] = {'sigma':sigma,'epsilon': eps}
wca.params[('X','B')] = {'sigma':sigma,'epsilon': eps}


#filename="initialfinal"+"_ucellHarnonPBC_FeringGel"+"_nbeads_"+str(no_beads)+"_kT_"+str(kbT)+".gsd"
#gsd_oper1 = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(langperiod),
#                         filename=filename,
#                         mode='wb',dynamic=['momentum','property'])

#sim.operations += gsd_oper1

sim.run(langperiod)

integrator.methods.remove(npt0)
integrator.methods.append(npt)

#sim.operations -= gsd_oper1
#integrator.forces.append(const_torque)
#integrator.forces.append(trap_sinaction)
integrator.forces.append(trap_action)
integrator.forces.append(constant)
sim.operations += gsd_oper
sim.run(int(200*rotperiod*4))
print("Finished ")
exit()
#integrator.forces.append(Constraint)

#integrator.forces.remove(Harmonic)
#integrator.forces.append(fenewca)
###############################################################################



for i in range(200):

    #print(trap_sinaction.trapvec)
    #trap_sinaction.trapvec = postrap
    trap_sinaction.trapvec = i

    #kdihed = Torque
    print(i ,"of step ",1000 )
    set_h5(i)
    di_har.params['ditype'] = dict(k =kdihed,d=1,n=1,phi0 = 0)
    #di_har.params['ditype2'] = dict(k =4*kdihed,d=1,n=1,phi0 = 0)
    #di_har.params['ditype3'] = dict(k =4*kdihed,d=1,n=1,phi0 = 0)



    #integrator.forces.remove(const_torque)
    sim.run(rotperiod)

    #integrator.forces.remove(const_torque)
    #new_dis = UpdateConstraint_distance(sim,constraint_pair)
    #print(dist_pair)
    #new_r0=new_dis()
    trap_action.trapvec =postrap

    #print("Iam here")
    #print(trap_action.trapforce)
    integrator.forces.append(trap_action)
    #integrator.forces.remove(const_torque)
    integrator.forces.remove(trap_sinaction)

    #integrator.methods.remove(npt)

    #integrator.methods.append(nvtLang1)
    integrator.forces.remove(di_har)
    #print(nvtLang1.kbT)

    sim.run(relaxationp)
    #integrator.methods.remove(nvtLang1)

    integrator.forces.append(di_har)
    #integrator.methods.append(npt)



    integrator.forces.remove(trap_action)
    #integrator.forces.append(const_torque)
    integrator.forces.append(trap_sinaction)


    #print(trap_sinaction.trapvec)
    #trap_sinaction.trapvec = postrap
    trap_sinaction.trapvec = i


    di_har.params['ditype'] = dict(k =kdihed,d=-1,n=1,phi0 =0)
    #di_har.params['ditype2'] = dict(k =4*kdihed,d=-1,n=1,phi0 =0)
    #di_har.params['ditype3'] = dict(k =4*kdihed,d=-1,n=1,phi0 =0)


    #integrator.forces.remove(const_torque)

    #integrator.methods.remove(npt)

    sim.run(rotperiod)

    integrator.forces.remove(di_har)

    #integrator.methods.append(nvtLang1)

    #new_dis = UpdateConstraint_distance(sim,constraint_pair)
    #new_r0=new_dis()
    trap_action.trapvec = postrap

    #print(trap_action.trapforce)
    integrator.forces.append(trap_action)
    #integrator.forces.remove(const_torque)
    integrator.forces.remove(trap_sinaction)

    sim.run(relaxationp)
    #integrator.methods.remove(nvtLang1)

    integrator.forces.append(di_har)

    #integrator.methods.append(npt)


    integrator.forces.remove(trap_action)
    #integrator.forces.append(const_torque)
    integrator.forces.append(trap_sinaction)


    reset_h5(i)

h5_writer.file.flush()

#with h5py.File(file, 'r') as traj:
#      print(len(traj))
#      print(len(traj[str(1)]['pangles'][:]))
#      exit()

#      for i in range(1,len(traj)):
#          print(traj[str(i)]['pangles'][:])
