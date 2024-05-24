import multiprocessing as mp
import gsd.hoomd
import hoomd
import numpy as np
import os
import sys
import math
import random

from scipy.cluster.hierarchy import *
from scipy.spatial.distance import *

# 15-DEC-2023: Code to simulate contagion dynamics {S,I,R,S} in a collection of Active Brownian Particles
# moving in two dimensions. This code implements Protocol B (one-to-many transmission), in which an 
# infected particle can potentially transmit the disease to many susceptible particles within a contagion radius.

l=len(sys.argv)

if l<14:
    print("\n Correct syntax is [microscopic_model_protocol_B.py] [path to input config file] [restart? (0/1)] [delta t] [no of timesteps]",\
          "[D_r] [output folder for results] [save_frequency]",\
           "[beta] [gamma] [alpha] [s0_frac] [i0_frac] [contag_rad] \n")
    exit(0)

else :
    pass


inp_file_path=sys.argv[1]
restart_param=int(float(sys.argv[2]))
time_step_width=float(sys.argv[3])       # timestep width used in simulation 
sim_length=int(float(sys.argv[4])) 	 # total number of timesteps for which simulation must be run
rot_diff_const=float(sys.argv[5])        # rotational diffusion constant
op_folder_path=sys.argv[6]               # path to input file, containing the raw, wrapped positions
save_freq=int(sys.argv[7])               # time-intervals at which the simulation output is written to file
rate_inf = float(sys.argv[8])            # rate of spread of infection
rate_rec = float(sys.argv[9])            # rate of recovery
rate_relapse = float(sys.argv[10])       # rate of conversion of R --> S
S0_by_N = float(sys.argv[11])            # fraction of susceptible particles in initial configuration
I0_by_N = float(sys.argv[12])            # fraction of infected particles in initial configuration
rc_param = float(sys.argv[13])           # interaction radius over which disease spreads 


class PrintTimestep(hoomd.custom.Action):

    def act(self, timestep):
        pass
#        print(timestep, " timesteps computed")

# Add thermal noise to Overdamped Viscous integrator
# using custom force method

class ThermalNoise_2D(hoomd.md.force.Custom):
 def __init__(self,D_t,N,dt):
   super().__init__(aniso=False)
   self.D_t = D_t
   self.thermal_noise=np.zeros((N,3))
   self.dt=dt
   self.N=N

 def update_force(self,timestep):
   translation_constant = np.sqrt(2*self.D_t*self.dt)
   self.thermal_noise = np.random.normal(0,1,[self.N,3])*(translation_constant)
   self.thermal_noise[:,2] = 0.

 def set_forces(self,timestep):
   self.update_force(timestep)
   with self.cpu_local_force_arrays as arrays:
     arrays.force[:] =self.thermal_noise
     pass


# Calculate custom force in HOOMD
class CustomActiveForce_2D(hoomd.md.force.Custom):
 def __init__(self,v0,rotation_diff,N,dt,sf, r_cut, rate_inf, rate_rec, rate_relapse):
   super().__init__(aniso=False)
   self.v0_ref = v0 #base value of self-propulsion speed
   self.rotation_diff = rotation_diff
   self.active_fi=np.zeros((N,3))
   self.dt=dt
   self.N=N
   self.sf=sf #how fast the infected species move w.r.t other species
   self.r_cut = r_cut
   self.rate_inf = rate_inf
   self.rate_rec = rate_rec
   self.rate_relapse = rate_relapse

 def tagged_dist_calc(self,pos_a,tag_pos,pos_b,b_ind,L):
   num_rows_a = len(pos_a)
   num_rows_b, num_cols_b = pos_b.shape

   P = []
   if (num_rows_a==0 | num_rows_b==0):
       D=np.array([-1,-1])

   D = np.zeros((1,num_rows_b))

   for i in range(1):
       for j in range(num_rows_b):
           dx = pos_b[j,0] - pos_a[0]
           dy = pos_b[j,1] - pos_a[1]
           if (dx > (0.5*L)):
               dx = dx - L
           if (dx < (-0.5*L)):
               dx = dx + L
           # similarly for the y-coordinate
           if (dy > (0.5*L)):
               dy = dy - L
           if (dy < (-0.5*L)):
               dy = dy + L
           D[i,j] = np.sqrt((dx*dx)+(dy*dy)) 
           P.append([tag_pos,b_ind[0][j]]) 

   return D,P

 def ret_nbd_info(self,D,P,r_cut):
 # Takes in the inter-particle distances at a given time-frame. 
 # Returns a list of particles that are within a distance "r_cut"
 # of each other
   ro,col = np.where( (D<=r_cut) & (D>0) )      
          
   cell_nbd_inf=[]
   
   for ci in range(len(col)):
       chk=P[col[ci]]
       cell_nbd_inf.append(chk[1])

   index_nbd_inf=np.array(cell_nbd_inf)
   nbd_inf=len(index_nbd_inf)

   return nbd_inf, index_nbd_inf


 def update_force(self,timestep):

   with self._state.cpu_local_snapshot as data:
       quati = np.array(data.particles.orientation.copy())
       ptag = np.array(data.particles.tag.copy())
       pos = np.array(data.particles.position.copy())
       type_list = np.array(data.particles.typeid.copy())


   s_indices = np.where(type_list == 0)
   i_indices = np.where(type_list == 1)
   r_indices = np.where(type_list == 2)

### v0_arr has to take on appropriate values based on the identity of the particle, i.e, if 
### a given particle is "S", "I", or "R".

   v0_arr = np.zeros((self.N,1))

   v0_arr[s_indices] = self.v0_ref
   v0_arr[i_indices] = self.v0_ref * self.sf
   v0_arr[r_indices] = self.v0_ref

   pos_sus = pos[s_indices]
   pos_inf = pos[i_indices]
   pos_rec = pos[r_indices]
   num_inf = len(pos_inf)
   num_rec = len(pos_rec)

   box = self._state.cpu_local_snapshot.global_box
   L = box.Lx

### rotational diffusion implementation

   rotation_constant = np.sqrt(2*self.rotation_diff*self.dt)
   temp_b=np.array([0.,0.,1.])  
   b=np.zeros((self.N,3))
   b[:]=temp_b	

   delta_theta=np.random.normal(0,1,[self.N,1])*(rotation_constant)

   sin_comp=np.multiply(b,np.sin(delta_theta/2.))
   cos_comp=np.cos(delta_theta/2.)

   q_scal = (quati[:,0]).reshape(self.N,1)
   q_vec = (quati[:,1:4]).reshape(self.N,3)

## See lines 853-856 in hoomd-blue/hoomd/VectorMath.h available on the gitHub page: 
## https://github.com/glotzerlab/hoomd-blue/blob/8fc5b26a4a28d42c86afc3740437aea66578af19/hoomd/VectorMath.h#L713
   scal_comp = cos_comp*q_scal - ((np.sum(sin_comp*q_vec,axis=1)).reshape(self.N,1))
## the second term on the RHS of the above equation implements row-wise dot-product
   vec_comp = np.multiply(cos_comp,q_vec) + np.multiply(q_scal,sin_comp) + np.cross(sin_comp,q_vec,axisa=1,axisb=1) 

   quati = np.concatenate((scal_comp,vec_comp),axis=1)

   q_scal = (quati[:,0]).reshape(self.N,1) #update after rotation
   q_vec = (quati[:,1:4]).reshape(self.N,3) #update after rotation

### end of rotational diffusion implementation. Now starting force calculation

   f = v0_arr*np.ones((self.N,3)) 
   f[:,1:3]=0.

### rotating f by the quaternion
## See lines 947-960 in hoomd-blue/hoomd/VectorMath.h available on the gitHub page:
## https://github.com/glotzerlab/hoomd-blue/blob/8fc5b26a4a28d42c86afc3740437aea66578af19/hoomd/VectorMath.h#L947

   vec_term1 = (q_scal*q_scal - ((np.sum(q_vec*q_vec,axis=1)).reshape(self.N,1)))*f
   vec_term2 = 2*q_scal*np.cross(q_vec,f)
   vec_term3 = ((2*np.sum(q_vec*f,axis=1)).reshape(self.N,1))*q_vec

   self.active_fi = vec_term1 +vec_term2 + vec_term3
       

### update orientations ####
   with self._state.cpu_local_snapshot as data:
       data.particles.orientation = quati

#### update particle identities based on a kinetic Monte Carlo like scheme

## loop over infected species
   E = np.zeros((3)) 
   event_flag = np.zeros(3)

   op_arr='%d\t%d\t%d\t%d' %(timestep,len(pos_sus),len(pos_inf),len(pos_rec))
 
   print(op_arr)

   num_inf_list = list(range(num_inf))
   rand_p_i = random.sample(num_inf_list,num_inf)
 
   for p_i in range(num_inf):
       ### return the indices of neighbors within a cut-off distance of the selected "I" particle
       rnd_num=np.random.rand(2)
       pos_inf_tag=pos_inf[p_i,:]
       
       D_IS, P_IS = self.tagged_dist_calc(pos_inf_tag,i_indices[0][p_i],pos_sus,s_indices,L)
       D_II, P_II = self.tagged_dist_calc(pos_inf_tag,i_indices[0][p_i],pos_inf,i_indices,L)

       ## Get census of all susceptible particles in the neighborhood of infected particles
 
       S_c, index_S_c = self.ret_nbd_info(D_IS,P_IS,self.r_cut)
       I_c, index_I_c = self.ret_nbd_info(D_II,P_II,self.r_cut)

       if(index_I_c.size==0):
           I_c=0
           index_I_c=np.append(index_I_c,i_indices[0][p_i])
           index_I_c=index_I_c.astype(int)

       I_c = I_c+1
       E[0] = self.rate_inf*I_c*self.dt

       if(E[0]>1):
           E[0]=1

       for i in range(3):
           event_flag[i] = np.random.binomial(1,E[i])

       # nomenclature
       # S : 0
       # I : 1
       # R : 2

       # statements for S --> I conversion
       if(index_S_c.size>0):
           inf_vec = np.random.binomial(1,E[0],S_c)
           num_infected = np.sum(inf_vec)
           chk=np.random.choice(len(index_S_c),num_infected);
           if(num_infected>0):
               rand_S=index_S_c[chk];
               type_list[rand_S] = 1


   E[1] = self.rate_rec*self.dt
   E[2] = self.rate_relapse*self.dt

   if(E[1]>1):
       E[1]=1
   if(E[2]>1):
       E[2]=1 
      
### adding in the part that governs the transition from I --> R
   if(len(pos_inf)>0):
       rec_vec = np.random.binomial(1,E[1],num_inf)
       num_recovered = np.sum(rec_vec)
       chk=np.random.choice(len(pos_inf),num_recovered) 
       if(num_recovered>0):
           rand_I=i_indices[0][chk]
           type_list[rand_I] = 2
 

#### adding in the part that governs the transition from R --> S   
   if(len(pos_rec)>0):
       sus_vec = np.random.binomial(1,E[2],num_rec)
       num_relapsed = np.sum(sus_vec)
       chk=np.random.choice(len(pos_rec),num_relapsed) 
       if(num_relapsed>0):
           rand_R=r_indices[0][chk]
           type_list[rand_R] = 0
   
### update particle identities####
   with self._state.cpu_local_snapshot as data:
       data.particles.typeid = type_list


 def set_forces(self,timestep):
   self.update_force(timestep)
   with self.cpu_local_force_arrays as arrays:
     arrays.force[:] =self.active_fi
     pass


# Function to build and run simulation
def simulate(fname,op_dir,v0,scale_fac,contag_rad,rate_inf,rate_rec,rate_relapse,S0_by_N,I0_by_N):

    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=1)
    if(restart_param==0):
        sim.timestep=0

    sim.create_state_from_gsd(fname)
 

    N = sim.state.N_particles
    S0 = math.ceil(S0_by_N * N)
    I0 = N - S0
    R0 = 0

    with sim._state.cpu_local_snapshot as data:
        type_list = np.array(data.particles.typeid.copy())
 
    type_list[0:S0] = 0 # susceptible
    type_list[S0:S0+I0] = 1 #infected
    type_list[S0+I0:S0+I0+R0] = 2 #recovered

    with sim._state.cpu_local_snapshot as data:
        data.particles.typeid = type_list
 
    # More simulation parameters
    tsteps=sim_length # total number of simulations steps to be executed
    dt = time_step_width # timestep width used for integration

    integrator = hoomd.md.Integrator(dt, integrate_rotational_dof=False)
    cell = hoomd.md.nlist.Cell(buffer=0.4)

    custom_active =CustomActiveForce_2D(v0 = v0, rotation_diff=d_r, N=N, dt=dt, \
                   sf = scale_fac, r_cut = contag_rad, \
                   rate_inf = rate_inf, rate_rec = rate_rec, rate_relapse = rate_relapse, \
                   )

    integrator.forces.append(custom_active)

    thermal_noise = ThermalNoise_2D(D_t=d_t, N=N, dt=dt)
    integrator.forces.append(thermal_noise)


    # Heyes-Melrose custom excluded volume interaction
    gamma = 1
    rad = 1.0
    r_hs_min = rad
    r_hs_cut = 2.*rad
    size_hs = 2.*rad
    r = np.linspace(r_hs_min, r_hs_cut, 2, endpoint=False)

    # Create hard sphere interaction potential and force
    U_hs = gamma/(4*dt)*(r-2*rad)**2
    F_hs = -gamma/(2*dt)*(r-2*rad)
    # Use tabulated potential for hard sphere interactions
    hard_sphere = hoomd.md.pair.Table(nlist=cell, default_r_cut=r_hs_cut)

    hard_sphere.params[('S', 'S')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)
    hard_sphere.params[('S', 'I')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)
    hard_sphere.params[('S', 'R')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)
    hard_sphere.params[('I', 'I')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)
    hard_sphere.params[('I', 'R')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)
    hard_sphere.params[('R', 'R')] = dict(r_min=r_hs_min, U=U_hs, F=F_hs)



    if (hs_flag>0):
        integrator.forces.append(hard_sphere)

    # Add thermal noise using langevin integrator
#    ld = hoomd.md.methods.Langevin(filter=hoomd.filter.All(),kT=0.1)
#    integrator.methods.append(ld)

    # Use overdamped-viscous integrator
    odv = hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All())
    odv.gamma.default = gamma
    odv.gamma_r.default = [0, 0, 0]  # Ignore rotational drag

    integrator.methods.append(odv)

    sim.operations.integrator = integrator

    custom_action = PrintTimestep()
    custom_op = hoomd.write.CustomWriter(
    action=custom_action,
    trigger=hoomd.trigger.Periodic(100))

    # Run simulation
 
    op_file_name='N_%s_D_t_%s_D_r_%s_v0_%s_tsteps_%s_dt_%s_beta_%s_gamma_%s_alpha_%s_rc_%s_sirs_S0_%s_I0_%s.gsd' %(N,d_t,d_r,v0,tsteps,dt,rate_inf,rate_rec,rate_relapse,contag_rad,S0_by_N,I0_by_N)
    gsd_writer = hoomd.write.GSD(
        filename= op_dir + op_file_name,
        trigger=hoomd.trigger.Periodic(save_freq),
        mode='wb',dynamic=['property','momentum','attribute'])
    sim.operations.writers.append(gsd_writer)
    sim.run(tsteps)

# Set global variables

inp_config=inp_file_path #input configuration file
hs_flag=1 #whether or not to include steric repulsion, 0 means phantom disks
d_r = rot_diff_const # rotational diffusion constant
d_t = 0. # translational diffusion constant
scale_fac = 0. # how fast the infected species move with respect to the susceptible or recovered species

contag_rad = rc_param # interaction radius over which disease spreads

fa_list=np.array([0.1]) #self-propulsion speed of active particle


# Set up simulation and write to the appropriate directory

curdir = os.getcwd()
resdir=op_folder_path

try:
    os.mkdir(resdir)
    print("Directory made")
except FileExistsError:
    print("Directory already exists")


v0 = 0.1

simulate(inp_config,resdir,v0,scale_fac,contag_rad,rate_inf,rate_rec,rate_relapse,S0_by_N,I0_by_N)


prompt_file_name=resdir + "input_prompt.txt"
p_file=open(prompt_file_name,"w")

p_file.write("Input syntax: [microscopic_model_protocol_B.py] [path to input config file] [restart? (0/1)] [delta t] [no of timesteps],\
      [D_r] [output folder for results] [save_frequency],\
      [beta] [gamma] [alpha] [s0_frac] [i0_frac] [contag_rad]")
 
p_file.write("\n")
p_file.write("Actual input: " + str(sys.argv))
p_file.close()



print("Simulations completed")
