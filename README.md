# Instructions
## To simulate the load simulation set k_stiff parameter to zero
- Run 1st equilibriumstiff.py, alat = 55,100,150,250 for n_beads = 21,40,62,84 respectively. Refer thesis for other inputs
- Then run walladdstiff.py to add walls
- Then graft the wall to gel using graftequilstiff.py and equilibrate
- now use pullgraftgelstiff.py to pull the gel with a constant force
- run production.py to run contraction simulations with 
    parameters :
    dof = 0
    cforce in the list
    k = 200
    k1= 200
    kang = 1000
    kdihed = 300
    no_beads = 21
    Torque = 35.0
    sample = 1
    rotperiod = 20000
    relaxationp = 20000
    langperiod = 60000000
    Ktors = 600

## To simulate the free simulation set k_stiff parameter to zero
- Run 1st ucell_moiSideStiffFree.py to create gel , alat = 55,100,150,250 for n_beads = 21,40,62,84 respectively. Refer thesis for 
  other inputs.
- After initialization run Freestiffgel.py for relaxing the gel
- Run FreeStiffproduction.py for contraction
- Continue contraction using continueproduction.py
$1 600 $3 20000 20000 759.375 0 200 200 $2
    parameters:
    k = 200
    k1=200
    Ktors = 600
    kmax = 759.375
    
    no_beads = 21 or 40 or 62 or 84
    sample = 1
    k_constraint =10000
    rotperiod = 20000
    relaxationp = 20000
    langperiod = 0
- 600  cycles of contraction already defined within the code
Please use dt=5e-4 for Thesis reproduction.
Hoomd download on request: https://github.com/judevishnu/mytors_hoomd.git