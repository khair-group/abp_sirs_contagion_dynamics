# abp_sirs_contagion_dynamics
MATLAB and HOOMD scripts to simulate contagion dynamics in a collection of active Brownian particles

In the description that follows, S=Susceptible, I=Infected, and R=Recovered. See
https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
for a detailed discussion on compartmental models in epidemiology. 

(1) The folder "macroscopic_model" consists of MATLAB scripts that solve a system of coupled ordinary differential 
    equations governing the time evolution of {S,I,R} populations. Within this folder: 

    (i) "transient_evolution_contagion_dynamics.mlx" plots the transient population as a function of time.
    (ii) "sir_model_engine.m" is a function with the same computational function as (i), except that it returns the 
      time series of {S,I,R} and does not plot anything.
    (iii) "steady_state_results_contagion_dynamics.m" is a driver routine that calls (ii) and stores the steady-state
       values of {S,I,R} for a range of epidemiological constants {\beta,\gamma,\alpha}.

The microscopic model for contagion dynamics considers each member of the various populations as a self-propelled agent, 
i.e., an active Brownian particle. 

(2) The file "microscopic_model_protocol_A.py" is a HOOMD script that implements the one-to-one protocol (Protocol A)
    in the microscopic model for contagion dynamics, in which each infected agent can potentially transmit the disease to
    only one susceptible agent within a contagion radius.

(3) The file "microscopic_model_protocol_B.py" is a HOOMD script that implements the one-to-many protocol (Protocol B)
    in the microscopic model for contagion dynamics, in which each infected agent can potentially transmit the disease to
    many susceptible agents within a contagion radius.  



