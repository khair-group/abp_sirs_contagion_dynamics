%%%% Driver routine that runs "sir_model_engine" and
%%%% calculates the steady-state values of the various
%%%% sub-populations for different parameter sets

N=1000;
T=1000;
dt=0.1;

S0_frac=0.9;
I0_frac=0.1;

S0=S0_frac*N;
I0=I0_frac*N;

list_of_beta=[0.2];
list_of_gamma=[0.002,0.005,0.01,0.02,0.05,0.08,0.1,0.125,0.15,0.2,0.3,0.5,1.,2.,3.,5.,10.,20.];
list_of_alpha=[0.15];


num_beta=length(list_of_beta);
num_gamma=length(list_of_gamma);
num_alpha=length(list_of_alpha);

chk=[];

for i=1:num_beta
    for j=1:num_gamma
        for k=1:num_alpha
            [S,I,R] = sir_model_engine(list_of_beta(i),list_of_gamma(j),list_of_alpha(k),N,S0,I0,T,dt);
            chk=[chk; list_of_beta(i) list_of_gamma(j) list_of_alpha(k) S(end)./N I(end)./N R(end)./N];
        end
    end
end

save("S0_0_9_I0_0_1_steady_vals_beta_0_2_alpha_0_15.mat","chk");
            
