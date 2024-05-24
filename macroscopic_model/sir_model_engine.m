function [S,I,R] = sir_model_engine(beta,gamma,alpha,N,S0,I0,T,dt)
    % if alpha = 0 we assume a model without immunity loss
    S = zeros(1,T/dt);
    S(1) = S0;
    I = zeros(1,T/dt);
    I(1) = I0;
    R = zeros(1,T/dt);
    R(1)=N-S0-I0;
    for tt = 1:(T/dt)-1
        % Equations of the model
        dS = (-beta*I(tt)*S(tt)./N + alpha*R(tt)) * dt;
        dI = (beta*I(tt)*S(tt)./N - gamma*I(tt)) * dt;
        dR = (gamma*I(tt) - alpha*R(tt)) * dt;
        S(tt+1) = S(tt) + dS;
        I(tt+1) = I(tt) + dI;
        R(tt+1) = R(tt) + dR;
    end
end

