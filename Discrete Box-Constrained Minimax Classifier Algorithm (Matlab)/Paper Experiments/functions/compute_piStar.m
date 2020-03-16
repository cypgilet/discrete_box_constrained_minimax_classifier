function [rStar,piStar,Rstar] = compute_piStar(Xcomb,pHat,YRTrain,K,N,L,Box)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # Xcomb   : profiles {x1,...,xT}.
%   # pHat    : see equation (14) in the paper.
%   # YRTrain : real labels.
%   # K       : number of classed.
%   # N       : Number of iteration in scheme (23) in the paper.
%   # L       : Loss function.
%   # Box     : Box-Constraint.
%======OUTPUTS:
%   # rStar       : Maximum of V over U
%   # rSpiStartar : priors which maximize of V over U
%   # Rstar       : conditional risks associated to BCDMC.
%--------------------------------------------------------------------------

T = size(Xcomb,1);
pi = zeros(1,K);

for k = 1:K
    pi(k) = sum(YRTrain==k)/size(YRTrain,1);
end


rStar = 0;
piStar = pi;
Rstar = 0;

for n = 1:N
    
    rho = 1/n;
    
    % compute subgradient:
    lambda = zeros(K,T);
    for l = 1:K
        for t = 1:T
            for k = 1:K
                lambda(l,t) = lambda(l,t) + L(k,l)*pi(k)*pHat(k,t);
            end
        end
    end
    
    R = zeros(1,K);
    for k = 1:K
        mu_k = 0;
        for t = 1:T
            [~,lmin] = min(lambda(:,t));
            mu_k = mu_k + L(k,lmin(1))*pHat(k,t);
        end
        R(k) = mu_k;
    end
    
    % compute V(pi) and update rStar, piStar, Rstar:
    r = dot(pi,R);
    if r > rStar
        rStar = r;
        piStar = pi;
        Rstar = R;
    end
    
    % subgradient step:
    eta = max(1,norm(R));
    pi_new = pi + rho*R/eta;
    pi_new = projection_onto_U(pi_new, Box);
    
    % update pi for next iteration
    pi = pi_new;
    
end


% Check if pi^N is solution:
lambda = zeros(K,T);
for l = 1:K
    for t = 1:T
        for k = 1:K
            lambda(l,t) = lambda(l,t) + L(k,l)*pi(k)*pHat(k,t);
        end
    end
end
R = zeros(1,K);
for k = 1:K
    mu_k = 0;
    for t = 1:T
        [~,lmin] = min(lambda(:,t));
        mu_k = mu_k + L(k,lmin(1))*pHat(k,t);
    end
    R(k) = mu_k;
end
r = dot(pi,R);
if r > rStar
    rStar = r;
    piStar = pi;
    Rstar = R;
end


end

