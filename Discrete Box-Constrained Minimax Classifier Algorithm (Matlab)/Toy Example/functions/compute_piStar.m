function [rStar,piStar,Rstar,stock_r] = compute_piStar(pHat,piTrain,parameters,Box)
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

L = parameters.L;
K = parameters.K;
N = parameters.N;
dispPlot = parameters.dispPlot;

T = parameters.T;

pi = piTrain;



rStar = 0;
piStar = pi;
Rstar = 0;
stock_r = zeros(1,N);

if dispPlot
    n_disp_n = [1:10 12:4:20 50 100:100:500 800:500:N N];
    stock_n_disp_n = zeros(1,size(n_disp_n,2));
    stock_r_disp_n = zeros(1,size(n_disp_n,2));
    q = 0;
end


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
    
    
    stock_r(n) = r;
    
    if dispPlot
        hFig = parameters.hFig;
        set(0,'CurrentFigure',hFig)
        if sum(n==n_disp_n)==1
            q = q+1;
            stock_n_disp_n(q) = n;
            stock_r_disp_n(q) = r;
            subplot(2,3,4:6)
            plot(stock_n_disp_n(1:q),stock_r_disp_n(1:q),'-','LineWidth',1.5,'Color',[0 0.45 0.74]);
            legend('V(\pi^{(n)})')
            xlim([1 N])
            xlabel('Iteration n')
            grid on
            title('Maximization of V over U')
            set(gca, 'XScale', 'log')
            drawnow limitrate
        end
    end
    
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

