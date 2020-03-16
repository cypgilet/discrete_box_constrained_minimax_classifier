function [Yhat] = delta_Bayes_discret(Xcomb,pi,pHat,X,K,L)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # Xcomb : profiles {x1,...,xT}.
%   # pi    : priors.
%   # pHat  : see equation (14) in the paper.
%   # X     : discretized features for each samples.
%   # K     : number of classed.
%   # L     : Loss function.
%======OUTPUTS:
%   # Yhat  : predicted labels.
%--------------------------------------------------------------------------

Yhat = zeros(size(X,1),1);
d = size(X,2);

for i = 1:size(X,1)
    t = find(sum(X(i,:)==Xcomb,2)==d);
    lambda = zeros(K,1);
    for l = 1:K
        for k = 1:K
            lambda(l,1) = lambda(l,1) + L(k,l)*pi(k)*pHat(k,t);
        end
    end
    [~,lbar] = min(lambda);
    lbar = sort(lbar);
    Yhat(i) = lbar(1);
end


end

