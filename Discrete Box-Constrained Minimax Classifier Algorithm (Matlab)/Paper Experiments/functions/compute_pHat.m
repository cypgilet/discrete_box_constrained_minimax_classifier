function [pHat] = compute_pHat(Xcomb,XTrain,YRTrain,K)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # Xcomb   : profiles {x1,...,xT}.
%   # YRTrain : real labels.
%   # XTrain  : discretized features for the learning samples.
%   # K       : number of classed.
%======OUTPUTS:
%   # pHat    : see equation (14) in the paper.
%--------------------------------------------------------------------------

d = size(XTrain,2);
T = size(Xcomb,1);
pHat = zeros(K,T);

for k = 1:K
    for t = 1:T
        pHat(k,t) = sum(sum(XTrain(YRTrain==k,:)==Xcomb(t,:),2)==d)/sum(YRTrain==k);
    end
end

end

