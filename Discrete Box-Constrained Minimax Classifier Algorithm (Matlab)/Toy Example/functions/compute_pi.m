function [pi] = compute_pi(YR,K)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
% Compute the class proportions piTrain.

pi = zeros(1,K);
for k = 1:K
    pi(k) = sum(YR==k)/size(YR,1);
end

end

