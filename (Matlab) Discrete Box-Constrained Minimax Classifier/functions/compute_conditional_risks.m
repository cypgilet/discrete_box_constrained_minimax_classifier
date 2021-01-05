function [R,confmat] = compute_conditional_risks(Yhat,YR,K,L)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # Yhat : predicted labels.
%   # YR   : real labels.
%   # K    : number of classed.
%   # L    : Loss function.
%======OUTPUTS:
%   # R       : class-conditional risks.
%   # confmat : confusion matrix.
%--------------------------------------------------------------------------

confmat = zeros(K,K);
R = zeros(1,K);

for k = 1:K
    mk = sum(YR==k);
    if mk > 0
        for l = 1:K
            confmat(k,l) = sum(Yhat(YR==k)==l)/mk;
        end
        R(k) = dot(L(k,:),confmat(k,:));
    end
end

end

