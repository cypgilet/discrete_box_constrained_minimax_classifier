function [Xcomb, T] = compute_Xcomb(XTrain)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%  Compute Xcomb = profiles {x1,...,xT}.

d = size(XTrain,2);
Xcomb = (1:max(XTrain(:,1)))';
for j = 2:d
    Tn = size(Xcomb,1);
    q = max(XTrain(:,j));
    B = repmat(Xcomb,q,1);
    Xcomb = [B, zeros(size(B,1),1)];
    for s = 1:q
        Xcomb((s-1)*Tn+1:s*Tn,j) = repmat(s,Tn,1);
    end
end

T = size(Xcomb,1);

end

