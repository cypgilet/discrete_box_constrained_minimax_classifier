function [stockfolds] = createfolds(m,propTrain)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------

mL = round(m*propTrain);
mT = m-mL;


nbFolds = round(m/mT);

stockfolds = zeros(nbFolds,mT);
stockInd = 1:m;
for f = 1:nbFolds
    ind = randperm(length(stockInd),min(mT,length(stockInd)));
    stockfolds(f,1:length(ind)) = stockInd(ind);
    stockInd(ind) = [];
end



end

