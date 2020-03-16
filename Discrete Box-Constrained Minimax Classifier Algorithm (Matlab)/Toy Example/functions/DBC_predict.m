function [Yhat] = DBC_predict(XTest,DBCfit)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
% Classes predictions using the Discrete Bayes Classifier.

parameters = DBCfit.parameters;
XTestQuant = discretization_XTest(XTest,parameters);

K = DBCfit.K;
L = DBCfit.L;
piTrain = DBCfit.piTrain;
Xcomb = DBCfit.Xcomb;
pHat = DBCfit.pHat;

Yhat = delta_Bayes_discret(Xcomb,piTrain,pHat,XTestQuant,K,L);

end

