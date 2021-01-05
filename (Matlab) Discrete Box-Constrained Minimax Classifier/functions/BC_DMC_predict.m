function [Yhat] = BC_DMC_predict(XTest,BCDMCfit)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
% Classes predictions using the Discrete Box-Constrained Minimax Classifier

parameters = BCDMCfit.parameters;
XTestQuant = discretization_XTest(XTest,parameters);

K = BCDMCfit.K;
L = BCDMCfit.L;
piStar = BCDMCfit.piStar;
Xcomb = BCDMCfit.Xcomb;
pHat = BCDMCfit.pHat;

Yhat = delta_Bayes_discret(Xcomb,piStar,pHat,XTestQuant,K,L);

end