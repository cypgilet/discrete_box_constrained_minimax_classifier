function [ XTestQuant ] = discretization_XTest(XTest,discretization)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # XTest  : real features associated to the test samples.
%   # discretization : parameters for discretization:
%                       # discretization.method = 'none'
%                       # discretization.method = 'kmeans'
%======OUTPUTS:
%   # XTestQuant : discretized features associated to the learning samples
%--------------------------------------------------------------------------

%discretization.method = 'none';
if strcmp(discretization.method,'none')
    XTestQuant = XTest;
end 

%discretization.method = 'kmeans';
if strcmp(discretization.method,'kmeans')
    Centroids = discretization.centroids;
    XTestQuant = findCentroid(Centroids,XTest);
end


end