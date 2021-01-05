function [ XTestQuant ] = discretization_XTest(XTest,parameters)
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
if strcmp(parameters.discretizationmethod,'none')
    XTestQuant = XTest;
end 

%discretization.method = 'kmeans';
if strcmp(parameters.discretizationmethod,'kmeans')
    Centroids = parameters.centroids;
    XTestQuant = findCentroid(Centroids,XTest);
end


end