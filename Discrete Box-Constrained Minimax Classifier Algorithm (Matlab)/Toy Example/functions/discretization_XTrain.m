function [XTrainQuant, parameters] = discretization_XTrain(XTrain,YRTrain,parameters)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # XTrain  : real features associated to the learning samples.
%   # YRTrain : real labels.
%   # K       : number of classed.
%   # L       : Loss function.
%   # discretization : parameters for discretization:
%                       # discretization.method = 'none'
%                       # discretization.method = 'kmeans'
%======OUTPUTS:
%   # XTrainQuant : discretized features associated to the learning samples
%   # discretization : parameters for discretization (example: centroids)
%--------------------------------------------------------------------------

% discretization.method = 'none';
if strcmp(parameters.discretizationmethod,'none')
    m = size(XTrain,1);
    d = size(XTrain,2);
    for j = 1:d
        if min(XTrain(:,j)) < 1
            XTrain(:,j) = XTrain(:,j) + (1+min(XTrain(:,j)))*ones(m,1);
        end
    end
    XTrainQuant = XTrain;
end


% discretization.method = 'kmeans';
if strcmp(parameters.discretizationmethod,'kmeans')
    [XTrainQuant,Centroids,Tquant] = DiscretizationKmeans(XTrain,YRTrain,parameters);
    parameters.centroids = Centroids;
    parameters.T = Tquant;
end


end

