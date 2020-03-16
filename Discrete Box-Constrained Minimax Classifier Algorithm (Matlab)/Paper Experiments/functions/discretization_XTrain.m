function [XTrainQuant, discretization] = discretization_XTrain(XTrain,YRTrain,K,L,discretization)
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
if strcmp(discretization.method,'none')
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
if strcmp(discretization.method,'kmeans')
    eps_generalization_error = discretization.eps_generalization_error;
    [XTrainQuant, Centroids, Tquant] = kmeansDiscretization(XTrain,...
        YRTrain, K, L, eps_generalization_error);
    discretization.centroids = Centroids;
    discretization.Tquant = Tquant;
end


end

