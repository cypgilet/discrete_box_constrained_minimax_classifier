function [ XTestQuant ] = findCentroid(Centroids, XTest)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # XTest     : real features associated to the test samples.
%   # Centroids : coordinates of each centroid.
%======OUTPUTS:
%   # XTrainQuant : closest centroid associated to each test sample
%--------------------------------------------------------------------------

D = pdist2(XTest,Centroids,'euclidean');
[~,XTestQuant] = min(D,[],2);

end
