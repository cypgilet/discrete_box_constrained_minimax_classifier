%==========================================================================
%============ DISCRETE BOX-CONSTRAINED MINIMAX CLASSIFIER =================
%==========================================================================
%
% This program is free software and is provided in the hope that it
% will be useful.
%
% Citation:
%
% @ARTICLE{BC_DMC,
%   author={C. {Gilet} and S. {Barbosa} and L. {Fillatre}},
%   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
%   title={Discrete Box-Constrained Minimax Classifier for Uncertain and Imbalanced Class Proportions}, 
%   year={2020},
%   doi={10.1109/TPAMI.2020.3046439}
% }
%
% On this script we present an example of how to use our algorithm on 2
% databases: Framingham (with K=2 classes) and Abalone (with K=5 classes).
% 
%--------------------------------------------------------------------------


clear;
close all;
addpath(genpath('functions'));
addpath(genpath('data_mat'));
rng(1);


%========================= LOAD DATA ======================================


% % %== FRAMINGHAM
load('framingham_data')  
L = ones(K,K)-eye(K);
Box = [0.6, 0.9; 0 1]; 
eps_generalization_error = 0.01;

% % %== ABALONE
% load('abalone_data.mat') 
% L = zeros(K,K);
% for k = 1:K
%     for l = 1:K
%         L(k,l) = (k-l)^2;
%     end
% end
% Box = [0 0.16; 0.5 0.78; 0.14 0.43; 0 0.2; 0 0.15];
% eps_generalization_error = 0.05;


%======================== DEFINE PARAMETERS ===============================

parameters.L = L;
parameters.K = K;
parameters.N = 1000;
parameters.discretizationmethod = 'kmeans';
parameters.eps_generalization_error = eps_generalization_error;
parameters.nbT = 20;
parameters.dispPlot = 1;

%====================== FIT DBC, BCDMC, DMC ===============================

XTrain = X;
YRTrain = YR;

% %== Discrete Bayes Classifier (DBC):
fprintf('fit Discrete Bayes Classifier \n')
DBCfit = fit_DBC(XTrain,YRTrain,parameters);

%== Discrete Minimax Classifier (DMC):
fprintf('fit Discrete Minimax Classifier \n')
simplex = [zeros(K,1), ones(K,1)];
DMCfit = fit_BC_DMC(XTrain,YRTrain,parameters,simplex);

% %== Discrete Box-Constrained Minimax Classifier (BCDMC):
fprintf('fit Discrete Box-Constrained Minimax Classifier \n')
BCDMCfit = fit_BC_DMC(XTrain,YRTrain,parameters,Box);


%=================== PREDICT with DBC, BCDMC, DMC =========================
% This part presents how to predict the classes of instances. For this
% example, we just consider that we want to evaluate the performances of
% the fitted classifiers on the training database. But given a new test 
% database, the way to predict the classes of the new test instances would
% remain similar to the following procedure.

XTest = X;
YRTest = YR;

% %== Discrete Bayes Classifier (DBC):
fprintf('Discrete Bayes Classifier predict \n')
YhatDBC = DBC_predict(XTest,DBCfit);

%== Discrete Minimax Classifier (DMC):
fprintf('Discrete Minimax Classifier predict \n')
YhatDMC = BC_DMC_predict(XTest,DMCfit);

% %== Discrete Box-Constrained Minimax Classifier (BCDMC):
fprintf('Discrete Box-Constrained Minimax Classifier predict \n')
YhatBCDMC = BC_DMC_predict(XTest,BCDMCfit);


%----------- Results Test set:

piTest = compute_pi(YRTest,K);

% %== Discrete Bayes Classifier (DBC):
[R_DBC_test,confmat_DBC_test] = compute_conditional_risks(YhatDBC,YRTest,K,L);
r_DBC_test = dot(piTest,R_DBC_test);
fprintf('r(piTest,delta_DBC) = %.4f\n', r_DBC_test);

%== Discrete Minimax Classifier (DMC):
[R_DMC_test,confmat_DMC_test] = compute_conditional_risks(YhatDMC,YRTest,K,L);
r_DMC_test = dot(piTest,R_DMC_test);
fprintf('r(piTest,delta_DMC) = %.4f\n', r_DMC_test);

% %== Discrete Box-Constrained Minimax Classifier (BCDMC):
[R_BCDMC_test,confmat_BCDMC_test] = compute_conditional_risks(YhatBCDMC,YRTest,K,L);
r_BCDMC_test = dot(piTest,R_BCDMC_test);
fprintf('r(piTest,delta_BCDMC) = %.4f\n', r_BCDMC_test);

