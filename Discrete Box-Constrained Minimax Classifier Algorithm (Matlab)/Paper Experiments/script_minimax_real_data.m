%==========================================================================
%============ DISCRETE BOX-CONSTRAINED MINIMAX CLASSIFIER =================
%==========================================================================
%
% This program was used to obtain the results in Table 2 of our paper:
% https://hal.archives-ouvertes.fr/hal-02296592
%
% Citation:
%
% @inproceedings{BCDMC_HAL,
%   title={Minimax Classifier with Box Constraint on the Priors},
%   author={Gilet, Cyprien and Barbosa, Susana and Fillatre, Lionel},
%   booktitle={HAL},
%   year={2019},
%   }
%
%--------------------------------------------------------------------------


clear;
close all;
addpath(genpath('functions'));
addpath(genpath('data_mat'));
rng(1);


%========================= LOAD DATA ======================================

load('framingham_data')   
%load('abalone_data.mat')     
%load('scaniaTrucks_data.mat') 
%load('DiabetesUCI_data')  
%load('NASApc3_data')   
%load('Satellite_data')        


%======================== LOSS FUNCTION ===================================

%== 0-1 loss:
L = ones(K,K)-eye(K);

%== Quadratic loss:
% L = zeros(K,K);
% for k = 1:K
%     for l = 1:K
%         L(k,l) = (k-l)^2;
%     end
% end

%== Scania Tucks database loss:
%L = [0 10; 500 0];




%=================== DISCRETIZATION PARAMETERS ============================

% Framingham:    eps = 0.01
% Abalone:       eps = 0.05
% ScaniaTrucks:  eps = 0.2
% DiabetesUCI:   eps = 0.05
% NASApc3:       eps = 0.03
% Satellite:     eps = 0.002

discretization.method = 'kmeans';
discretization.eps_generalization_error = 0.01;




%======================= GENERATION OF U ==================================

%== Size of each class:
Ctab = {num2str(zeros(2,K))};
size_class = zeros(2,K);
for k = 1:K
    size_class(1,k) = sum(YR==k);
    tmp = ['C' int2str(k)];
    Ctab{k} = tmp;
end
size_class(2,:) = size_class(1,:)/size(X,1);
disp_size_class = mat2dataset(size_class,'VarNames',Ctab,'ObsNames',...
    {'Number of samples','priors'});
display(disp_size_class);

%== Generation of U:
piTrainGlob = size_class(2,:);
[XQuant, discretization] = discretization_XTrain(X,YR,K,L,discretization);

%== Discrete Bayes Classifier (DBC):
[Xcomb,T] = compute_Xcomb(XQuant);
pHat = compute_pHat(Xcomb,XQuant,YR,K);
Yhat = delta_Bayes_discret(Xcomb,piTrainGlob,pHat,XQuant,K,L);
[R_DBC_glob,~] = compute_conditional_risks(Yhat,YR,K,L);
r_DBC_glob = dot(piTrainGlob,R_DBC_glob);

%== Discrete Minimax Classifier (DMC):
simplex = [zeros(K,1), ones(K,1)];
N = 1000;
[r_DMC_glob, piBarGlob, R_DMC_glob] = compute_piStar(...
    Xcomb,pHat,YR,K,N,L,simplex);

%== Box-constraint generation:
BoxGlob = zeros(K,2);
beta = 0.5;
rho = beta * norm(piTrainGlob-piBarGlob,'inf');
for k = 1:K
    BoxGlob(k,1) = max(piTrainGlob(k)-rho,0);
    BoxGlob(k,2) = min(piTrainGlob(k)+rho,1);
end

    


%====================== FOLDS GENERATION ==================================

%== Size for training:
propTrain = 0.9;

%== Generation samples's indices for each fold:
stockfolds = createfolds(size(X,1),propTrain);
nbFolds = size(stockfolds,1);




%========================== SETTINGS ======================================

% Let considers the following classifiers delta:
%   # LR    : Logistic Regression
%   # DLR   : Discrete Logistic Regression
%   # RF    : Random Forest
%   # DRF   : Discrete Random Forest
%   # DBC   : Discrete Bayes Classifier
%   # DMC   : Discrete Minimax Classifier
%   # BCDMC : Box-constarined Discrete Minimax Classifier


%--------- At each iteration f of the crossavalidation procedure, stock:

%-stock tables for the training steps:
stock_time_discretization_training = zeros(nbFolds,1);

%-for all classifier delta:
%   # stock_r_delta_train : global risk of errors of delta for training set
%   # stock_r_delta_test  : global risk of errors of delta for the test set
%   # stock_R_delta_train : conditional risk of errors for the training set
%   # stock_R_delta_test  : conditional risk of errors for the test set
%   # stock_confmat_delta_train : confusion matrix for the training set
%   # stock_confmat_delta_test  : confusion matrix for the test set
%   # stock_time_delta_train    : time for fitting the classifier delta

stock_r_RF_train = zeros(nbFolds,1);
stock_r_LR_train = zeros(nbFolds,1);
stock_r_DRF_train = zeros(nbFolds,1);
stock_r_DLR_train = zeros(nbFolds,1);
stock_r_DBC_train = zeros(nbFolds,1);
stock_r_DMC_train = zeros(nbFolds,1);
stock_r_BCDMC_train = zeros(nbFolds,1);

stock_r_LR_test = zeros(nbFolds,1);
stock_r_RF_test = zeros(nbFolds,1);
stock_r_DLR_test = zeros(nbFolds,1);
stock_r_DForest_test = zeros(nbFolds,1);
stock_r_DBC_test = zeros(nbFolds,1);
stock_r_DMC_test = zeros(nbFolds,1);
stock_r_BCDMC_test = zeros(nbFolds,1);

stock_R_LR_train = zeros(nbFolds,K);
stock_R_RF_train = zeros(nbFolds,K);
stock_R_DLR_train = zeros(nbFolds,K);
stock_R_DRF_train = zeros(nbFolds,K);
stock_R_DBC_train = zeros(nbFolds,K);
stock_R_DMC_train = zeros(nbFolds,K);
stock_R_BCDMC_train = zeros(nbFolds,K);

stock_R_LR_test = zeros(nbFolds,K);
stock_R_RF_test = zeros(nbFolds,K);
stock_R_DLR_test = zeros(nbFolds,K);
stock_R_DRF_test = zeros(nbFolds,K);
stock_R_DBC_test = zeros(nbFolds,K);
stock_R_DMC_test = zeros(nbFolds,K);
stock_R_BCDMC_test = zeros(nbFolds,K);

stock_confmat_LR_train = zeros(K,K,nbFolds);
stock_confmat_RF_train = zeros(K,K,nbFolds);
stock_confmat_DLR_train = zeros(K,K,nbFolds);
stock_confmat_DRF_train = zeros(K,K,nbFolds);
stock_confmat_DBC_train = zeros(K,K,nbFolds);
stock_confmat_DMC_train = zeros(K,K,nbFolds);
stock_confmat_BCDMC_train = zeros(K,K,nbFolds);

stock_confmat_LR_test = zeros(K,K,nbFolds);
stock_confmat_RF_test = zeros(K,K,nbFolds);
stock_confmat_DLR_test = zeros(K,K,nbFolds);
stock_confmat_DRF_test = zeros(K,K,nbFolds);
stock_confmat_DBC_test = zeros(K,K,nbFolds);
stock_confmat_DMC_test = zeros(K,K,nbFolds);
stock_confmat_BCDMC_test = zeros(K,K,nbFolds);

stock_time_LR_train = zeros(1,nbFolds);
stock_time_RF_train = zeros(1,nbFolds);
stock_time_DLR_train = zeros(1,nbFolds);
stock_time_DRF_train = zeros(1,nbFolds);
stock_time_DBC_train = zeros(1,nbFolds);
stock_time_DMC_train = zeros(1,nbFolds);
stock_time_BCDMC_train = zeros(1,nbFolds);

%-for the 2 minimax classifiers DMC, BCDMC:
%   # stock_rbar    : maximum of the function V over the simplex S
%   # stock_rstar   : maximum of V over the Box-Constrained simplex U
%   # stock_Rbar    : estimated conditional risk for the classifier DMC
%   # stock_Rstar   : estimated conditional risk for the classifier BCDMC

stock_rbar = zeros(nbFolds,1);
stock_rstar = zeros(nbFolds,1);
stock_Rbar = zeros(nbFolds,K);
stock_Rstar = zeros(nbFolds,K);

%-the class proportions:
%   # stock_piTrain : class proportions of the training set
%   # stock_piPrime : class proportions of the test set
%   # stock_piBar   : priors which maximize V over the simplex S
%   # stock_piStar  : priors which maximize V over U

stock_piTrain = zeros(nbFolds,K);
stock_piPrime = zeros(nbFolds,K);
stock_piBar = zeros(nbFolds,K);
stock_piStar = zeros(nbFolds,K);

%-for evaluating robustness over U when piPrime differs from piTrain:
%   # nbpiTest             : number of class proportions to be tested
%   # STOCK_PITEST_U       : class proportions for each test subset
%   # STOCK_SIZE_TEST_t    : size of each test subset
%   # stock_r_delta_test_U : class proportions in the training set

% Generation of piTest uniformly over U:
nbpiTest = 1000;
mu = 1;
STOCK_PITEST_U = zeros(nbpiTest,K);
s = 0;
while s < nbpiTest
    piTest = exprnd(mu,1,K);
    piTest = piTest/sum(piTest);
    check_U = 0;
    for k = 1:K
        if piTest(k) >= BoxGlob(k,1) && piTest(k) <= BoxGlob(k,2)
            check_U = check_U + 1;
        end
    end
    if check_U == K
        s = s+1;
        STOCK_PITEST_U(s,:) = piTest;
    end
end
STOCK_SIZE_TEST_t = zeros(nbFolds,nbpiTest);
stock_r_LR_test_U = zeros(nbFolds,nbpiTest);
stock_r_RF_test_U = zeros(nbFolds,nbpiTest);
stock_r_DLR_test_U = zeros(nbFolds,nbpiTest);
stock_r_DRF_test_U = zeros(nbFolds,nbpiTest);
stock_r_DBC_test_U = zeros(nbFolds,nbpiTest);
stock_r_DMC_test_U = zeros(nbFolds,nbpiTest);
stock_r_BCDMC_test_U = zeros(nbFolds,nbpiTest);


%-Experiment when the radius of the box-constraint changes:
%   # beta_f               : parameters beta to be tested
%   # stock_rStar_beta     : maximum of V over U_beta
%   # stock_psi_Rstar_beta : psi(BCDMC) associated to U_beta

beta_f = 0.1:0.1:1;
stock_rStar_beta = zeros(nbFolds,size(beta_f,2));
stock_psi_Rstar_beta = zeros(nbFolds,size(beta_f,2));






%======================== FOLDS PROCESSING ================================

for f = 1:nbFolds
    
    fprintf('----------------- Processing fold: %i -----------------\n',f);
    
    %---------------- BUILD TRAINING AND TEST SETS:
    folds_f = stockfolds;
    indTest = folds_f(f,:); indTest(indTest==0) = [];
    folds_f(f,:) = [];
    indTrain = reshape(folds_f,1,size(folds_f,1)*size(folds_f,2));
    indTrain(indTrain==0) = [];
    
    YRTrain = YR(indTrain);
    XTrain = X(indTrain,:);
    YRTest = YR(indTest);
    XTest = X(indTest,:);
    
    fprintf('processing discretization...\n');
    tic
    [XTrainQuant, discretization] = discretization_XTrain(XTrain,...
        YRTrain,K,L,discretization);
    stock_time_discretization_training(f) = toc;
    
    
    %---------------- LEARNING STEP:
    fprintf('----- LEARNING -------\n');
    
    piTrain = zeros(1,K);
    for k = 1:K
        piTrain(k) = sum(YRTrain==k)/size(YRTrain,1);
    end
    stock_piTrain(f,:) = piTrain;
    
    % Logistic Regression (LR):
    tic
    BetaLR = mnrfit(XTrain,YRTrain);
    stock_time_LR_train(f) = toc;
    pTrain = mnrval(BetaLR,XTrain);
    [~,indMax] = max(pTrain,[],2);
    Yhat = indMax;
    [R,confmat] = compute_conditional_risks(Yhat,YRTrain,K,L);
    stock_confmat_LR_train(:,:,f) = confmat;
    stock_R_LR_train(f,:) = R;
    stock_r_LR_train(f) = dot(piTrain,R);
    fprintf('risk training LR %.2f \n', stock_r_LR_train(f));
    
    % Random Forest (RF):
    tic
    Forestmdl = fitcensemble(XTrain,YRTrain,'Cost',L, ...
        'OptimizeHyperparameters',{'NumLearningCycles','LearnRate',...
        'MaxNumSplits'});
    stock_time_RF_train(f) = toc;
    Yhat = predict(Forestmdl,XTrain);
    [R,confmat] = compute_conditional_risks(Yhat,YRTrain,K,L);
    stock_confmat_RF_train(:,:,f) = confmat;
    stock_R_RF_train(f,:) = R;
    stock_r_RF_train(f) = dot(piTrain,R);
    fprintf('risk training Random Forest: %.2f \n', ...
        stock_r_RF_train(f));
    
    % Discrete Logistic Regression (DLR):
    tic
    BetaLRDiscrete = mnrfit(XTrainQuant,YRTrain);
    stock_time_DLR_train(f) = toc;
    pTrain = mnrval(BetaLRDiscrete,XTrainQuant);
    [~,indMax] = max(pTrain,[],2);
    Yhat = indMax;
    [R,confmat] = compute_conditional_risks(Yhat,YRTrain,K,L);
    stock_confmat_DLR_train(:,:,f) = confmat;
    stock_R_DLR_train(f,:) = R;
    stock_r_DLR_train(f) = dot(piTrain,R);
    fprintf('risk training Discrete LR %.2f \n', stock_r_DLR_train(f));
    
    % Discrete Random Forest (DRF):
    tic
    ForestmdlDiscrete = fitcensemble(XTrainQuant,YRTrain,'Cost',L, ...
        'OptimizeHyperparameters',{'NumLearningCycles','LearnRate',...
        'MaxNumSplits'});
    stock_time_DRF_train(f) = toc;
    Yhat = predict(ForestmdlDiscrete,XTrainQuant);
    [R,confmat] = compute_conditional_risks(Yhat,YRTrain,K,L);
    stock_confmat_DRF_train(:,:,f) = confmat;
    stock_R_DRF_train(f,:) = R;
    stock_r_DRF_train(f) = dot(piTrain,R);
    fprintf('risk training Discrete Random Forest: %.2f \n', ...
        stock_r_DRF_train(f));
    
    % Discrete Bayes classifier (DBC):
    tic
    [Xcomb,T] = compute_Xcomb(XTrainQuant);
    pHat = compute_pHat(Xcomb,XTrainQuant,YRTrain,K);
    stock_time_DBC_train(f) = toc;
    [Yhat] = delta_Bayes_discret(Xcomb,piTrain,pHat,XTrainQuant,K,L);
    [R,confmat] = compute_conditional_risks(Yhat,YRTrain,K,L);
    stock_confmat_DBC_train(:,:,f) = confmat;
    stock_R_DBC_train(f,:) = R;
    stock_r_DBC_train(f) = dot(piTrain,R);
    fprintf('risk training DBC: %.2f \n', ...
        stock_r_DBC_train(f));
    
    % Discrete Minimax Classifier (DMC):
    simplex = [zeros(K,1), ones(K,1)];
    N = 1000;
    tic
    [Xcomb,~] = compute_Xcomb(XTrainQuant);
    pHat = compute_pHat(Xcomb,XTrainQuant,YRTrain,K);
    [rbar,piBar,Rbar] = compute_piStar(Xcomb,pHat,YRTrain,K,N,L,simplex);
    stock_time_DMC_train(f) = toc;
    stock_Rbar(f,:) = Rbar;
    stock_rbar(f) = rbar;
    stock_piBar(f,:) = piBar;
    [Yhat] = delta_Bayes_discret(Xcomb,piBar,pHat,XTrainQuant,K,L);
    [R,confmat] = compute_conditional_risks(Yhat,YRTrain,K,L);
    stock_confmat_DMC_train(:,:,f) = confmat;
    stock_R_DMC_train(f,:) = R;
    stock_r_DMC_train(f) = dot(piTrain,R);
    fprintf('risk training DMC: %.2f \n', stock_r_DMC_train(f));
    
    % Discrete Box-Constrained Minimax Classifier (BCDMC):
    tic
    [Xcomb,~] = compute_Xcomb(XTrainQuant);
    pHat = compute_pHat(Xcomb,XTrainQuant,YRTrain,K);
    [rStar,piStar,Rstar] = compute_piStar(Xcomb,pHat,YRTrain,K,N,L,BoxGlob);
    stock_time_BCDMC_train(f) = toc;
    stock_rstar(f) = rStar;
    stock_piStar(f,:) = piStar;
    stock_Rstar(f,:) = Rstar;
    Yhat = delta_Bayes_discret(Xcomb,piStar,pHat,XTrainQuant,K,L);
    [R,confmat] = compute_conditional_risks(Yhat,YRTrain,K,L);
    stock_confmat_BCDMC_train(:,:,f) = confmat;
    stock_R_BCDMC_train(f,:) = R;
    stock_r_BCDMC_train(f) = dot(piTrain,R);
    fprintf('risk training BCDMC: %.2f \n', stock_r_BCDMC_train(f));
    
    
    %------------------ TEST STEP:
    fprintf('----- TEST STEP -------\n');
    
    piPrime = zeros(1,K);
    for k = 1:K
        piPrime(k) = sum(YRTest==k)/size(YRTest,1);
    end
    stock_piPrime(f,:) = piPrime;
    
    XTestQuant = discretization_XTest(XTest,discretization);
    
    % Logistic Regression:
    pTest = mnrval(BetaLR,XTest);
    [~,indMax] = max(pTest,[],2);
    Yhat = indMax;
    [R,confmat] = compute_conditional_risks(Yhat,YRTest,K,L);
    stock_confmat_LR_test(:,:,f) = confmat;
    stock_R_LR_test(f,:) = R;
    stock_r_LR_test(f) = dot(piPrime,R);
    
    % Random Forest
    Yhat = predict(Forestmdl,XTest);
    [R,confmat] = compute_conditional_risks(Yhat,YRTest,K,L);
    stock_confmat_RF_test(:,:,f) = confmat;
    stock_R_RF_test(f,:) = R;
    stock_r_RF_test(f) = dot(piPrime,R);
    
    % Discrete Logistic Regression:
    pTest = mnrval(BetaLRDiscrete,XTestQuant);
    [~,indMax] = max(pTest,[],2);
    Yhat = indMax;
    [R,confmat] = compute_conditional_risks(Yhat,YRTest,K,L);
    stock_confmat_DLR_test(:,:,f) = confmat;
    stock_R_DLR_test(f,:) = R;
    stock_r_DLR_test(f) = dot(piPrime,R);
    
    % Discrete Random Forest
    Yhat = predict(ForestmdlDiscrete,XTestQuant);
    [R,confmat] = compute_conditional_risks(Yhat,YRTest,K,L);
    stock_confmat_DRF_test(:,:,f) = confmat;
    stock_R_DRF_test(f,:) = R;
    stock_r_DForest_test(f) = dot(piPrime,R);
    
    % Discrete Bayes classifier (DBC):
    Yhat = delta_Bayes_discret(Xcomb,piTrain,pHat,XTestQuant,K,L);
    [R,confmat] = compute_conditional_risks(Yhat,YRTest,K,L);
    stock_confmat_DBC_test(:,:,f) = confmat;
    stock_R_DBC_test(f,:) = R;
    stock_r_DBC_test(f) = dot(piPrime,R);
    
    % Discrete Minimax Classifier (DMC):
    Yhat = delta_Bayes_discret(Xcomb,piBar,pHat,XTestQuant,K,L);
    [R,confmat] = compute_conditional_risks(Yhat,YRTest,K,L);
    stock_confmat_DMC_test(:,:,f) = confmat;
    stock_R_DMC_test(f,:) = R;
    stock_r_DMC_test(f) = dot(piPrime,R);
    
    % Discrete Box-Constrained Minimax Classifier (BCDMC):
    Yhat = delta_Bayes_discret(Xcomb,piStar,pHat,XTestQuant,K,L);
    [R,confmat] = compute_conditional_risks(Yhat,YRTest,K,L);
    stock_confmat_BCDMC_test(:,:,f) = confmat;
    stock_R_BCDMC_test(f,:) = R;
    stock_r_BCDMC_test(f) = dot(piPrime,R);
    
    
    %----------------- Robustness over U when piPrime differs from piTrain:
    for s = 1:nbpiTest
        fprintf('fold f = %i - risks training: LR=%.2f|RF=%.2f|DLR=%.2f|DRF=%.2f|DBC=%.2f|DMC=%.2f|BCDMC=%.2f| -  piTest over U:  ',...
            [f,stock_r_LR_train(f),stock_r_RF_train(f),stock_r_DLR_train(f),stock_r_DRF_train(f),stock_r_DBC_train(f),stock_r_DMC_train(f),stock_r_BCDMC_train(f)])
        fprintf('processing test s = %i \n',s);
        
        % Select test samples from (XTest,YTest) satisfying piTest
        piTest = STOCK_PITEST_U(s,:);
        mT = size(XTest,1);
        stocEffk = zeros(1,K);
        for k = 1:K
            stocEffk(k) = sum(YRTest==k);
        end
        mprime = min(stocEffk);
        indTestpiTest = [];
        for k = 1:K
            mprime_k = max(1,floor(piTest(k)*mprime));
            indCk = find(YRTest==k);
            indTest_k = randperm(size(indCk,1),mprime_k);
            stock = indTestpiTest;
            indTestpiTest = [stock; indCk(indTest_k)];
        end
        YRTestU = YRTest(indTestpiTest);
        XTestU = XTest(indTestpiTest,:);
        XTestQuantU = XTestQuant(indTestpiTest,:);
        piTest = zeros(1,K);
        for k = 1:K
            piTest(k) = sum(YRTestU==k)/size(YRTestU,1);
        end
        STOCK_PITEST_U(s,:) = piTest;
        STOCK_SIZE_TEST_t(f,s) = size(YRTestU,1);
        
        % Logistic Regression:
        pTest = mnrval(BetaLR,XTestU);
        [~,indMax] = max(pTest,[],2);
        Yhat = indMax;
        [R,~] = compute_conditional_risks(Yhat,YRTestU,K,L);
        stock_r_LR_test_U(f,s) = dot(piTest,R);
        
        % Random Forest
        Yhat = predict(Forestmdl,XTestU);
        [R,~] = compute_conditional_risks(Yhat,YRTestU,K,L);
        stock_r_RF_test_U(f,s) = dot(piTest,R);
        
        % Discrete Logistic Regression:
        pTest = mnrval(BetaLRDiscrete,XTestQuantU);
        [~,indMax] = max(pTest,[],2);
        Yhat = indMax;
        [R,~] = compute_conditional_risks(Yhat,YRTestU,K,L);
        stock_r_DLR_test_U(f,s) = dot(piTest,R);
        
        % Discrete Random Forest
        Yhat = predict(ForestmdlDiscrete,XTestQuantU);
        [R,~] = compute_conditional_risks(Yhat,YRTestU,K,L);
        stock_r_DRF_test_U(f,s) = dot(piTest,R);
        
        % Discrete Bayes Classifier (DBC):
        Yhat = delta_Bayes_discret(Xcomb,piTrain,pHat,XTestQuantU,K,L);
        [R,~] = compute_conditional_risks(Yhat,YRTestU,K,L);
        stock_r_DBC_test_U(f,s) = dot(piTest,R);
        
        % Discrete Minimax Classifier (DMC):
        Yhat = delta_Bayes_discret(Xcomb,piBar,pHat,XTestQuantU,K,L);
        [R,~] = compute_conditional_risks(Yhat,YRTestU,K,L);
        stock_r_DMC_test_U(f,s) = dot(piTest,R);
        
        % Discrete Box-Constrained Minimax Classifier (BCDMC):
        Yhat = delta_Bayes_discret(Xcomb,piStar,pHat,XTestQuantU,K,L);
        [R,~] = compute_conditional_risks(Yhat,YRTestU,K,L);
        stock_r_BCDMC_test_U(f,s) = dot(piTest,R);
        
    end
    
    
    %------------------ Change radius beta of BOX_beta:
    fprintf('\n\n processing fold f = %i - BOX_beta EXPERIMENT \n',f);
    
    % Box_beta generation:
    for l = 1:size(beta_f,2)
        beta = beta_f(l);
        fprintf('processing fold f = %i - ',f);
        fprintf('learning step: beta = %i \n',beta);
        
        rho = beta * norm(piTrain-piBar,'inf');
        Box_f = zeros(K,2);
        for k = 1:K
            Box_f(k,1) = max(piTrain(k)-rho,0);
            Box_f(k,2) = min(piTrain(k)+rho,1);
        end
        
        [Xcomb,T] = compute_Xcomb(XTrainQuant);
        pHat = compute_pHat(Xcomb,XTrainQuant,YRTrain,K);
        [stock_rStar_beta(f,l),piStar_beta,Rstar] = ...
            compute_piStar(Xcomb,pHat,YRTrain,K,N,L,Box_f);
        stock_psi_Rstar_beta(f,l) = max(Rstar)-min(Rstar);  
    end
      
end





%=================== PSI TRAIN, PSI TEST, UPSILON =========================
%-For each classifier delta:
%   # stock_psi_delta_train : compute psi(delta) training for each fold f
%   # stock_psi_delta_test : compute psi(delta) for each test set of fold f

stock_psi_LR_train = zeros(nbFolds,1);
stock_psi_Forest_train = zeros(nbFolds,1);
stock_psi_Discrete_LR_train = zeros(nbFolds,1);
stock_psi_Discrete_Forest_train = zeros(nbFolds,1);
stock_psi_DBC_train = zeros(nbFolds,1);
stock_psi_DMC_train = zeros(nbFolds,1);
stock_psi_BCDMC_train = zeros(nbFolds,1);

stock_psi_LR_test = zeros(nbFolds,1);
stock_psi_Forest_test = zeros(nbFolds,1);
stock_psi_Discrete_LR_test = zeros(nbFolds,1);
stock_psi_Discrete_Forest_test = zeros(nbFolds,1);
stock_psi_DBC_test = zeros(nbFolds,1);
stock_psi_DMC_test = zeros(nbFolds,1);
stock_psi_BCDMC_test = zeros(nbFolds,1);

for f = 1:nbFolds
    %-psi train:
    stock_psi_LR_train(f) = max(stock_R_LR_train(f,:)) ...
        - min(stock_R_LR_train(f,:));
    stock_psi_Forest_train(f) = max(stock_R_RF_train(f,:)) ...
        - min(stock_R_RF_train(f,:));
    stock_psi_Discrete_LR_train(f) = max(stock_R_DLR_train(f,:))...
        - min(stock_R_DLR_train(f,:));
    stock_psi_Discrete_Forest_train(f) = ...
        max(stock_R_DRF_train(f,:))...
        - min(stock_R_DRF_train(f,:));
    stock_psi_DBC_train(f) = max(stock_R_DBC_train(f,:))...
        - min(stock_R_DBC_train(f,:));
    stock_psi_DMC_train(f) = max(stock_Rbar(f,:)) ...
        - min(stock_Rbar(f,:));
    stock_psi_BCDMC_train(f) = max(stock_Rstar(f,:)) ...
        - min(stock_Rstar(f,:));
    %-psi test:
    stock_psi_LR_test(f) = max(stock_R_LR_test(f,:)) ...
        - min(stock_R_LR_test(f,:));
    stock_psi_Forest_test(f) = max(stock_R_RF_test(f,:)) ...
        - min(stock_R_RF_test(f,:));
    stock_psi_Discrete_LR_test(f) = max(stock_R_DLR_test(f,:))...
        - min(stock_R_DLR_test(f,:));
    stock_psi_Discrete_Forest_test(f) = ...
        max(stock_R_DRF_test(f,:))...
        - min(stock_R_DRF_test(f,:));
    stock_psi_DBC_test(f) = max(stock_R_DBC_test(f,:))...
        - min(stock_R_DBC_test(f,:));
    stock_psi_DMC_test(f) = max(stock_R_DMC_test(f,:))...
        - min(stock_R_DMC_test(f,:));
    stock_psi_BCDMC_test(f) = max(stock_R_BCDMC_test(f,:))...
        - min(stock_R_BCDMC_test(f,:));
end




%==================== DISPLAY ALL RESULTS =================================

fprintf('\n\n------- ALL RESULTS ------\n\n');

fprintf('-- Priors data and Box-constraint: \n\n');
display(disp_size_class);
disp(BoxGlob);



fprintf('-- AVERAGE TRAINING STEP: \n\n');

fprintf('time LR = %.2f +- %.2f \n', [mean(stock_time_LR_train), std(stock_time_LR_train)]);
fprintf('time RF = %.2f +- %.2f\n', [mean(stock_time_RF_train), std(stock_time_RF_train)]);
fprintf('time DLR = %.2f +- %.2f\n', [mean(stock_time_DLR_train), std(stock_time_DLR_train)]);
fprintf('time DRF = %.2f +- %.2f\n', [mean(stock_time_DRF_train), std(stock_time_DRF_train)]);
fprintf('time DBC = %.2f +- %.2f\n', [mean(stock_time_DBC_train), std(stock_time_DBC_train)]);
fprintf('time BCDMC = %.2f +- %.2f\n', [mean(stock_time_BCDMC_train), std(stock_time_BCDMC_train)]);
fprintf('time DMC = %.2f +- %.2f\n\n', [mean(stock_time_DMC_train), std(stock_time_DMC_train)]);

fprintf('risk training LR %.2f +- %.2f \n', [mean(stock_r_LR_train),std(stock_r_LR_train)]);
fprintf('risk training RF: %.2f +- %.2f\n', [mean(stock_r_RF_train), std(stock_r_RF_train)]);
fprintf('risk training DLR %.2f +- %.2f \n', [mean(stock_r_DLR_train), std(stock_r_DLR_train)]);
fprintf('risk training DRF: %.2f +- %.2f\n', [mean(stock_r_DRF_train), std(stock_r_DRF_train)]);
fprintf('risk training DBC: %.2f +- %.2f\n', [mean(stock_r_DBC_train), std(stock_r_DBC_train)]);
fprintf('risk training BCDMC: %.2f +- %.2f \n', [mean(stock_r_BCDMC_train), std(stock_r_BCDMC_train)]);
fprintf('risk training DMC: %.2f +- %.2f\n', [mean(stock_r_DMC_train), std(stock_r_DMC_train)]);
fprintf('Average rStar: %.4f +- %.4f \n', [mean(stock_rstar), std(stock_rstar)]);
fprintf('Average rBar: %.4f +- %.4f\n\n', [mean(stock_rbar), std(stock_rbar)]);

fprintf('maxRk LR train = %.2f +- %.2f \n', [mean(max(stock_R_LR_train,[],2)),std(max(stock_R_LR_train,[],2))]);
fprintf('maxRk RF train = %.2f +- %.2f \n', [mean(max(stock_R_RF_train,[],2)),std(max(stock_R_RF_train,[],2))]);
fprintf('maxRk DLR train = %.2f +- %.2f \n', [mean(max(stock_R_DLR_train,[],2)),std(max(stock_R_DLR_train,[],2))]);
fprintf('maxRk DRF train = %.2f +- %.2f \n', [mean(max(stock_R_DRF_train,[],2)),std(max(stock_R_DRF_train,[],2))]);
fprintf('maxRk DBC train = %.2f +- %.2f \n', [mean(max(stock_R_DBC_train,[],2)),std(max(stock_R_DBC_train,[],2))]);
fprintf('maxRk BCDMC train = %.2f +- %.2f \n', [mean(max(stock_Rstar,[],2)),std(max(stock_Rstar,[],2))]);
fprintf('maxRk DMC train = %.2f +- %.2f \n\n', [mean(max(stock_Rbar,[],2)),std(max(stock_Rbar,[],2))]);

fprintf('psi LR train = %.2f +- %.2f \n', [mean(stock_psi_LR_train),std(stock_psi_LR_train)]);
fprintf('psi RF train = %.2f +- %.2f \n', [mean(stock_psi_Forest_train),std(stock_psi_Forest_train)]);
fprintf('psi DLR train = %.2f +- %.2f \n', [mean(stock_psi_Discrete_LR_train),std(stock_psi_Discrete_LR_train)]);
fprintf('psi DRF train = %.2f +- %.2f \n', [mean(stock_psi_Discrete_Forest_train),std(stock_psi_Discrete_Forest_train)]);
fprintf('psi DBC train = %.2f +- %.2f \n', [mean(stock_psi_DBC_train),std(stock_psi_DBC_train)]);
fprintf('psi BCDMC train = %.2f +- %.2f \n', [mean(stock_psi_BCDMC_train),std(stock_psi_BCDMC_train)]);
fprintf('psi DMC train = %.2f +- %.2f \n\n', [mean(stock_psi_DMC_train),std(stock_psi_DMC_train)]);




fprintf('-- RESULTS TEST STEP: \n\n');

fprintf('risk test LR %.2f +- %.2f \n', [mean(stock_r_LR_test), std(stock_r_LR_test)]);
fprintf('risk test RF: %.2f +- %.2f\n', [mean(stock_r_RF_test), std(stock_r_RF_test)]);
fprintf('risk test DLR %.2f +- %.2f \n', [mean(stock_r_DLR_test), std(stock_r_DLR_test)]);
fprintf('risk test DRF: %.2f +- %.2f\n', [mean(stock_r_DForest_test), std(stock_r_DForest_test)]);
fprintf('risk test DBC: %.2f +- %.2f\n', [mean(stock_r_DBC_test), std(stock_r_DBC_test)]);
fprintf('risk test BCDMC: %.2f +- %.2f \n', [mean(stock_r_BCDMC_test), std(stock_r_BCDMC_test)]);
fprintf('risk test DMC: %.2f +- %.2f\n\n', [mean(stock_r_DMC_test),std(stock_r_DMC_test)]);

fprintf('maxRk LR test = %.2f +- %.2f \n', [mean(max(stock_R_LR_test,[],2)),std(max(stock_R_LR_test,[],2))]);
fprintf('maxRk RF test = %.2f +- %.2f \n', [mean(max(stock_R_RF_test,[],2)),std(max(stock_R_RF_test,[],2))]);
fprintf('maxRk DLR test = %.2f +- %.2f \n', [mean(max(stock_R_DLR_test,[],2)),std(max(stock_R_DLR_test,[],2))]);
fprintf('maxRk DRF test = %.2f +- %.2f \n', [mean(max(stock_R_DRF_test,[],2)),std(max(stock_R_DRF_test,[],2))]);
fprintf('maxRk DBC test = %.2f +- %.2f \n', [mean(max(stock_R_DBC_test,[],2)),std(max(stock_R_DBC_test,[],2))]);
fprintf('maxRk BCDMC test = %.2f +- %.2f \n', [mean(max(stock_R_BCDMC_test,[],2)),std(max(stock_R_BCDMC_test,[],2))]);
fprintf('maxRk DMC test = %.2f +- %.2f \n\n', [mean(max(stock_R_DMC_test,[],2)),std(max(stock_R_DMC_test,[],2))]);

fprintf('psi LR test = %.2f +- %.2f \n', [mean(stock_psi_LR_test),std(stock_psi_LR_test)]);
fprintf('psi RF test = %.2f +- %.2f \n', [mean(stock_psi_Forest_test),std(stock_psi_Forest_test)]);
fprintf('psi DLR test = %.2f +- %.2f \n', [mean(stock_psi_Discrete_LR_test),std(stock_psi_Discrete_LR_test)]);
fprintf('psi DRF test = %.2f +- %.2f \n', [mean(stock_psi_Discrete_Forest_test),std(stock_psi_Discrete_Forest_test)]);
fprintf('psi DBC test = %.2f +- %.2f \n', [mean(stock_psi_DBC_test),std(stock_psi_DBC_test)]);
fprintf('psi BCDMC test = %.2f +- %.2f \n', [mean(stock_psi_BCDMC_test),std(stock_psi_BCDMC_test)]);
fprintf('psi DMC test = %.2f +- %.2f \n\n', [mean(stock_psi_DMC_test),std(stock_psi_DMC_test)]);





%========================= FIGURES ========================================

%--------------Figure Class proportions:

figure('Name','FIgure class proportions')
subplot(1,3,1)
pie(mean(stock_piTrain))
title('$\hat{\pi}$','interpreter','latex','FontSize',24)
subplot(1,3,2)
pie(mean(stock_piStar))
title('$\pi^{\star}$','interpreter','latex','FontSize',24)
subplot(1,3,3)
pie(mean(stock_piBar))
title('$\bar{\pi}$','interpreter','latex','FontSize',24)



%--------------Figure boxplot test steps in the cross-validation
figure('Name','Global risks in the cross-validation')

subplot(1,7,1)
h1 = boxplot(stock_r_LR_train,'colors',[0.2 0.5 0.8],'positions',1,'width',0.4);
set(h1,{'linew'},{2})
hold on
h2 = boxplot(stock_r_LR_test,'colors',[0.9 0.3 0.2],'positions',2,'width',0.4);
set(h2,{'linew'},{2})
set(gca, 'XTick',1:2, 'XTickLabel',{'Training','Test'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Global risks in the cross-validation')
grid on
title('LR')

subplot(1,7,2)
h1 = boxplot(stock_r_RF_train,'colors',[0.2 0.5 0.8],'positions',1,'width',0.4);
set(h1,{'linew'},{2})
hold on
h2 = boxplot(stock_r_RF_test,'colors',[0.9 0.3 0.2],'positions',2,'width',0.4);
set(h2,{'linew'},{2})
set(gca, 'XTick',1:2, 'XTickLabel',{'Training','Test'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Global risks in the cross-validation')
grid on
title('RF')

subplot(1,7,3)
h1 = boxplot(stock_r_DLR_train,'colors',[0.2 0.5 0.8],'positions',1,'width',0.4);
set(h1,{'linew'},{2})
hold on
h2 = boxplot(stock_r_DLR_test,'colors',[0.9 0.3 0.2],'positions',2,'width',0.4);
set(h2,{'linew'},{2})
set(gca, 'XTick',1:2, 'XTickLabel',{'Training','Test'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Global risks in the cross-validation')
grid on
title('DLR')

subplot(1,7,4)
h1 = boxplot(stock_r_DRF_train,'colors',[0.2 0.5 0.8],'positions',1,'width',0.4);
set(h1,{'linew'},{2})
hold on
h2 = boxplot(stock_r_DForest_test,'colors',[0.9 0.3 0.2],'positions',2,'width',0.4);
set(h2,{'linew'},{2})
set(gca, 'XTick',1:2, 'XTickLabel',{'Training','Test'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Global risks in the cross-validation')
grid on
title('DRF')

subplot(1,7,5)
h1 = boxplot(stock_r_DBC_train,'colors',[0.2 0.5 0.8],'positions',1,'width',0.4);
set(h1,{'linew'},{2})
hold on
h2 = boxplot(stock_r_DBC_test,'colors',[0.9 0.3 0.2],'positions',2,'width',0.4);
set(h2,{'linew'},{2})
set(gca, 'XTick',1:2, 'XTickLabel',{'Training','Test'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Global risks in the cross-validation')
grid on
title('DBC')

subplot(1,7,6)
h1 = boxplot(stock_r_BCDMC_train,'colors',[0.2 0.5 0.8],'positions',1,'width',0.4);
set(h1,{'linew'},{2})
hold on
h2 = boxplot(stock_r_BCDMC_test,'colors',[0.9 0.3 0.2],'positions',2,'width',0.4);
set(h2,{'linew'},{2})
set(gca, 'XTick',1:2, 'XTickLabel',{'Training','Test'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Global risks in the cross-validation')
grid on
title('BCDMC')

subplot(1,7,7)
h1 = boxplot(stock_r_DMC_train,'colors',[0.2 0.5 0.8],'positions',1,'width',0.4);
set(h1,{'linew'},{2})
hold on
h2 = boxplot(stock_r_DMC_test,'colors',[0.9 0.3 0.2],'positions',2,'width',0.4);
set(h2,{'linew'},{2})
set(gca, 'XTick',1:2, 'XTickLabel',{'Training','Test'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Global risks in the cross-validation')
grid on
title('DMC')




%--------------Figure conditional risks:

figure('Name','Figure class conditional risks')
maxYlimite = max([max(mean(stock_R_LR_train)), ...
    max(mean(stock_R_RF_train)), ...
    max(mean(stock_R_DLR_train)), ...
    max(mean(stock_R_DRF_train)), ...
    max(mean(stock_R_DBC_train)), max(mean(stock_R_DMC_train)), ...
    max(mean(stock_R_BCDMC_train)), max(mean(stock_R_LR_test)), ...
    max(mean(stock_R_RF_test)), ...
    max(mean(stock_R_DLR_test)), ...
    max(mean(stock_R_DRF_test)), ...
    max(mean(stock_R_DBC_test)), ...
    max(mean(stock_R_DMC_test)), max(mean(stock_R_BCDMC_test))]);

subplot(1,7,1)
bar([mean(stock_R_LR_train); mean(stock_R_LR_test)]')
ylim([0 maxYlimite]);
xlabel('R_k(\delta)')
grid on
legend('training','test')
title('LR')

subplot(1,7,2)
bar([mean(stock_R_RF_train); mean(stock_R_RF_test)]')
ylim([0 maxYlimite]);
xlabel('R_k(\delta)')
grid on
legend('training','test')
title('RF')

subplot(1,7,3)
bar([mean(stock_R_DLR_train); mean(stock_R_DLR_test)]')
ylim([0 maxYlimite]);
xlabel('R_k(\delta)')
grid on
legend('training','test')
title('DLR')

subplot(1,7,4)
bar([mean(stock_R_DRF_train); ...
    mean(stock_R_DRF_test)]')
ylim([0 maxYlimite]);
xlabel('R_k(\delta)')
grid on
legend('training','test')
title('DRF')

subplot(1,7,5)
bar([mean(stock_R_DBC_train); mean(stock_R_DBC_test)]')
ylim([0 maxYlimite]);
xlabel('R_k(\delta)')
grid on
legend('training','test')
title('DBC')

subplot(1,7,6)
bar([mean(stock_R_BCDMC_train); mean(stock_R_BCDMC_test)]')
ylim([0 maxYlimite]);
xlabel('R_k(\delta)')
grid on
legend('training','test')
title('BCDMC')

subplot(1,7,7)
bar([mean(stock_R_DMC_train); mean(stock_R_DMC_test)]')
ylim([0 maxYlimite]);
xlabel('R_k(\delta)')
grid on
legend('training','test')
title('DMC')





%--------------Figure Stability over U for each \pi^{(s)}, s = 1,...,1000:

av_LR_test_r = mean(stock_r_LR_test_U,1);
av_Forest_test_r = mean(stock_r_RF_test_U,1);
av_Discrete_LR_test_r = mean(stock_r_DLR_test_U,1);
av_Discrete_Forest_test_r = mean(stock_r_DRF_test_U,1);
av_Bayes_test_r = mean(stock_r_DBC_test_U,1);
av_bar_test_r = mean(stock_r_DMC_test_U,1);
av_star_test_r = mean(stock_r_BCDMC_test_U,1);

figure('Name','Figure Stability over U')
TAB_AV_ALL = [av_LR_test_r', av_Forest_test_r', av_Discrete_LR_test_r',...
    av_Discrete_Forest_test_r', av_Bayes_test_r', av_star_test_r',...
     av_bar_test_r'];
NAMES_TAB_AV_ALL = {'LR','RF','DLR','DRF','DBC','BCDMC','DMC'};
h = boxplot(TAB_AV_ALL);
set(h,{'linew'},{2})
set(gca,'TickLabelInterpreter','none','Fontweight','bold','Fontsize',20)
set(gca, 'XTick',1:size(TAB_AV_ALL,2), 'XTickLabel',NAMES_TAB_AV_ALL);
ax = gca;
grid
title('Dispersion of risks over U for each \pi^{(s)},  s = 1,...,1000')









%--------------Figure associated to changes in the Box-Constraint radius:

figure('Name','Changes in the Box-Constraint radius')

% V(piStar)
subplot(1,2,1)
plot([0,beta_f],mean([stock_r_DBC_train,stock_rStar_beta]),'Color',[1,0.84,0],'LineWidth',2)
hold on
plot([0,beta_f],mean(stock_r_DBC_train)*ones(1,size(beta_f,2)+1),'Color',[0.3,0.75,0.96],'LineWidth',2);
hold on
plot([0,beta_f],mean(stock_rbar)*ones(1,size(beta_f,2)+1),'Color',[0,0.45,0.74],'LineWidth',2);
hold on
plot([0,beta_f],mean([stock_r_DBC_train,stock_rStar_beta]) + ...
    std([stock_r_DBC_train,stock_rStar_beta]),'--','Color',[1,0.84,0],'LineWidth',1)
hold on
plot([0,beta_f],mean([stock_r_DBC_train,stock_rStar_beta]) - ...
    std([stock_r_DBC_train,stock_rStar_beta]),'--','Color',[1,0.84,0],'LineWidth',1)
hold on
plot([0,beta_f],(mean(stock_r_DBC_train)+std(stock_r_DBC_train)) *...
    ones(1,size(beta_f,2)+1),'--','Color',[0.3,0.75,0.96],'LineWidth',1);
hold on
plot([0,beta_f],(mean(stock_r_DBC_train)-std(stock_r_DBC_train)) *...
    ones(1,size(beta_f,2)+1),'--','Color',[0.3,0.75,0.96],'LineWidth',1);
hold on
plot([0,beta_f],(mean(stock_rbar)+std(stock_rbar)) *...
    ones(1,size(beta_f,2)+1),'--','Color',[0,0.45,0.74],'LineWidth',1);
hold on
plot([0,beta_f],(mean(stock_rbar)-std(stock_rbar)) *...
    ones(1,size(beta_f,2)+1),'--','Color',[0,0.45,0.74],'LineWidth',1);

legend('V(piStar)','V(piHat)','V(piBar)')
xlabel('Parameter \beta')
grid on
title('Changes in the Box-Constraint radius')


% psi(delta)
subplot(1,2,2)
plot([0,beta_f],mean([(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2)), stock_psi_Rstar_beta]),'Color',[1,0,0],'LineWidth',2);
hold on
plot([0,beta_f],mean(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2))*ones(1,size(beta_f,2)+1),'Color',[0.3,0.75,0.96],'LineWidth',2);
hold on
plot([0,beta_f],mean(max(stock_Rbar,[],2) - ...
    min(stock_Rbar,[],2))*ones(1,size(beta_f,2)+1),'Color',[0,0.45,0.74],'LineWidth',2);
hold on 
plot([0,beta_f],mean([(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2)), stock_psi_Rstar_beta]) + ...
    std([(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2)), stock_psi_Rstar_beta]),'--','Color',[1,0,0],'LineWidth',1);
hold on 
plot([0,beta_f],mean([(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2)), stock_psi_Rstar_beta]) - ...
    std([(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2)), stock_psi_Rstar_beta]),'--','Color',[1,0,0],'LineWidth',1);
hold on
plot([0,beta_f],(mean(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2)) - ...
    std(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2))) * ones(1,size(beta_f,2)+1),'--','Color',[0.3,0.75,0.96],'LineWidth',1);
hold on
plot([0,beta_f],(mean(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2)) + ...
    std(max(stock_R_DBC_train,[],2) - ...
    min(stock_R_DBC_train,[],2))) * ones(1,size(beta_f,2)+1),'--','Color',[0.3,0.75,0.96],'LineWidth',1);
hold on
plot([0,beta_f],(mean(max(stock_Rbar,[],2) - min(stock_Rbar,[],2)) + ...
    std(max(stock_Rbar,[],2) - min(stock_Rbar,[],2))) * ...
    ones(1,size(beta_f,2)+1),'--','Color',[0,0.45,0.74],'LineWidth',1);
hold on
plot([0,beta_f],max(0,(mean(max(stock_Rbar,[],2) - ...
    min(stock_Rbar,[],2)) - std(max(stock_Rbar,[],2) - ...
    min(stock_Rbar,[],2)))) * ones(1,size(beta_f,2)+1),'--','Color',[0,0.45,0.74],'LineWidth',1);

legend('psi(BCDMC)','psi(DBC)','psi(DMC)')
xlabel('Parameter \beta')
grid on
title('Changes in the Box-Constraint radius')





