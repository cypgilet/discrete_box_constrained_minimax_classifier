function [XTrainQuant, Centroids, Tquant] = kmeansDiscretization(XTrain, YRTrain, K, L, eps_generalization_error)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # XTrain  : real features associated to the learning samples.
%   # YRTrain : real labels.
%   # K       : number of classed.
%   # L       : Loss function.
%   # eps_generalization_error : bound for generalisaton error for DBC.
%======OUTPUTS:
%   # XTrainQuant : discretized features associated to the learning samples
%   # Centroids   : used for discretizing test samples.
%   # Tquant      : number of profiles (centroids) {x1,...,xT}.
%--------------------------------------------------------------------------


mTrain = size(XTrain,1);
propTrain = 0.9;
[stockfolds] = createfolds(mTrain,propTrain);
m = 8*size(stockfolds,2) + min(sum(stockfolds>0,2));

if m <= 200
    nbCluster = [K:10:m-10 m];
end
if m > 200 && m <= 400
    nbCluster = [K:10:150, 200:50:m-50 m];
end
if m > 400 && m <= 1000
    nbCluster = [K:10:150, 200:50:400 500:100:m-100 m];
end
if m > 1000 && m <= 3000
    nbCluster = [K:10:150, 200:50:400 500:100:1000 1500:500:m-500 m];
end
if m > 3000
    nbCluster = [K 50 100 200:200:1000 2000 3000];
end

    

stockRiskBayesTrain = zeros(size(stockfolds,1),size(nbCluster,2));
stockRiskBayesValidSet = zeros(size(stockfolds,1),size(nbCluster,2));

for f = 1:size(stockfolds,1)
    fprintf('-> eval centroids: subfold f = %i/%i,   ',[f,size(stockfolds,1)])
    folds_f = stockfolds;
    indTest = folds_f(f,:); indTest(indTest==0) = [];
    folds_f(f,:) = [];
    indTrain = reshape(folds_f,1,size(folds_f,1)*size(folds_f,2));
    indTrain(indTrain==0) = [];
    YRTrainFold = YRTrain(indTrain);
    XTrainFold = XTrain(indTrain,:);
    YRValid = YRTrain(indTest);
    XValid = XTrain(indTest,:);
    piTrain = zeros(1,K);
    for k = 1:K
        piTrain(k) = sum(YRTrainFold==k)/size(YRTrainFold,1);
    end
    piValid = zeros(1,K);
    for k = 1:K
        piValid(k) = sum(YRValid==k)/size(YRValid,1);
    end
    
    fprintf(' centroids T = ')
    for l = 1:size(nbCluster,2)
        T = nbCluster(l);
        fprintf('%i ',T)
        % Training step
        [XTrainQuant, Centroids] = kmeans(XTrainFold,T,'Replicates',2);
        [Xcomb,~] = compute_Xcomb(XTrainQuant);
        pHat = compute_pHat(Xcomb,XTrainQuant,YRTrainFold,K);
        [YtrainBayespi0] = delta_Bayes_discret(Xcomb,piTrain,pHat,XTrainQuant,K,L);
        [R,~] = compute_conditional_risks(YtrainBayespi0,YRTrainFold,K,L);
        stockRiskBayesTrain(f,l) = dot(piTrain,R);
        % Validation step
        XTestQuant = findCentroid(Centroids,XValid);
        YtestBayes = delta_Bayes_discret(Xcomb,piTrain,pHat,XTestQuant,K,L);
        [R,~] = compute_conditional_risks(YtestBayes,YRValid,K,L);
        stockRiskBayesValidSet(f,l) = dot(piValid,R);
    end
    fprintf('\n')
end


% Choice of number of centroids:
avRiskTrain = mean(stockRiskBayesTrain,1);
avRiskValid = mean(stockRiskBayesValidSet,1);

tab_generalization_error = abs(avRiskTrain-avRiskValid);
indT = find(tab_generalization_error <= eps_generalization_error);
if indT
    [~,indminTrain] = min(avRiskTrain(indT));
    Tquant = nbCluster(indT(indminTrain));
else
    [~,indT] = min(tab_generalization_error);
    Tquant = nbCluster(indT);
end


h = figure;
plot(nbCluster,mean(stockRiskBayesTrain,1),'-')
hold on
plot(nbCluster,mean(stockRiskBayesValidSet,1),'-')
hold on
plot([Tquant Tquant],[0 max(max(avRiskTrain), max(avRiskValid))],...
    'color',[0 1 1])
hold on
plot(nbCluster,mean(stockRiskBayesValidSet,1)+std(stockRiskBayesValidSet,1),'--')
hold on
plot(nbCluster,mean(stockRiskBayesTrain,1)+std(stockRiskBayesTrain,1),'--')
hold on
plot(nbCluster,mean(stockRiskBayesTrain,1)-std(stockRiskBayesTrain,1),'--')
hold on
plot(nbCluster,mean(stockRiskBayesValidSet,1)-std(stockRiskBayesValidSet,1),'--')
xlabel('T')
legend('Training set: average risk','Validation set: average risk',['Tquant = ' num2str(Tquant)])
grid on
ylim([0, max(max(avRiskTrain), max(avRiskValid))]);
set(h, 'Visible', 'off');
name = 'results/kmeans_quatization/kmeans_T_.png';
name = insertAfter(name,'results/kmeans_quatization/kmeans_T_',string(datetime('now')));
saveas(h,name);




[XTrainQuant, Centroids] = kmeans(XTrain,Tquant,'Replicates',2);




end

