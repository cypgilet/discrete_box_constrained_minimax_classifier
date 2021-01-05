function [XTrainQuant, Centroids, Tquant] = DiscretizationKmeans(XTrain,YRTrain,parameters)
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

dispPlot = parameters.dispPlot;
eps_generalization_error = parameters.eps_generalization_error;
nbT = parameters.nbT;
L = parameters.L;
K = parameters.K;


mTrain = size(XTrain,1);
propTrain = 0.9;
[stockfolds] = createfolds(mTrain,propTrain);
m = 8*size(stockfolds,2) + min(sum(stockfolds>0,2));

% Establish all the numbers of centroids T to be tested
if m > 2000
    Tmax = 2000;
else
    Tmax = m;
end
a = 1/(nbT-1) * log(Tmax/K);
b = log(K) - a;
nbCluster = zeros(1,nbT);
for t = 1:nbT
    nbCluster(t) = round(exp(a*(t) + b));
end
        

F = size(stockfolds,1);
stock_r_DBC_Train = zeros(F,nbT);
stock_r_DBC_Valid = zeros(F,nbT);


if dispPlot == 0
    fprintf('Kmeans discretization...\n')
    fprintf('Number of centroids T = ');
else 
    hFig = parameters.hFig;
end

for l = 1:size(nbCluster,2)
    
    T = nbCluster(l);
    
    if dispPlot == 0
        fprintf('%i ', T);
    end
    
    for f = 1:F
        folds_f = stockfolds;
        indTest = folds_f(f,:); 
        indTest(indTest==0) = [];
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
        
        % Training step
        [XTrainQuant, Centroids] = kmeans(XTrainFold,T,'Replicates',2);
        [Xcomb,~] = compute_Xcomb(XTrainQuant);
        pHat = compute_pHat(Xcomb,XTrainQuant,YRTrainFold,K);
        [YtrainBayespi0] = delta_Bayes_discret(Xcomb,piTrain,pHat,XTrainQuant,K,L);
        [R,~] = compute_conditional_risks(YtrainBayespi0,YRTrainFold,K,L);
        stock_r_DBC_Train(f,l) = dot(piTrain,R);
        % Validation step
        XTestQuant = findCentroid(Centroids,XValid);
        YtestBayes = delta_Bayes_discret(Xcomb,piTrain,pHat,XTestQuant,K,L);
        [R,~] = compute_conditional_risks(YtestBayes,YRValid,K,L);
        stock_r_DBC_Valid(f,l) = dot(piValid,R);
    end
    
    if dispPlot == 1
        set(0,'CurrentFigure',hFig)
        av_r_Train = mean(stock_r_DBC_Train,1);
        av_r_Valid = mean(stock_r_DBC_Valid,1);
        std_r_Train = std(stock_r_DBC_Train,1);
        std_r_Valid = std(stock_r_DBC_Valid,1);
        plot(nbCluster(1:l),av_r_Train(1:l),'-','LineWidth',1.5,'Color',[0 0.45 0.74]);
        hold on
        plot(nbCluster(1:l),av_r_Valid(1:l),'-','LineWidth',1.5,'Color',[1 0.84 0]);
        hold on
        plot(nbCluster(1:l),av_r_Train(1:l)+std_r_Train(1:l),'--','LineWidth',1,'Color',[0 0.45 0.74]);
        hold on
        plot(nbCluster(1:l),av_r_Train(1:l)-std_r_Train(1:l),'--','LineWidth',1,'Color',[0 0.45 0.74]);
        hold on
        plot(nbCluster(1:l),av_r_Valid(1:l)+std_r_Valid(1:l),'--','LineWidth',1,'Color',[1 0.84 0]);
        hold on
        plot(nbCluster(1:l),av_r_Valid(1:l)-std_r_Valid(1:l),'--','LineWidth',1,'Color',[1 0.84 0]);
        grid on
        xlim([1 nbCluster(end)])
        xlabel(['Number of centroids T = ' num2str(T)])
        ylabel('Empirical global risk of errors')
        title('Computation of the number of centroids T')
        drawnow limitrate
    end
    
end


% Choice of number of centroids:
av_r_Train = mean(stock_r_DBC_Train,1);
av_r_Valid = mean(stock_r_DBC_Valid,1);

tab_generalization_error = abs(av_r_Train-av_r_Valid);
indT = find(tab_generalization_error <= eps_generalization_error);
if indT
    [~,indminTrain] = min(av_r_Train(indT));
    Tquant = nbCluster(indT(indminTrain));
else
    [~,indT] = min(tab_generalization_error);
    Tquant = nbCluster(indT);
end


%------------------Figure
if dispPlot == 1
    set(0,'CurrentFigure',hFig)
    axis([1 nbCluster(end) min([min(av_r_Train-std_r_Train), min(av_r_Valid-std_r_Valid)]) max([max(av_r_Train+std_r_Train), max(av_r_Valid+std_r_Valid)])])
    plot(nbCluster(1:l),av_r_Train(1:l),'-k.','MarkerSize',15,'LineWidth',1.5,'Color',[0 0.45 0.74]);
    drawnow
    hold on
    plot(nbCluster(1:l),av_r_Valid(1:l),'-k.','MarkerSize',15,'LineWidth',1.5,'Color',[1 0.84 0]);
    drawnow
    hold on
    line([Tquant Tquant],[0 max([max(av_r_Train+std_r_Train), max(av_r_Valid+std_r_Valid)])],'LineWidth',1.5,'color','green')
    drawnow
    hold on
    plot(nbCluster(1:l),av_r_Train(1:l)+std_r_Train(1:l),'--','LineWidth',1,'Color',[0 0.45 0.74]);
    drawnow
    hold on
    plot(nbCluster(1:l),av_r_Train(1:l)-std_r_Train(1:l),'--','LineWidth',1,'Color',[0 0.45 0.74]);
    drawnow
    hold on
    plot(nbCluster(1:l),av_r_Valid(1:l)+std_r_Valid(1:l),'--','LineWidth',1,'Color',[1 0.84 0]);
    drawnow
    hold on
    plot(nbCluster(1:l),av_r_Valid(1:l)-std_r_Valid(1:l),'--','LineWidth',1,'Color',[1 0.84 0]);
    drawnow
    grid on
    xlabel('Number of centroids T')
    ylabel('Empirical global risk of errors')
    legend('Training set: average risk','Validation set: average risk')
    title(['Computation of the number of centroids T:   T_{opt} = ' num2str(Tquant)])
else
    fprintf('\n');
    fprintf('Optimal number of centroids Topt = %i \n', Tquant);
end



[XTrainQuant, Centroids] = kmeans(XTrain,Tquant,'Replicates',2);




end

