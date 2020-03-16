function [DBCfit] = fit_DBC(XTrain,YRTrain,parameters)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%=============INPUTS:
%   # XTrain     : real features associated to the learning samples.
%   # YRTrain    : real labels associated to the learning samples.
%   # parameters : includes:
%                   - loss function L.
%                   - number of classes K.
%                   - discretization = other parameters for discretizing.
%                     the numeric features.
%                   - dispPlot = 1 if display plot of convergence, 0
%                     otherwise.
%============OUTPUTS:
%   # DBCfit : includes:
%                   - pHat see equation (14) in the paper.
%                   - Xcomb = profiles {x1,...,xT}.
%                   - T = number of profiles {x1,...,xT}.
%                   - piTrain = Class proportions of the training set.
%                   - centroids = used for discretizing test samples.
%--------------------------------------------------------------------------

fprintf('fit_DBC...\n')

L = parameters.L;
K = parameters.K;
dispPlot = parameters.dispPlot;

if dispPlot
    hFig = figure('name','fit_DBC');
    set(0,'CurrentFigure',hFig)
    set(0,'defaultfigurecolor',[1 1 1]);
    get(0,'Factory');
    parameters.hFig = hFig;
    subplot(2,3,1:3)
end

piTrain = compute_pi(YRTrain,K);
[XQuant, parameters] = discretization_XTrain(XTrain,YRTrain,parameters);
[Xcomb,T] = compute_Xcomb(XQuant);
pHat = compute_pHat(Xcomb,XQuant,YRTrain,K);



lambda = zeros(K,T);
for l = 1:K
    for t = 1:T
        for k = 1:K
            lambda(l,t) = lambda(l,t) + L(k,l)*piTrain(k)*pHat(k,t);
        end
    end
end

R_DBC = zeros(1,K);
for k = 1:K
    mu_k = 0;
    for t = 1:T
        [~,lmin] = min(lambda(:,t));
        mu_k = mu_k + L(k,lmin(1))*pHat(k,t);
    end
    R_DBC(k) = mu_k;
end

r_DBC = dot(piTrain,R_DBC);



if dispPlot
    set(0,'CurrentFigure',hFig)
    
    subplot(2,3,4);
    colormap(gca,spring)
    PiePlotTrain = pie(piTrain);
    PieText = findobj(PiePlotTrain,'Type','text');
    percentValues = get(PieText,'String');
    classlabels = {num2str(zeros(2,K))};
    title('$\hat{\pi}$','Interpreter','latex','FontSize',20)
    for k = 1:K
        tmp = strcat('C', int2str(k), '=', percentValues(k));
        classlabels{k} = tmp;
        PieText(k).String = strcat(classlabels{k});
    end
    set(PiePlotTrain,'EdgeColor','none','LineStyle','none')
    drawnow
    
    if K == 2
        subplot(2,3,5);
        plot_V(parameters,pHat,T,piTrain,[zeros(K,1), ones(K,1)])
    end
        
    subplot(2,3,6);
    xticknames = {num2str(zeros(2,K))};
    for k = 1:K
        tmp = ['$\hat{R}_{' int2str(k) '} \left(\delta \right)$'];
        xticknames{k} = tmp;
    end
    bar(R_DBC,'FaceColor',[0 1 0],'EdgeColor',[0 1 0]);
    set(gca, 'YGrid', 'on', 'XGrid', 'off')
    legend('DBC')
    title('Class-conditional Risks')
    set(gca,'TickLabelInterpreter', 'latex');
    set(gca,'XTick',1:K,'xticklabel',xticknames)
    if K == 2
        ylim([0 max(max(L))])
    else
        ylim([0 min(2*max(R_DBC), max(max(L)))])
    end
    drawnow
end


fprintf('r(piTrain,delta_DBC) = %.4f\n\n', r_DBC);


DBCfit.L = L;
DBCfit.K = K;
DBCfit.Xcomb = Xcomb;
DBCfit.T = T;
DBCfit.pHat = pHat;
DBCfit.parameters = parameters;
DBCfit.piTrain = piTrain;

end

