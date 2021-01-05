function [BCDMCfit] = fit_BC_DMC(XTrain,YRTrain,parameters,Box)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%=============INPUTS:
%   # XTrain     : real features associated to the learning samples.
%   # YRTrain    : real labels associated to the learning samples.
%   # Box        : Box Constraint.
%   # parameters : includes:
%                   - loss function L.
%                   - number of classes K.
%                   - number of iterations N for computing piStar.
%                   - discretization = other parameters for discretizing.
%                     the numeric features.
%                   - dispPlot = 1 if display plot of convergence, 0
%                     otherwise.
%============OUTPUTS:
%   # BCDMCfit : includes:
%                   - pHat see equation (14) in the paper.
%                   - Xcomb = profiles {x1,...,xT}.
%                   - T = number of profiles {x1,...,xT}.
%                   - piStar = least favorable priors over U.
%                   - centroids = used for discretizing test samples.
%--------------------------------------------------------------------------

fprintf('fit_BC_DMC...\n')

L = parameters.L;
K = parameters.K;
N = parameters.N;
dispPlot = parameters.dispPlot;

if dispPlot
    hFig = figure('name','fit_BC_DMC');
    set(0,'CurrentFigure',hFig)
    set(0,'defaultfigurecolor',[1 1 1]);
    get(0,'Factory');
    parameters.hFig = hFig;
    subplot(2,3,1:2)
end

piTrain = compute_pi(YRTrain,K);
[XQuant, parameters] = discretization_XTrain(XTrain,YRTrain,parameters);
[Xcomb,T] = compute_Xcomb(XQuant);
pHat = compute_pHat(Xcomb,XQuant,YRTrain,K);



if dispPlot
    set(0,'CurrentFigure',hFig)
    if K == 2
        subplot(2,3,3);
        R_DBC = plot_V(parameters,pHat,T,piTrain,Box);
    else
        subplot(2,3,3);
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
    end
end

[rStar, piStar, R_BCDMC, stock_r] = compute_piStar(pHat,piTrain,parameters,Box);

if dispPlot
    
    subplot(2,3,4)
    plot(1:N,stock_r,'-','LineWidth',1.5,'Color',[0 0.45 0.74]);
    legend('V(\pi^{(n)})')
    xlim([1 N])
    xlabel('Iteration n')
    grid on
    title('Convergence')
    set(gca, 'XScale', 'log')
    
    subplot(2,3,5)
    if K==2
        colormap(gca,[0.3 0.75 0.96; 0 1 0]);
        xticknames = {num2str(zeros(2,K))};
        for k = 1:K
            tmp = ['$\hat{R}_{' int2str(k) '} \left(\delta \right)$'];
            xticknames{k} = tmp;
        end
        bar([R_DBC' R_BCDMC'],'EdgeColor',[1 1 1]);
        set(gca, 'YGrid', 'on', 'XGrid', 'off')
        legend('DBC','BCDMC')
        title('Class-conditional Risks')
        set(gca,'TickLabelInterpreter', 'latex');
        set(gca,'XTick',1:K,'xticklabel',xticknames)
        xlim([0.5 K+0.5]);
        ylim([0 max(max(L))])
        drawnow
    else
        xticknames = {num2str(zeros(2,K))};
        for k = 1:K
            tmp = ['$\hat{R}_{' int2str(k) '} \left(\delta \right)$'];
            xticknames{k} = tmp;
        end
        bar(R_BCDMC,'FaceColor',[0 1 0],'EdgeColor',[0 1 0]);
        set(gca, 'YGrid', 'on', 'XGrid', 'off')
        legend('BCDMC')
        title('Class-conditional Risks')
        set(gca,'TickLabelInterpreter', 'latex');
        set(gca,'XTick',1:K,'xticklabel',xticknames)
        ylim([0 min(2*max(R_BCDMC), max(max(L)))])
        drawnow
    end
    
    subplot(2,3,6)
    colormap(gca,spring)
    PiePlot = pie(piStar);
    PieText = findobj(PiePlot,'Type','text');
    percentValues = get(PieText,'String');
    classlabels = {num2str(zeros(2,K))};
    title('$\pi^{\star}$','Interpreter','latex','FontSize',20)
    for k = 1:K
        tmp = strcat('C', int2str(k), '=', percentValues(k));
        classlabels{k} = tmp;
        PieText(k).String = strcat(classlabels{k});
    end
    set(PiePlot,'EdgeColor','none','LineStyle','none')
    drawnow
    
    if K == 2
        subplot(2,3,3);
        plot_V_piStar(parameters,pHat,T,piTrain,Box,piStar,R_BCDMC,rStar);
    end
    hold off
    fprintf('V(piStar) = %.4f\n', rStar);
    fprintf('r(piTrain,delta_BCDMC) = %.4f\n\n', dot(piTrain,R_BCDMC));
else
    fprintf('piStar = [ ');
    fprintf('%g ', piStar)
    fprintf('] \n');
    fprintf('V(piStar) = %.4f\n', rStar);
    fprintf('r(piTrain,delta_BCDMC) = %.4f\n', dot(piTrain,R_BCDMC));
    fprintf('R_BC_DMC = [ ');
    fprintf('%g ', R_BCDMC)
    fprintf('] \n\n');
end


BCDMCfit.L = L;
BCDMCfit.K = K;
BCDMCfit.Xcomb = Xcomb;
BCDMCfit.T = T;
BCDMCfit.pHat = pHat;
BCDMCfit.parameters = parameters;
BCDMCfit.piStar = piStar;


end

