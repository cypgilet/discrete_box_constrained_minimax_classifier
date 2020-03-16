function [ V ] = plot_V_piStar(parameters,pHat,T,piTrain,Box,piStar,Rstar,rStar)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
% If K=2 or K=3, plot the function V over U.

K = parameters.K;
L = parameters.L;
hFig = parameters.hFig;
set(0,'CurrentFigure',hFig)

if K == 2
    pi1 = sort(unique([0:0.01:1 piTrain(1) piStar(1)]));
    pi2 = ones(size(pi1)) - pi1;
    V = zeros(size(pi1));
    div_DBC = zeros(size(pi1));
    div_BCDMC = zeros(size(pi1));
    pi1Box = [];
    V_U = [];
    div_DBC_U = [];
    div_BCDMC_U = [];
    
    for i = 1:size(pi1,2)
        pi = [pi1(i), pi2(i)];
        
        lambda = zeros(K,T);
        for l = 1:K
            for t = 1:T
                for k = 1:K
                    lambda(l,t) = lambda(l,t) + L(k,l)*pi(k)*pHat(k,t);
                end
            end
        end
        
        R = zeros(1,K);
        for k = 1:K
            mu_k = 0;
            for t = 1:T
                [~,lmin] = min(lambda(:,t));
                mu_k = mu_k + L(k,lmin(1))*pHat(k,t);
            end
            R(k) = mu_k;
        end
        
        V(i) = dot(pi,R);
        
        
        if pi(1) == piTrain(1)
            R1 = R(1);
            R2 = R(2);
            R_DBC = [R1, R2];
        end
        
        
        if pi(1) >= Box(1,1) && pi(1) <= Box(1,2)
            stockBox = pi1Box;
            pi1Box = [stockBox, pi(1)];
            stockVU = V_U;
            V_U = [stockVU, V(i)];
        end
        
    end
    
    for i = 1:size(pi1,2)
        pi = [pi1(i), pi2(i)];
        div_DBC(i) = pi(1)*(R1-R2) + R2;
        div_BCDMC(i) = pi(1)*(Rstar(1)-Rstar(2)) + Rstar(2);
        if pi(1) >= Box(1,1) && pi(1) <= Box(1,2)
            stockDBCU = div_DBC_U;
            div_DBC_U = [stockDBCU, div_DBC(i)];
            stockBCDMCU = div_BCDMC_U;
            div_BCDMC_U = [stockBCDMCU, div_BCDMC(i)];
        end
    end
    
    plot(pi1,V,'k:','LineWidth',1);
    hold on
    plot(pi1Box,V_U,'k-','LineWidth',1.5);
    hold on
    plot(pi1,div_DBC,':','LineWidth',1,'Color',[0.3 0.75 0.96]);
    hold on
    plot(pi1Box,div_DBC_U,'-','LineWidth',1,'Color',[0.3 0.75 0.96]);
    hold on
    plot(pi1,div_BCDMC,':','LineWidth',1,'Color','green');
    hold on
    plot(pi1Box,div_BCDMC_U,'-','LineWidth',1,'Color','green');
    hold on
    plot([piTrain(1) piTrain(1)],[0 V(piTrain(1)==pi1)],'LineWidth',0.5,'color','k')
    plot([piStar(1) piStar(1)],[0 rStar],'LineWidth',0.5,'color','k')
    set(gca,'TickLabelInterpreter', 'latex');
    if piStar(1) >= piTrain(1) && piStar(1) < 1
        set(gca,'XTick',[0 piTrain(1) piStar(1) 1], 'xticklabel',{0 '$\hat{\pi}_1$' '$\pi^{\star}_1$' 1})
    end
    if piStar(1) == 1
        set(gca,'XTick',[0 piTrain(1) piStar(1)], 'xticklabel',{0 '$\hat{\pi}_1$' '$\pi^{\star}_1$'})
    end
    if piStar(1) == 0
        set(gca,'XTick',[piStar(1) piTrain(1) 1], 'xticklabel',{'$\pi^{\star}_1$' '$\hat{\pi}_1$' 1})
    end
    if piStar(1) < piTrain(1) && piStar(1) > 0
        set(gca,'XTick',[0 piStar(1) piTrain(1) 1], 'xticklabel',{0 '$\pi^{\star}_1$' '$\hat{\pi}_1$' 1})
    end
    grid on
    legend('V over Simplex','V over U')
    title('Function V over U')
    xlabel('\pi_1')
    drawnow
    
end


end

