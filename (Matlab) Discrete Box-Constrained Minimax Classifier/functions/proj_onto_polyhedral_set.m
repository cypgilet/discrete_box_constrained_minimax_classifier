function [piStar] = proj_onto_polyhedral_set(pi, Box, K)
% This algorithm is based on Theorem 2 of the paper from K. E. RUTKOWSKI.

Box(Box<0) = 0;
Box(Box>1) = 1;

% Generate matrix G:
U = [eye(K); -eye(K); ones(1,K); -ones(1,K)];
eta = [Box(:,2); -Box(:,1); 1; -1];

n = size(U,1);
G = zeros(n,n);
for i = 1:n
    for j = 1:n
        G(i,j) = dot(U(i,:),U(j,:));
    end
end


% Generate subsets of {1,...,n}:
M = 2^n-1;
I = num2cell(zeros(1,M));
i = 0;
for l = 1:n
    T = combnk(1:n,l);
    for p = i+1:i+size(T,1)
        I{p} = T(p-i,:);
    end
    i = i+size(T,1);
end

% Algorithm:
for m = 1:M
    
    Im = I{m};
    Gmm = G(Im,Im);
    
    if det(Gmm)~=0
        
        nu = zeros(2*K+2,1);
        
        w = zeros(length(Im),1);
        for i = 1:length(Im)
            w(i) = dot(pi,U(Im(i),:)) - eta(Im(i));
        end
        nu(Im) = linsolve(Gmm,w); 
        
        if sum(nu<-10^(-10)) == 0 
            A = G*nu;
            z = zeros(1,2*K+2);
            for j = 1:2*K+2
                z(j) = dot(pi,U(j,:)) - eta(j) - A(j);
            end
            if sum(z<=10^(-10)) == 2*K+2
                pi_new = pi;
                for i = 1:2*K+2
                    pi_new = pi_new - nu(i)*U(i,:);
                end
                break
            end
        end
    end
end

piStar = pi_new;

% Remove noisy small calculus errors:
piStar = piStar/sum(piStar);

end

