function [ pi_new ] = projection_onto_U(pi, Box)
%--------------------------------------------------------------------------
% Paper: Discrete Box-Constrained Minimax Classifier for Uncertain and
% Imbalanced Class Proportions.
%--------------------------------------------------------------------------
%======INPUTS:
%   # pi  : vector of class proportion to project.
%   # Box : Box-Constraint.
%======OUTPUTS:
%   # pi_new : projeted priors onto U.
%--------------------------------------------------------------------------

K = length(pi);

% check if pi is in U:
check_U = 0;
if sum(pi)==1
    for k = 1:K
        if pi(k) >= Box(k,1) && pi(k) <= Box(k,2)
            check_U = check_U + 1;
        end
    end
end

if check_U == K
    pi_new = pi;
end


if check_U < K
    
    % check if the Box is the cube [0,1]:
    n = 0;
    for k = 1:K
        if Box(k,1) <= 0 && Box(k,2) >= 1
            n = n+1;
        end
    end
    
    if n==K % proj simplex Condat:
        proj_simplex_vector = @(y) max(y-max((cumsum(sort(y,1,'descend'),1)-1)./(1:size(y,1))'),0);
        pi_new = proj_simplex_vector(pi');
        pi_new = pi_new';
    end
    
    if n<K % proj onto polyhedral set:
        pi_new = proj_onto_polyhedral_set(pi, Box, K);
    end
    
end

end

