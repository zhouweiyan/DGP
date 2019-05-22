function [Kn] = normCorrMtrx(gamma,dim)
% NORMCORRMTRX get normalised correlation matrix given hyperparameters of
% the GP regression
%   [Kn] = normCorrMtrx(gamma,dim)
%
% Description: 
%       inputs are the parameters of a lower triangular matrix L of size [m*k]
%           |   gamma_1     0               ...  0           |
%       L = |   gamma_2     gamma_3         ...  0           |
%                       ...
%           |   gamma_m*k-k gamma_m-k-k+1   ...  gamma_m*k   |
% 
% Inputs:  
%       gamma       vector [m*k x 1], containing all hyperparameters
%       dim         scalar, number of tasks / dimensions (here it would be m)
% 
% Outputs
%       Kn          normalised correlation matrix, with diag(Kn) = 1
% 

if nargin < 2
    help normCorrMtrx;
    return;
end

% reshape original hyperparameters in a lower triangular matrix L
L = triu(ones(dim))'; 
L(L(:)==1) = gamma(1:sum(L(:)));
S = sign(L);    % compute signal of entry
D = diag(L*L'); % get diagonal of "unnormalised" correlation matrix    
Ln = S.*(sqrt((L.^2)./repmat(D,1,size(L,2)))); 
Kn = Ln*Ln';

end