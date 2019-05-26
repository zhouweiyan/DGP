function [gamma_n, Kc_n] = normalize_Kc(gamma,dim)
% normalize the parameters of a "free-form" covariance function Kc to [-1 1]
%
% inputs are the parameters of a lower triangular matrix L of size [m*k]
%       |   gamma_1     0               ...  0           |
% L =   |   gamma_2     gamma_3         ...  0           |
%                       ...
%       |   gamma_m*k-k gamma_m-k-k+1   ...  gamma_m*k   |
%
% 
% Input:
%   gamma   - ector [m*k x 1] containing all parameters
%   dim     - number of tasks / dimensions (here it would be m)
% 
% Output:
%   gamma_n - normalized results
%   Kc_n    - normalized matrix Kc
%
% by Robert Duerichen
%
% 3/02/2014
% zwy: seems wrong

% check if number of valid
num_para = 1:dim;
if sum(num_para) ~= numel(gamma)
    error('number of parameters disagree with dimenstion');
end

% parametrize initial lower triangular matrix L
L = triu(ones(dim,dim));
[ind] = find(L(:) ==1);
 
[ind2d(:,1), ind2d(:,2)] = find(L ~=0);
L(ind) =  gamma;
L = L';

% normalize parameter 
for cnt = 1:length(gamma)    
    gamma_n(cnt) = sqrt(gamma(cnt).^2./sum(L(ind2d(cnt,2),:).^2))*sign(gamma(cnt));
end

% parametrize normalized lower triangular matrix L
L_n = triu(ones(dim,dim));
L_n(ind) =  gamma_n;

% compute normalized matrix Kc_n
Kc_n = L_n'*L_n;


