function K = MTGP_covSEisoU(hyp, x, z, i)

% Squared Exponential covariance function with isotropic distance and scaling
% measure. 
%
% Based on the covSEisoU.m function of the GPML Toolbox - 
%   with the following changes:
%       - only elements of x(:,1:end-1)/z(:,1:end-1) will be analyzed, 
%       - x(:,end)/z(:,end) will be ignored, as it contains only the label information
%       - independent of the label all x values will have the same hyp
%       - output-scaling hyperparameter is fixed to 1
%
% The covariance function is parameterized as:
%
% k(x^p,x^q) = exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
%
% where the P matrix is ell^2 times the unit matrix.
% The hyperparameters are:
%
% hyp = [ log(ell)]
%
% by Robert Duerichen
% 18/11/2013

if nargin<2, K = '1'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

ell = exp(hyp(1));                                 % characteristic length scale

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x(:,1:end-1),1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(x(:,1:end-1)'/ell);
  else                                                   % cross covariances Kxz
    K = sq_dist(x(:,1:end-1)'/ell,z(:,1:end-1)'/ell);
  end
end

if nargin<4                                                        % covariances
  K = exp(-K/2);
else                                                               % derivatives
  if i==1
    K = exp(-K/2).*K;
  else
    error('Unknown hyperparameter')
  end
end