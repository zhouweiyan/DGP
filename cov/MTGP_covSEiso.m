function K = MTGP_covSEiso(hyp, x, z, i)

% Squared Exponential covariance function with isotropic distance measure. 
										  
%
% Based on the covSEiso.m function of the GPML Toolbox - 
%   with the following changes:
%       - only elements of x(:,1:end-1)/z(:,1:end-1) will be analyzed, 
%       - x(:,end) will be ignored, as it contains only the label information
%       - independent of the label all x values will have the same hyp
%
% The covariance function is parameterized as :
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
%
% where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
% variance. The hyperparameters are:
%
				  
					
 
																		   
 
% hyp = [ log(ell);
%           log(sf)]
%
% by Robert Duerichen
% 04/02/2014

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

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
  K = sf2*exp(-K/2);
else                                                               % derivatives
  if i==1
    K = sf2*exp(-K/2).*K;
  elseif i==2
    K = 2*sf2*exp(-K/2);
  else
    error('Unknown hyperparameter')
  end
end