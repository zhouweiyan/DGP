function K = MTGP_covSEard(hyp, x, z, i)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
 
															  
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
									
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% by zhouweiyan
% 20190619

if nargin<2, K = 'D'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1); D = size(x,2)-1;
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x(:,1:end-1),1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(diag(1./ell)*x(:,1:end-1)');
  else                                                   % cross covariances Kxz
    K = sq_dist(diag(1./ell)*x(:,1:end-1)',diag(1./ell)*z(:,1:end-1)');
  end
end

																				
K = sf2*exp(-K/2);                                                  % covariance
if nargin>3                                                        % derivatives
  if i<=D                                              % length scale parameters
    if dg
      K = K*0;
    else
      if xeqz
        K = K.*sq_dist(x(:,i)'/ell(i));
      else
        K = K.*sq_dist(x(:,i)'/ell(i),z(:,i)'/ell(i));
      end
    end
  elseif i==D+1                                            % magnitude parameter
    K = 2*K;
  else
    error('Unknown hyperparameter')
  end
end