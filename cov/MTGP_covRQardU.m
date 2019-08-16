function K = MTGP_covRQardU(hyp, x, z, i)

% Rational Quadratic covariance function with Automatic Relevance Determination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space, sf2 is the signal variance and alpha
% is the shape parameter for the RQ covariance. The hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          ..
%         log(ell_D)
%         log(alpha) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-08-04.
%
% by zhouweiyan
% 20190725

if nargin<2, K = '(D+1)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1); D = size(x,2)-1;
ell = exp(hyp(1:D));
alpha = exp(hyp(D+1));

% precompute squared distances
if dg                                                               % vector kxx
  D2 = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    D2 = sq_dist(diag(1./ell)*x(:,1:end-1)');
  else                                                   % cross covariances Kxz
    D2 = sq_dist(diag(1./ell)*x(:,1:end-1)',diag(1./ell)*z(:,1:end-1)');
  end
end

if nargin<4                                                        % covariances
  K = (1+0.5*D2/alpha).^(-alpha);
else                                                               % derivatives
  if i<=D                                               % length scale parameter
    if dg
      K = D2*0;
    else
      if xeqz
        K = (1+0.5*D2/alpha).^(-alpha-1).*sq_dist(x(:,i)'/ell(i));
      else
        K = (1+0.5*D2/alpha).^(-alpha-1).*sq_dist(x(:,i)'/ell(i),z(:,i)'/ell(i));
      end
    end
  elseif i==D+1                                            % magnitude parameter
    K = 2*(1+0.5*D2/alpha).^(-alpha);
  elseif i==D+2
    K = (1+0.5*D2/alpha);
    K = K.^(-alpha).*(0.5*D2./K - alpha*log(K));
  else
    error('Unknown hyperparameter')
  end
end