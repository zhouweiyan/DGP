function K = MTGP_covSEisoU_shift_mask(mask,hyp, x, z, i)

% Squared Exponential covariance function with isotropic distance and scaling
% measure which includes a time schift hyperparameter for all signals relative to 
% the first signal dimension (which leads to dim-1 additional hyp)
%
% Based on the covSEiso.m function of the GPML Toolbox - 
%   with the following changes:
%       - only elements of x(:,1)/z(:,1) will be analyzed, 
%       - x(:,end) will be ignored, as it contains only the label information
%       - independent of the label all x values will have the same hyp
%       - output-scaling hyperparameter is fixed to 1
%       - function considers additional hyperparameter theta_s for features shift
%           (function is limited to 1D features)
%       - mask parameter is a vector of size hyp and if mask(i) == 0, the
%       derivative of hyp(i) will be 0
%       
% The covariance function is parameterized as:
%
% k(x^p,x^q) = exp(-((x-theta_s)^p - ((x-theta_s)^q))'*inv(P)*((x-theta_s)^p - (x-theta_s)^q)/2) 
%
% where the P matrix is ell^2 times the unit matrix.
% The hyperparameters are:
%
% hyp = [ log(ell);
%           theta_s(1)
%           ...
%           theta_s(nL-1)]
%
% by Robert Duerichen
% 04/02/2014

if nargin<3, K = 'nL'; return; end                  % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

% check if size of mask is correct
if length(mask) ~= length(hyp)
    error('Size of mask vector is not equivalent to hyperparameter vector');
end

% check if derivate shall be computed or not
if exist('i','var') 
    if mask(i) == 0
        if xeqz                                    % symmetric matrix Kxx
            K = zeros(length(x));
        else                                       % cross covariances Kxz
            K = zeros(length(x),length(z));
        end
        return;                                      % terminate function
    end
end

nL = max(x(:,end));                                % get number of labels
ell = exp(hyp(1));                                 % characteristic length scale
shift = (hyp(2:end));                              % time shift hyp


%% perform shift
for ii = 2:nL
   x(x(:,end)== ii,1) = x(x(:,end)== ii,1)+shift(ii-1);
   if ~isempty(z)
       z(z(:,2)== ii,1) = z(z(:,2)== ii,1)+shift(ii-1);
   end
end

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x(:,1:end-1),1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(x(:,1)'/ell);
  else                                                   % cross covariances Kxz
    K = sq_dist(x(:,1)'/ell,z(:,1)'/ell);
  end
end

if nargin<5                                                        % covariances
  K = exp(-K/2);
else                                                               % derivatives
  if i<=nL
      if i == 1 % derivative of the x-scaling hyperparameter
          K = exp(-K/2).*K;
      else  % derivatives of the shift hyperparameters
          ind_i = (x(:,2) ==i);
          ind_ni = (x(:,2) ~=i);
          B = zeros(length(x));
          B(ind_ni,ind_i) = ones(sum(ind_ni),sum(ind_i));
          B(ind_i,ind_ni) = -ones(sum(ind_i),sum(ind_ni));
          A = repmat(x(:,1) ,[1 length(x)]);
          K = B.*((A-A')./(ell^2).*exp(-K/2));
      end
  else
    error('Unknown hyperparameter')
  end
end