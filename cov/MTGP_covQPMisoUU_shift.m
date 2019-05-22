function [K] = MTGP_covQPMisoUU_shift_fix(d,hyp, x, z, i)

% Stationary covariance function for a quasi-periodic function based on a 
% multiplication of a Matern and Periodic function
%
%       - only elements of x(:,1:end-1)/z(:,1:end-1) will be analyzed, 
%       - x(:,end)/z(:,end) will be ignored, as it contains only the label information 
%       - independent of the label all x values will have the same hyp
%       - feature scaling hyperparameter is fixed to 1
%       - output scaling hyperparameter is fixed to 1
%
% k(x,y) = exp(-((x-theta_s) - ((y-theta_s)))'*inv(P)*((x-theta_s) - (y-theta_s)^q)/2)  * ...
%               exp( -2*sin^2( pi*||(x-theta_s)-(y-theta_s)||/p ) )
%
% where the P matrix is ell^2 times the unit matrix. 
% The hyperparameters are:
%
% hyp = [ log(ell)
%         log(p);
%         theta_s(1)
%           ...
%         theta_s(nL-1)]
%
% modified by Robert Duerichen
% 10/04/2014

if nargin<3, K = 'nL+1'; return; end                        % report number of parameters
if nargin<4, z = []; end                                    % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;    % determine mode


nL = max(x(:,2));                                   % get number of labels
ell = exp(hyp(1));                                  % characteristic length scale
p   = exp(hyp(2));                                  % period 
shift = (hyp(3:end));                               % time shift hyp

%% define Matern function
if all(d~=[1,3,5]), error('only 1, 3 and 5 allowed for d'), end         % degree

switch d
  case 1, f = @(t) 1;               df = @(t) 1;
  case 3, f = @(t) 1 + t;           df = @(t) t;
  case 5, f = @(t) 1 + t.*(1+t/3);  df = @(t) t.*(1+t)/3;
end
          m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);



%% perform shift
for ii = 2:nL
   x(x(:,2)== ii,1) = x(x(:,2)== ii,1)+shift(ii-1);
   if ~isempty(z)
       z(z(:,2)== ii,1) = z(z(:,2)== ii,1)+shift(ii-1);
   end
end

% precompute distances
if dg                                                               % vector kxx
  K_p = zeros(size(x(:,1),1),1);
  K_m = zeros(size(x(:,1),1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K_m = sqrt( sq_dist(sqrt(d)*x(:,1:end-1)'/ell) );
    K_p = sqrt(sq_dist(x(:,1)'));
  else                                                   % cross covariances Kxz
    K_m = sqrt( sq_dist(sqrt(d)*x(:,1:end-1)'/ell,sqrt(d)*z(:,1:end-1)'/ell) );
    K_p = sqrt(sq_dist(x(:,1)',z(:,1)'));
  end
end

K_p = pi*K_p/p;
if nargin<5                                                        % covariances
    K_p = sin(K_p); K_p = K_p.*K_p; K_p =   exp(-2*K_p);
    K_m = m(K_m,f);
    K = K_p.*K_m;
else                                                               % derivatives
    if i<=nL+1
        if i==1         % derivatives of the se hyperparameter
            K_p = sin(K_p); K_p = K_p.*K_p; K_p =   exp(-2*K_p);
            K = K_p.*K_m.*dm(K_m,f);
        elseif i==2         % derivatives of the periodic hyperparameter
            K_m = m(K_m,f);
            R = sin(K_p); K = K_m.* 4.*exp(-2*R.*R).*R.*cos(K_p).*K_p;
        elseif i > 2 && i <= nL+1% derivatives of the shift hyperparameters
          ind_i = (x(:,2) ==i-1);
          ind_ni = (x(:,2) ~=i-1);
          B = zeros(length(x));
          B(ind_ni,ind_i) = ones(sum(ind_ni),sum(ind_i));
          B(ind_i,ind_ni) = -ones(sum(ind_i),sum(ind_ni));
          A = repmat(x(:,1) ,[1 length(x)]);
          
          switch d
              case 1
                dK_m = B.*dm(K_m,f)./(ell).*sign(A-A');
              case 3
                dK_m = B.*sqrt(d).*dm(K_m,f)./(ell).*sign(A-A');
              case 5
                dK_m = sqrt(d).*(K_m.^2 +K_m)./(3*ell).*exp(-K_m).*B.*sign(A-A'); 
          end
          
          R = sin(K_p);
          dK_p = B.*4.*exp(-2*R.*R).*R.*cos(K_p).*pi./p.*sign(A-A');
          
          K_p = sin(K_p); K_p = K_p.*K_p; K_p =   exp(-2*K_p);
          
          K_m = m(K_m,f);
          
          K = dK_m.*K_p + K_m.*dK_p;
      end
    else
        error('Unknown hyperparameter')
    end
        
end