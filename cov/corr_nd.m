function C = corr_nd(a, b,K_f)

if nargin<1  || nargin>3 || nargout>1, error('Wrong number of arguments.'); end

C = K_f(a(:),b(:));


