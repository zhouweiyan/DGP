function C = corr_2d(a, b,cc)

if nargin<1  || nargin>3 || nargout>1, error('Wrong number of arguments.'); end

C_a = repmat(a,1,length(b));
C_b = repmat(b,1,length(a))';

C_eq = bsxfun(@eq,C_a,C_b);     % eq: Determine equality
C_nq = bsxfun(@ne,C_a,C_b)*cc;  % ne: Determine inequality

C = C_eq+C_nq;
