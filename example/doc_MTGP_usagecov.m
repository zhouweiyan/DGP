% demonstrate the usage of MTGP covariance functions
% zhouweiyan 20190725

clear

%% initialization
n = 5; D = 2;
seed = 0;
rand('state',seed);
randn('state',seed);
x = [randn(n,D) ones(n,1)];
xs = [randn(3,D) 2*ones(3,1)];

%% simple MTGP covariance library
opt = -11;
switch opt
    case 1
        cov = {'MTGP_covCC_chol_nD'};
%         hyp = [1;0;1];    % independent
        hyp = [1;1;0];      % dependent
%         hyp = [0.3;1;1];
    case 3
        cov = {'MTGP_covNoise'};
        sf = 1; hyp = log(sf);
    case -11
        cov = {'MTGP_covProd',{'MTGP_covCC_chol_nD',{'MTGP_covMaterniso',3}}};
        hyp(1:3) = [1;1;0]; ell = 2; sf = 0.5; hyp(4:5) = log([ell;sf]);
    case -12 
        cov = {'MTGP_covProd',{{'MTGP_covMaterniso',3},{'MTGP_covMaterniso',3}}};
        ell = 2; sf = 0.5; hyp = log([ell;sf;ell;sf]);
    case -21
        cov = {'MTGP_covSum',{{'MTGP_covMaterniso',3},{'MTGP_covMaterniso',3}}};
        ell = 2; sf = 0.5; hyp = log([ell;sf;ell;sf]);
    case -22
        cov = {'MTGP_covSum',{'MTGP_covCC_chol_nD',{'MTGP_covMaterniso',3}}};
        hyp(1:3) = [1;0;1]; ell = 2; sf = 0.5; hyp(4:5) = log([ell;sf]);
    case 11
        cov = {'MTGP_covMaterniso',3};
        ell = 2; sf = 0.5; hyp = log([ell;sf]);
    case 12
        cov = {'MTGP_covMaternisoU',5};
        ell = 2; hyp = log(ell);
    case 13
        cov = {'MTGP_covMaternard',3};
        L = [2;2]; sf = 0.5; hyp = log([L;sf]);
    case 14
        cov = {'MTGP_covMaternardU',3};
        L = [2;2]; hyp = log(L);
    case 21
        cov = {'MTGP_covRQiso'};
        ell = 0.9; sf = 1; alp = 1; hyp = log([ell;sf;alp]);
    case 22
        cov = {'MTGP_covRQisoU'};
        ell = 0.9; alp = 1; hyp = log([ell;alp]);
    case 23
        cov = {'MTGP_covRQard'};
        L = [3;0.9]; sf = 2; alp = 1; hyp = log([L;sf;alp]);
    case 24
        cov = {'MTGP_covRQardU'};
        L = [3;0.9]; alp = 1; hyp = log([L;alp]);
    case 31
        cov = {'MTGP_covSEiso'};
        ell = 0.5; sf = 1; hyp = log([ell;sf]);
    case 32
        cov = {'MTGP_covSEisoU'};
        ell = 0.5; hyp = log(ell);
    case 33
        cov = {'MTGP_covSEard'};
        L = [0.5;0.5]; sf = 1; hyp = log([L;sf]);
    case 34
        cov = {'MTGP_covSEardU'};
        L = [0.5;0.5]; hyp = log(L);
    case 41
        cov = {'MTGP_covPPiso',3};
        ell = 2; sf = 1; hyp = log([ell;sf]);
    case 42
        cov = {'MTGP_covPPisoU',3};
        ell = 2; hyp = log(ell);
    case 43
        cov = {'MTGP_covPPard',3};
        L = [1;2]; sf = 1; hyp = log([L;sf]);
    case 44
        cov = {'MTGP_covPPardU',3};
        L = [1;2]; hyp = log(L);
end

%% visualization
set(0,'DefaultFigureWindowStyle','docked');

% 1) query the number of parameters
feval(cov{:})

% 2) evaluate the function on x, xs to get cross-term
% K = feval(cov{:},hyp,x)
% Ks = feval(cov{:},hyp,x,xs)
% Kss = feval(cov{:},hyp,xs)

X = [x;xs];
Km = feval(cov{:},hyp,X)
        
% 3) plot a draw from the kernel
n_xstar = 71;
xrange = linspace(-5,5,n_xstar)';
if D~=1
    [a,b] = meshgrid(xrange);
    xstar = [a(:) b(:) ones(length(a(:)),1)];
    
    K0 = feval(cov{:},hyp,xstar,[0 0 1]);
    figure;
    surf(a,b,reshape(K0,n_xstar,n_xstar),'EdgeColor','none',...
        'LineStyle','none','FaceLighting','phong');
    xlabel('x_1'); ylabel('x_2'); title('K(X,0)');
    colormap(jet)
    
    K1 = feval(cov{:},hyp,xstar);
    K1 = K1+(1e-5)*eye(size(K1));
    n_samples = 1;
    samples = mvnrnd(zeros(size(a(:))),K1,n_samples)';
    figure;
    surf(a,b,reshape(samples,n_xstar,n_xstar)); axis equal
    title('the sample based on the covfunc')
    colormap(jet)
end





















