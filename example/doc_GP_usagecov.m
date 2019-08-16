% demonstrate the usage of simple covariance functions
% zhouweiyan 20190724

clear
% close all

%% initialization
n = 5; D = 2;
seed = 0;
rand('state',seed);
randn('state',seed);
x = randn(n,D)
xs = randn(3,D)

%% simple covariance library
opt = 10;
switch opt
    % simple covariance function
    case 1
        cov = {'covZero'}; hyp = [];
    case 2
        cov = {'covEye'}; hyp = [];
    case 3
        cov = {'covOne'}; hyp = [];
    case 4
%         cov = {'covGE','iso',[]}; 
%         ell = 2; gamma = 1.5; hyp = log([ell;gamma/(2-gamma)]);
        % to verify K(1,2)
        % x1 = x(1,:)'; x2 = x(2,:)'; r2 = (x1-x2)'*ell^(-2)*(x1-x2); exp(-sqrt(r2)^gamma)
        % to verify Ks(1,2)
        % x1 = x(1,:)'; xs2 = xs(2,:)'; r2 = (x1-xs2)'*ell^(-2)*(x1-xs2); exp(-sqrt(r2)^gamma)
        cov = {'covGE','ard',[]};
        L = [2;2]; gamma = 1.5; hyp = log([L;gamma/(2-gamma)]);
        % to verify K(1,2)
        % x1 = x(1,:)'; x2 = x(2,:)'; r2 = (x1-x2)'*inv((diag(L))^2)*(x1-x2); exp(-sqrt(r2)^gamma)
        % to verify Ks(1,2)
        % x1 = x(1,:)'; xs2 = xs(2,:)'; r2 = (x1-xs2)'*inv((diag(L))^2)*(x1-xs2); exp(-sqrt(r2)^gamma)
    case 5
%         cov = {'covMaterniso',3}; 
%         ell = 2; sf = 0.5; hyp = log([ell;sf]);
        % to verify K(1,2)
        % f = @(d,t)(d==1)*t+(d==3)*(1+t)+(d==5)*(1+t+t^2/3);
        % x1 = x(1,:)'; x2 = x(2,:)'; r = sqrt((x1-x2)'*ell^(-2)*(x1-x2));
        % d = 3; sf^2*f(d,sqrt(d)*r)*exp(-sqrt(d)*r)
        
        cov = {'covMaternard',3};
        L = [1;2]; sf = 0.5; hyp = log([L;sf]);
        % to verify K(1,2)
        % f = @(d,t)(d==1)*t+(d==3)*(1+t)+(d==5)*(1+t+t^2/3);
        % x1 = x(1,:)'; x2 = x(2,:)'; r = sqrt((x1-x2)'*inv((diag(L))^2)*(x1-x2));
        % d = 5; sf^2*f(d,sqrt(d)*r)*exp(-sqrt(d)*r)
    case 6
        cov = {'covPPiso',3};
        ell = 2; sf = 1; hyp = log([ell;sf]);
        % verify K(1,2)
        % v = 3; D = 2; j = floor(D/2)+v+1; x1 = x(1,:)'; x2 = x(2,:)';
        % r = sqrt((x1-x2)'*ell^(-2)*(x1-x2)); 
        
%         cov = {'covPPard',3};
%         L = [1;2]; sf = 1; hyp = log([L;sf]);
        % verify K(1,2)
        % v = 3; D = 2; j = floor(D/2)+v+1; x1 = x(1,:)'; x2 = x(2,:)';
        % r = sqrt((x1-x2)'*(diag(L))^(-2)*(x1-x2));
        
        % kpp = @(D,v,r)sf^2*((v==0)*max(0,(1-r)^j)+(v==1)*max(0,(1-r)^(j+1))*((j+1)*r+1))+...
        %    (v==2)*max(0,(1-r)^(j+2))*((j^2+4*j+3)*r^2+(3*j+6)*r+3)/3+...
        %    (v==3)*max(0,(1-r)^(j+3))*((j^3+9*j^2+23*j+15)*r^3+(6*j^2+36*j+45)*r^2+(15*j+45)*r+15)/15;
    case 7
%         cov = {'covRQiso'};
%         ell = 0.9; sf = 2; alp = 1; hyp = log([ell;sf;alp]);
        % verify K(1,2)
        % x1 = x(1,:)'; x2 = x(2,:)'; sf^2*(1+1/(2*alp*ell^2)*(x1-x2)'*(x1-x2))^(-alp)
        
        cov = {'covRQard'};
        L = [3;0.9]; sf = 2; alp = 1; hyp = log([L;sf;alp]);
        % verify K(1,2)
        % x1 = x(1,:)'; x2 = x(2,:)'; sf^2*(1+1/(2*alp)*(x1-x2)'*(diag(L))^(-2)*(x1-x2))^(-alp)
    case 8
        cov = {'covSEiso'};
        ell = 0.5; sf = 1; hyp = log([ell;sf]);
        % verify K(1,2)
        % x1 = x(1,:)'; x2 = x(2,:)'; sf^2*exp(-1/(2*ell^2)*(x1-x2)'*(x1-x2))
        
%         cov = {'covSEisoU'};
%         ell = 0.5; hyp = log(ell);
        % verify K(1,2)
        % x1 = x(1,:)'; x2 = x(2,:)'; exp(-1/(2*ell^2)*(x1-x2)'*(x1-x2))
       
%         cov = {'covSEard'};
%         L = [0.5;0.5]; sf = 1; hyp = log([L;sf]);
        % verify K(1,2)
        % x1 = x(1,:)'; x2 = x(2,:)'; sf^2*exp(-1/2*(x1-x2)'*(diag(L))^(-2)*(x1-x2))
    case 9
        cov = {'covNoise'};
        sf = 1; hyp = log(sf);
    case 10
        cov = {'covProd',{{'covMaterniso',3},{'covMaterniso',3}}};
        ell = 2; sf = 0.5; hyp = log([ell;sf;ell;sf]);
end

%% visualization
set(0,'DefaultFigureWindowStyle','docked');
% 1) query the number of parameters
feval(cov{:})
% 2) evaluate the function on x, x and xs to get cross-term
[K,dK] = feval(cov{:},hyp,x)
[ks,dkss] = feval(cov{:},hyp,xs)
[Ks,dKs] = feval(cov{:},hyp,x,xs)
% 3) plot a draw from the kernel
n_xstar = 71;
xrange = linspace(-5,5,n_xstar)';
if D~=1
    [a,b] = meshgrid(xrange);
    xstar = [a(:),b(:)];
    K0 = feval(cov{:},hyp,xstar,[0,0]);
    figure;
    surf(a,b,reshape(K0,n_xstar,n_xstar),'EdgeColor','none','LineStyle',...
        'none','FaceLighting','phong');
    xlabel('x_1'); ylabel('x_2'); title('K(X,0)');
    colormap(jet)
    
    K1 = feval(cov{:},hyp,xstar);
    K1 = K1+(1e-5)*eye(size(K1));
    n_samples = 1;
    samples = mvnrnd(zeros(size(a(:))),K1,n_samples)';
    figure;
    surf(a,b,reshape(samples,n_xstar,n_xstar)); axis equal
    xlabel('x_1'); ylabel('x_2'); title('the sample based on the covfunc');
    colormap(jet)
    
else
    K0 = feval(cov{:},hyp,xrange,0);
    figure;
    plot(xrange,K0,'LineWidth',2); axis equal;
    
    K1 = feval(cov{:},hyp,xrange);
    K1 = K1+(1e-5)*eye(size(K1));
    n_samples = 1;
    samples = mvnrnd(zeros(n_xstar,1),K1,n_samples)';
    figure;
    plot(xrange,samples); axis equal;
    
end
    








