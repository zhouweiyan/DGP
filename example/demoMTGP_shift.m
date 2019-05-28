% example file to illustrate the use of multi-task Gaussian Process models
%
% by Robert Duerichen
% 31/01/2014

clear; %close all

% please select one of the example cases for shifting the feature space: (1-3)
MTGP_case = 2;
% case1: 3 signals which are phase shifted to each other, modelled by a 
%           normal MTGP with k_t = SE covariance function
%
% case2: 3 signals which are phase shifted to each other, modelled by a 
%           MTGP which considers a shift between the tasks with k_t = SE covariance function
%
% case3: Demonstration of an example with two tasks and two dimensions,
%           representing e.g. x and y of a spatial coordinate
%           Even though different labels are known from each task, MTGP
%           can estimate the offset in both dimensions (e.g. template
%           matching)
%           
seed=0; rng(seed); 

if MTGP_case <= 2
    % generate synthetic data for case 1 and 2 
    t = (0:0.05:5)';
    phase_shift_t = pi/4; 
    f = 1;
    translation = 0;
    z1 = sin(2*pi*f*t)+randn(length(t),1)./50;
    z2 = sin(2*pi*f*t+phase_shift_t)+randn(length(t),1)./50;
    z3 = sin(2*pi*f*t+2*(phase_shift_t))+randn(length(t),1)./100;
else
    % generate synthetic data for case 3
    translation = [0.15 -0.26];
    x = (-0.5:0.1:0.5)';
    y = (-0.4:0.06:0.6)';
    f = 1;
    x_mat = repmat(x,[1 length(y)]);
    y_mat = repmat(y,[1 length(x)]);
    y_mat = y_mat'; % [y_mat, x_mat] = meshgrid(y, x);
    z1 = sin(2*pi*f*x_mat).*sin(2*pi*f*y_mat)+randn(length(x),length(y))/50;
    z2 = sin(2*pi*f*x_mat).*sin(2*pi*f*y_mat)+randn(length(x),length(y))/50;
end

%% initial parameter
switch MTGP_case 
    case 1
        opt.training_data{1} = 1:100;   % index of know training points of signal 1
        opt.training_data{2} = 1:50;    % index of know training points of signal 2
        opt.training_data{3} = 1:50;    % index of know training points of signal 3
        opt.cov_func = 1;               % select cov. function
        opt.start = 1;                  % start index for prediction
        opt.end = 100;                  % end index for prediction
        z = [z1 z2 z3];
    case 2
        opt.training_data{1} = 1:100;   % index of know training points of signal 1
        opt.training_data{2} = 1:50;    % index of know training points of signal 2
        opt.training_data{3} = 1:50;    % index of know training points of signal 3
        opt.cov_func = 2;               % select cov. function
        opt.start = 1;                  % start index for prediction
        opt.end = 100;                  % end index for prediction
        z = [z1 z2 z3];
    case 3
        opt.training_data{1} = 1:2:length(x_mat(:));    % index of know training points of signal 1
        opt.training_data{2} = 20:2:length(x_mat(:));   % index of know training points of signal 2
        opt.cov_func = 3;               % select cov. function
        opt.start = 1;                  % start index for prediction
        opt.end = length(y_mat(:));     % end index for prediction
        z = [z1(:) z2(:)];
        t = [x_mat(:) y_mat(:)];
end

opt.init_num_opt = 200;         % number of optimization steps
opt.show = 1;                   % show result plot
opt.random = 1;rng('shuffle');  % if 1 - hyp for correlation and SE will set randomly
opt.num_rep = 1;                % number of trails for selecting hyp

% init values for hyperparameter (if opt.random ~= 1)
opt.se_hyp = 1;
opt.cc_hyp = [1 1 0 1 0 0];     % assumes that all signals are dependent
% opt.cc_hyp = [1 0 1 0 0 1];     % assumes that all signals are independent
opt.noise_lik = 0.1;
opt.shift_hyp = [0 0];

num_dim = size(opt.training_data,2);
num_cc_hyp = sum(1:size(opt.training_data,2));

%% generate training and test data with labels 
% also substract mean
x_train = [t(opt.training_data{1},:)-repmat(translation,[length(opt.training_data{1}),1]), ...
    ones(length(opt.training_data{1}),1)];

z_train = z(opt.training_data{1},1);

x_test = [t(opt.start:opt.end,:) ones(opt.end-opt.start+1,1)];
z_test = z(opt.start:opt.end,1);

for cnt_dim = 2:num_dim
        x_train = [x_train(:,:);...
             t(opt.training_data{cnt_dim},:) ...
             ones(length(opt.training_data{cnt_dim}),1)*cnt_dim];
       
        z_train = [z_train; z(opt.training_data{cnt_dim},cnt_dim)];
        
        x_test = [x_test(:,:);...
            t(opt.start:opt.end,:) ones(opt.end-opt.start+1,1)*cnt_dim];
        z_test = [z_test; z(opt.start:opt.end,cnt_dim)];
end
   
%% init covariance functions and hyperparameters
switch opt.cov_func
    case 1
        disp('Covariance Function: K = CC(l) x (SE_U(t))');
%         covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covSEisoU'}};
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD',{'MTGP_covMaternisoU',3}}};
        hyp.cov(1:num_cc_hyp) = opt.cc_hyp(1:num_cc_hyp);
        hyp.cov(num_cc_hyp+1) = log(sqrt(opt.se_hyp));

    case 2
        disp('Covariance Function: K = CC(l) x (SE_U_shift(t))');
%         covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covSEisoU_shift'}};
%         covfunc = {'MTGP_covProd',{{'MTGP_covCC_chol_nD_mask',ones(1,9)},'MTGP_covSEisoU_shift'}};
%         covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD',{'MTGP_covMaternisoU_shift',3}}}; % wrong, minimize doesn't work
%         covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD',{'MTGP_covMaternisoU_shift_mask',3,ones(1,9)}}};  % wrong
%         covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covPeriodicisoUU_shift'}};                  % wrong
%         covfunc = {'MTGP_covProd', {'MTGP_covCC_chol_nD',{'MTGP_covPeriodicisoUU_shift_mask',ones(1,9)}}};% wrong
%         covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD',{'MTGP_covQPSisoUU_shift'}}}; hyp.cov(num_cc_hyp+4) = .1;
%         covfunc = {'MTGP_covProd', {'MTGP_covCC_chol_nD',{'MTGP_covQPSisoUU_shift_mask',ones(1,10)}}}; hyp.cov(num_cc_hyp+4) = .1;
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covRQiso'}};
        hyp.cov(1:num_cc_hyp) = opt.cc_hyp(1:num_cc_hyp);
        hyp.cov(num_cc_hyp+1) = log(opt.se_hyp);
        hyp.cov(num_cc_hyp+2) = opt.shift_hyp(1);
        hyp.cov(num_cc_hyp+3) = opt.shift_hyp(2);
    case 3
        disp('Covariance Function: K = CC(l) x (SE_U_shift_nD(t))');
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covSEisoU_shift_nD'}};

        hyp.cov(1:num_cc_hyp) = opt.cc_hyp(1:num_cc_hyp);
        hyp.cov(num_cc_hyp+1) = log(opt.se_hyp);
        hyp.cov(num_cc_hyp+2) = opt.shift_hyp(1);
        hyp.cov(num_cc_hyp+3) = opt.shift_hyp(2);
end

% likelihood function
likfunc = @likGauss;
hyp.lik = log(opt.noise_lik);

% optimize hyperparameters
[results.hyp] = minimize(hyp, @MTGP, -opt.init_num_opt, @MTGP_infExact, [], covfunc, likfunc, x_train,z_train);
results.nlml = MTGP(results.hyp, @MTGP_infExact, [], covfunc, likfunc, x_train, z_train);

%% perform prediction
[results.m, results.s2, fmu, fs2, results.p] = MTGP(results.hyp, @MTGP_infExact, [], covfunc, likfunc, x_train, z_train, x_test, z_test);

% reshape of results
results.m = reshape(results.m,[opt.end-opt.start+1 num_dim]);
results.s2 = reshape(results.s2,[opt.end-opt.start+1 num_dim]);
results.p = exp(reshape(results.p,[opt.end-opt.start+1 num_dim]));

%% compute RMSE for training and test data for each dimension
for cnt_dim = 1:num_dim
    index_test_data = opt.start:opt.end;
    
    index_test_data(ismember(index_test_data,opt.training_data{cnt_dim})) = [];

    results.rmse_test(cnt_dim) = rms(results.m(index_test_data-opt.start+1,cnt_dim)-...
        z(index_test_data,cnt_dim));

    results.rmse_train(cnt_dim) = rms(results.m(opt.training_data{cnt_dim},cnt_dim)-...
        z(opt.training_data{cnt_dim}+opt.start-1,cnt_dim)); 
    
end

% compute resulting K_f matrix
vec_dim = 1:num_dim;
L = zeros(num_dim,num_dim);
for cnt_dim = 1:num_dim
    L(cnt_dim,1:vec_dim(cnt_dim)) = [results.hyp.cov(sum(vec_dim(1:cnt_dim-1))+1:sum(vec_dim(1:cnt_dim-1))+vec_dim(cnt_dim))];
end

results.K_f =  L*L';

% print results on console:
disp('Estimated cross correlation covariance K_f:');
results.K_f

if opt.cov_func == 2
    disp('Estimated time shift: ');
    disp(['\Delta t_1: ',num2str(results.hyp.cov(end-1))]);
    disp(['\Delta t_2: ',num2str(results.hyp.cov(end))]);
elseif opt.cov_func == 3
    disp('Estimated shift: ');
    disp([' dim1: ',num2str(results.hyp.cov(end-1)), '(true shift: ',num2str(-translation(1)),')']);
    disp([' dim2: ',num2str(results.hyp.cov(end)), '(true shift: ',num2str(-translation(2)),')']);
end

%% plot basic results
if opt.show == 1 
    if opt.cov_func <= 2
    h=figure('Position',[1 1 1400 800]);
        for cnt_dim = 1:num_dim
            % plot prediction results
            subplot(2,num_dim,cnt_dim);

            min_val = min(results.m(:,cnt_dim))-abs(min(results.s2(:,cnt_dim)));
            max_val = max(results.m(:,cnt_dim))+abs(max(results.s2(:,cnt_dim)));

            hTlines = plot([t(opt.training_data{cnt_dim}) t(opt.training_data{cnt_dim})]',...
                [ min_val max_val]','Color',[0.85 0.85 0.5]);
            hTGroup = hggroup;
            set(hTlines,'Parent',hTGroup);
            set(get(get(hTGroup,'Annotation'),'LegendInformation'),...
                'IconDisplayStyle','on'); 
            hold on;
            f = [results.m(:,cnt_dim)+2*sqrt(results.s2(:,cnt_dim)); flip(results.m(:,cnt_dim)-2*sqrt(results.s2(:,cnt_dim)),1)];
            fill([t(opt.start:opt.end); flip(t(opt.start:opt.end),1)], f, [0.7 0.7 0.7],'EdgeColor','none')

            % plot mean org signal
            plot(t(opt.start:opt.end),z(opt.start:opt.end,cnt_dim),'b');

            % plot mean predicted signal
            plot(t(opt.start:opt.end),results.m(1:opt.end-opt.start+1,cnt_dim),'r');

            if cnt_dim == 1
                legend('training data','95% conf. int.','org. values','pred. values','Orientation','horizontal','Location',[0.45 0.49 0.15 0.04]);
            end
            axis tight

            title(['S',num2str(cnt_dim),': RMSE_{train}: ',num2str(results.rmse_train(cnt_dim)),...
                ' - RMSE_{test}: ',num2str(results.rmse_test(cnt_dim))]);
            xlabel('time [s]');
            ylabel('amplitude y [mm]');


            subplot(2,num_dim,num_dim+cnt_dim);
            min_val = min(results.p(:,cnt_dim));
            max_val = max(results.p(:,cnt_dim));
            plot([t(opt.training_data{cnt_dim}) t(opt.training_data{cnt_dim})]',...
                [ min_val max_val]','Color',[0.85 0.85 0.5]);
            hold on
            plot(t(opt.start:opt.end),results.p(1:opt.end-opt.start+1,cnt_dim),'b');

            title('probability p');
            axis tight
            xlabel('time [s]');
            ylabel('p');    
        end
    else
        figure
        subplot(2,1,1);
        xlim([-0.8 0.8]);
        ylim([-0.8 0.8]);
        zlim([-1.1 1.1]);
        hold on
        plot3(x_train(x_train(:,3)==1,1),x_train(x_train(:,3)==1,2),z_train(x_train(:,3)==1,1),'k*','Markersize',8,'Linewidth',2);
        surface(x_mat-translation(1), y_mat-translation(2),z1)
        xlabel('Dim 1');
        ylabel('Dim 2');
        title({'Task 1';[': RMSE_{train}: ',num2str(results.rmse_train(1)),...
            ' - RMSE_{test}: ',num2str(results.rmse_test(1))]});
        hold on;
        legend('training points task 1')
        
        subplot(2,1,2);
        xlim([-0.8 0.8]);
        ylim([-0.8 0.8]);
        zlim([-1.1 1.1]);
        hold on
        plot3(x_train(x_train(:,3)==2,1)',x_train(x_train(:,3)==2,2)',z_train(x_train(:,3)==2,1)','k*','Markersize',10,'Linewidth',2);
        surface(x_mat, y_mat,z2);%surface(x,y,z2');
        colormap(jet);
        xlabel('Dim 1');
        ylabel('Dim 2');
        title({['Task 2 - shifted by dim1: ',num2str(results.hyp.cov(end-1)),' (true: ',num2str(-translation(1)),') / dim2: ',num2str(results.hyp.cov(end)),' (true: ',num2str(-translation(2)),')'];
            ['RMSE_{train}: ',num2str(results.rmse_train(2)),' - RMSE_{test}: ',num2str(results.rmse_test(2))]});
        legend('training points task 2')
    end
end
