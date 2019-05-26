% example file to illustrate the use of multi-task Gaussian Process models
% to compute a "noise-free" correlation coefficient
%
% within this example two sinusodial signals are given, which have a
% Pearson correlation coefficient of 1
% depending on the scaling factor, Gaussian noise is added to both signals 
% which will effect the Pearson correlation coefficient but not the MTGP
% correlation coefficient
%
%
% by Robert Duerichen
% 31/01/2014
close all; clear

path_gpml = 'E:\OneDrive - hnu.edu.cn\tools\matlabcourse\GPML_matlab\gpml-matlab-v4.2-2018-06-11';                     % please insert here path of GPML Toolbox

% add folders of MTGP and GPML Toolbox
if ~isunix  % windows system
    addpath(genpath('..\..\'));
    addpath(genpath(path_gpml));
else        % linux system
    addpath(genpath('../../'));
    addpath(genpath(path_gpml));
end

phase_shift = 0;                % phase shift between the two signals
scale1 = 1;                     % amplitude of signal 1
scale2 = 2;                     % amplitude of signal 2
scaling_factor = [0.001 0.005 0.01 0.05 0.1 0.5 1];             
                                % scaling factor of the noise component


%% options for MTGP
opt.init_num_opt = 200; 
opt.training_data{1} = 1:50;    % index of know training points of signal 1
opt.training_data{2} = 1:50;    % index of know training points of signal 2

%% initial parameter
opt.show = 1;                   % show result plot
opt.start = 1;                  % start index for prediction
opt.end = 50;                   % end index for prediction
opt.random = 1;                 % if 1 - hyp for correlation and SE will set randomly
opt.num_rep = 30;               % number of trails for selecting hyp

% init values for hyperparameter - only relevant if if opt.random ~= 1
opt.se_hyp = 6;                 % init hyp for SE
opt.cc_hyp = [1 0 1 0 0 1];     % init hyp for correlation
opt.noise_lik = 0.1;            % init hyp for lik

num_cc_hyp = sum(1:size(opt.training_data,2));  % define number of correlation hyperparameters
random_bounds.cc = [-1,1];                      % define bounds for random estimation of correlation hyp
random_bounds.SE = [0,1];                       % define bounds for random estimation of SE hyp
random_bounds.noise = [0,1];                    % define bounds for random estimation of lik hyp


%% loop over scaling factors
for count = 1:length(scaling_factor)
    % generate data with noise
    t = (0:0.05:10)';               
    y1 = scale1*sin(2*pi*t)+randn([length(t),1])*(scaling_factor(count));
    y2 = scale2*sin(2*pi*t+phase_shift)+randn([length(t),1])*(scaling_factor(count));

    y = [y1,y2];

    num_dim = size(y,2);                        % define number of tasks
    
    %% generate training and test data with labels 
    % also substract mean
    x_train = [t(opt.training_data{1},1), ...
        ones(length(opt.training_data{1}),1)];

    y_train_mean(1) = mean(y(opt.training_data{1},1));
    y_train = y(opt.training_data{1},1)-y_train_mean(1);

    x_test = [t(opt.start:opt.end) ones(opt.end-opt.start+1,1)];
    y_test = y(opt.start:opt.end,1)-y_train_mean(1);


    for cnt_dim = 2:num_dim
        x_train = [x_train(:,1) x_train(:,2);...
             t(opt.training_data{cnt_dim},1) ...
             ones(length(opt.training_data{cnt_dim}),1)*cnt_dim];

        y_train_mean(cnt_dim) = mean(y(opt.training_data{cnt_dim},cnt_dim));
        y_train = [y_train; y(opt.training_data{cnt_dim},cnt_dim)-...
            y_train_mean(cnt_dim)];

        x_test = [x_test(:,1) x_test(:,2);...
            t(opt.start:opt.end) ones(opt.end-opt.start+1,1)*cnt_dim];
        y_test = [y_test; y(opt.start:opt.end,cnt_dim)-y_train_mean(cnt_dim)];
    end

    % define covariance function
    disp('Covariance Function: K = CC(l) x (SE_U(t))');
    covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covSEisoU'}};
    hyp.cov(1:num_cc_hyp) = opt.cc_hyp(1:num_cc_hyp);
    hyp.cov(num_cc_hyp+1) = log(opt.se_hyp);
     
    % likelihood function
    likfunc = @likGauss;
    hyp.lik = log(opt.noise_lik);

    % optimize hyperparameters
    for cnt_rep = 1:opt.num_rep
        disp(['Number of rep: ',num2str(cnt_rep)]);
        % if opt. random == 1 - hyper will be choosen randomly
        if opt.random
            % random hyp for first correlation term
            hyp.cov(1:num_cc_hyp) = rand(num_cc_hyp,1)*(random_bounds.cc(2)-random_bounds.cc(1))+...
                random_bounds.cc(1);

            hyp.cov(num_cc_hyp+1) = rand(1)*(random_bounds.SE(2)-random_bounds.SE(1))+...
                random_bounds.SE(1);

            hyp.lik(1) = rand(1)*(random_bounds.noise(2)-random_bounds.noise(1))+...
                random_bounds.noise(1);
        end

        % optimize hyperparameter
        [results.hyp{cnt_rep}] = minimize(hyp, @MTGP, -opt.init_num_opt, @MTGP_infExact, [], covfunc, likfunc, x_train,y_train);

        % training
        results.nlml(cnt_rep) = MTGP(results.hyp{cnt_rep}, @MTGP_infExact, [], covfunc, likfunc, x_train, y_train);
    end
    
    % find best  nlml
    [results.nlml, best_hyp] = min(results.nlml);
    results.hyp = results.hyp{best_hyp};

    %% perform prediction
    [results.m, results.s2, fmu, fs2, results.p] = MTGP(results.hyp, @MTGP_infExact, [], covfunc, likfunc, x_train, y_train, x_test, y_test);

    % reshape of results
    results.m = reshape(results.m,[opt.end-opt.start+1 num_dim]);
    results.s2 = reshape(results.s2,[opt.end-opt.start+1 num_dim]);
    results.p = exp(reshape(results.p,[opt.end-opt.start+1 num_dim]));

    %% compute RMSE for training and test data for each dimension
    for cnt_dim = 1:num_dim
        results.m(:,cnt_dim) = results.m(:,cnt_dim) + y_train_mean(cnt_dim);

        index_test_data = [opt.start:opt.end];

        index_test_data(ismember(index_test_data,opt.training_data{cnt_dim})) = [];

        results.rmse_test(cnt_dim) = rms(results.m(index_test_data-opt.start+1,cnt_dim)-...
            y(index_test_data,cnt_dim));

        results.rmse_train(cnt_dim) = rms(results.m(opt.training_data{cnt_dim}-opt.start+1,cnt_dim)-...
            y(opt.training_data{cnt_dim},cnt_dim));

    end

    % compute resulting K_f matrix
    vec_dim = 1:num_dim;
    L = zeros(num_dim,num_dim);
    for cnt_dim = 1:num_dim
        L(cnt_dim,1:vec_dim(cnt_dim)) = [results.hyp.cov(sum(vec_dim(1:cnt_dim-1))+1:sum(vec_dim(1:cnt_dim-1))+vec_dim(cnt_dim))];
    end
    results.K_f =  L*L';
    
    
    MTGP_results{count} = results;

    est_hyp(count,:) = results.hyp.cov(1:3);
    
    % normalization of K_f matrix
    [a, Kc_n]= normalize_Kc(est_hyp(count,:),num_dim);
    MTGP_cc_n(count,1) = Kc_n(2,1);

    % print results on console:
    disp(['Estimated cross correlation covariance Kc_n:']);
    Kc_n

    
    % Pearsons correlation coefficient of the output function
    a = corrcoef(results.m(:,1),results.m(:,2));
    Pear_cc_output(count,1) = a(2);

    % Pearsons correlation coefficient of the training data
    a = corrcoef(y1(opt.training_data{1}),y2(opt.training_data{2}));
    Pear_cc_input(count,1) = a(2);
    
    
    %% plot basic results
    if opt.show == 1 
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
            f = [results.m(:,cnt_dim)+2*sqrt(results.s2(:,cnt_dim)); flipdim(results.m(:,cnt_dim)-2*sqrt(results.s2(:,cnt_dim)),1)];
            fill([t(opt.start:opt.end); flipdim(t(opt.start:opt.end),1)], f, [0.7 0.7 0.7],'EdgeColor','none')

            % plot mean org signal
            plot(t(opt.start:opt.end),y(opt.start:opt.end,cnt_dim));

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
            plot(t(opt.start:opt.end),results.p(1:opt.end-opt.start+1,cnt_dim));

            title('probability p');
            axis tight
            xlabel('time [s]');
            ylabel('p');
        end
    end
    
    clear results
end


%% plot results 
figure
h=figure('Position',[1 1 1400 300]);
semilogx(scaling_factor(3:end),Pear_cc_input(3:end),'o--')
hold on
%semilogx(scaling_factor(3:end),Pear_cc_output(3:end),'d--')
semilogx(scaling_factor(3:end),MTGP_cc_n(3:end),'*r--');
legend('Pearsons CC_{input}','MTGP - norm');
ylabel('correlation');
xlabel('noise scaling factor');
