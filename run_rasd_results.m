%% Particle Gibbs with Ancestor Sampling Duffing

close all
clear
clc

addpath('./functions/')
addpath('./data/')

rng(1);

scale = 1;
storefigs = 1;
save_loc = '';
plt = 0;

% Load data
load('duffing_data2.mat') % Load in data excited with OddMultisine



%%
disc = 'euler'; % Use euler scheme
T = size(y,1);
dt = 1/fs;
rng(2);

% Initial Guess
props.m = m+0.1*m*randn/round(m,1);
props.k = k+0.1*k*randn/round(k,1);
props.c = c+0.1*c*randn/round(c,1);
props.k3 = k3+0.1*k3*randn/round(k3,1);
props.xstar = y+[0.1 0.1].*rms(y).*randn(size(y));

Q = [eps 0; 0 1e-14];

% Set up model
process_model = @(xx,tt) duffing_step(xx,tt,dt,props,tup,F,disc,scale);
proposal = @(xx,tt) mvnrnd(process_model(xx,tt),Q);
proposalpdf = @(xn,xx,tt) gaussLik(xn,process_model(xx,tt),Q);
observation_model = @(xx,tt) xx(:,1);
likelihood = @(yy,xx,tt) gaussLik(yy,observation_model(xx,tt),0.5*rms(y(:,1)));
dynamics.proposal = proposal;
dynamics.proposalpdf = proposalpdf;
dynamics.likelihood = likelihood;

% Prior
params.mu = [0;0];
params.sigma = 0.005*eye(2);
prior = @(n) prior_sampler(n,'gaussian',params);

props.prior = prior;
props.N_particles = 50; % 50 particles
props.t = t;
props.sampler = 'syst'; % Use systematic resampling

% Noisy observations
dynamics.y = y+0.5*rms(y).*randn(size(y));

% MCMC settings
nburn = 250;
nsamp = 14250;
K = nburn+nsamp; % number of MCMC steps

phi = @(x,f) [f...
    -x(:,1) ...
    -x(:,2)*scale...
    -x(:,1).^3].*dt./scale; % Basis expansion

ps_lik = @(x,y) sum(gaussLik(y,x,1)); % Calculate joint likelihoods
%%

Nth  = 4; % 4 parameters

% Priors for IW
M = [0 0 1 0;
    1/m k/m c/m k3/m];

V = 1e-3*eye(4);
VU = chol(V);
Lam = eye(2);
ell = 1;
A = M;

% Preallocate
Qsamps = NaN(2,2,K);
lik = NaN(K,1);
xk = NaN(T,2,K);
xk(:,:,1) = props.xstar;


%%
rng(2); % Repeatability
th_true = [1/m,k/m,c/m,k3/m];
th_scales = 1*10.^floor(log(abs(th_true))./log(10));
m0 = [1/m k/m c/m k3/m] + 0.5.*[1/m k/m c/m k3/m].*randn(1,4)./round(th_true,1); % Prior with random offset
t0 = 1./2./([1/m k/m c/m k3/m]); % Precision


%%

th_samps = NaN(K,4);
mm = th_samps;
th_samps(1,:) = m0;
tausamp = ones(K,1);
tausamps = ones(K,1);
N = T-1;
resampled_x = zeros(size(props.xstar));
x = props.xstar;

tic % timing

for kk = 2:K
    
    fprintf('Gibbs Sample %i/%i \n',kk,K)
    
    % Run CPF with PR
    pf = bootstrapCPF_PR(dynamics,props);
    
    
    % Sample new trajectory from CPF
    ind = randi(props.N_particles);
    props.xstar = squeeze(pf.xpaths(ind,:,:))';
    
    % Sample new parameters
    resampled_x = resampled_x+(x==props.xstar); % Count number of times resampled for path degeneracy
    x = props.xstar;
    
    % Set up linear regression
    xt = x(1:end-1,:);
    xtt = x(2:end,:);
    
    M = [0 0 1 0;
        th_samps(kk-1,:)];
    
    % Basis expansion
    phix = [f(1:end-1)...
        -xt(:,1) ...
        -xt(:,2)*scale...
        -(xt(:,1)).^3].*dt./scale;
    
    % Scaling for numerical stability
    phiscale = 1*10.^floor(log(abs(mean(phix,1)))./log(10))*100;
    phix = phix./phiscale;
    
    
    % 1D Bayes Linear Regression over parameters (must use 1st order disc!)
    yi = (xtt(:,2)-xt(:,2));
    xss = sum(phix.^2);
    
    tausamp(kk) = tausamp(kk-1);
    
    th_samps(kk,:) = th_samps(kk-1,:);
    for tt = 1:4
        
        tp = setdiff(1:4,tt);
        res = yi-(sum(th_samps(kk,tp).*(phix(:,tp).*phiscale(tp)),2));
        sig = 1/(t0(tt)+tausamp(kk)*(xss(tt).*phiscale(tt).^2));
        mu = (t0(tt)*m0(tt)+tausamp(kk)*(res'*(phix(:,tt).*phiscale(tt)))).*sig;
        
        th_samps(kk,tt) = mvnrnd(mu,sqrt(sig));
        
        mm(kk,tt) = mu;
    end
    
    % Sample noise since now 1D problem
    a = 1;
    b = 1;
    np = size(yi,1);
    res = yi-(sum(th_samps(kk,:).*(phix.*phiscale),2));
    tausamps(kk) = gamRnd(a+np/2,(b+sum(res.^2)/2),1);
    tausamp(kk) = tausamps(kk);
 
    
    % Update Model
    %     fprintf('Lik = %.3e\n',ps_lik(y(:,1),x(:,1)))
    lik(kk) = ps_lik(y(:,1),x(:,1));
    if kk > 0
        props.m = 1./th_samps(kk,1);
        props.k = th_samps(kk,2)*props.m;
        props.c = th_samps(kk,3)*props.m;
        props.k3 = th_samps(kk,4)*props.m;
    end
    %
    process_model = @(xx,tt) duffing_step(xx,tt,dt,props,tup,F,disc,scale);
    dynamics.proposal = @(xx,tt) mvnrnd(process_model(xx,tt),Q);
    proposalpdf = @(xn,xx,tt) gaussLik(xn,process_model(xx,tt),Q);
    dynamics.proposalpdf = proposalpdf;
    
    % Store paths
    xk(:,:,kk) = props.xstar;
end

pgas_timer = toc;

%% PLOTTING

cmap = lines(5);

%%


if ~isdir(['./outputs/',save_loc])
    mkdir(['./outputs/',save_loc])
end

th_samps_pos = keep_positive(th_samps(nburn+1:end,:));
mth = mean(th_samps_pos(nburn+1:end,:));
th_true = [1/m,k/m,c/m,k3/m];
% th_samps_pos(:,1) = 1./th_samps_pos(:,1);
% th_samps = [ms,ks.*kscale,cs,k3s.*k3scale];
varname = {'$1/m$','$k/m$','$c/m$','$k_3/m$'};
figure;
for kk = 1:4
    subplot(2,2,kk)
    plotProbs(th_samps_pos(nburn+1:end,kk),varname{kk});
%     line(keep_positive([m0(kk) m0(kk)]),ylim,'Color',[1 1 0],'LineWidth',1)
    line([th_true(kk) th_true(kk)],ylim,'Color',[0 1 0],'LineWidth',1)
    %     xlims = xlim;
    %     xlim([0.95*min(min(th_samps_pos(:,kk)),th_true(kk)) 1.05*max(max(th_samps_pos(:,kk)),th_true(kk))])
end

if storefigs   
    save_figs(['./outputs/',save_loc],'/transform_distributions')  
end

map_trans = mean(th_samps_pos(nburn+1:end,:));
percent_err_trans = (map_trans-th_true)./th_true*100;


%%
varname = {'$m$','$k$','$c$','$k_3$'};
figure;
th_true = [m,k,c,k3];
pm0 = [1./m0(1) m0(2:4)./m0(1)];
th_samps_pos_split = [1./th_samps_pos(:,1),bsxfun(@rdivide,th_samps_pos(:,2:4),th_samps_pos(:,1))];
for kk = 1:4
    subplot(2,2,kk)
    plotProbs(th_samps_pos_split(nburn+1:end,kk),varname{kk},1000);
    line([th_true(kk) th_true(kk)],ylim,'Color',[0 1 0],'LineWidth',1)
    %     line([pm0(kk) pm0(kk)],ylim,'Color',[1 1 0],'LineWidth',1)
    xlims = xlim;
    %         xlim([0.9*min([min(th_samps_pos_split(:,kk)),th_true(kk)]) 1.1*max([max(th_samps_pos_split(:,kk)),th_true(kk)])])
    %      xlim([0.1*th_true(kk) 2.0*th_true(kk)])
end

map_split = mean(th_samps_pos_split(nburn+1:end,:));
percent_err_split = (map_split-th_true)./th_true*100;

if storefigs
    save_figs(['./outputs/',save_loc],'/parameter_distributions')
end

%%
figure;
subplot(211)
plot(squeeze(xk(:,1,nburn+1:end)),'Color',[0 0 0 0.005],'LineWidth',0.1);
hold on;
plot(mean(squeeze(xk(:,1,nburn+1:end)),2),'LineStyle','--','Color',cmap(3,:));
% plot(squeeze(x(1,1,:))','r');
plot(y(:,1),'Color',cmap(5,:));
xlabel('Time')
ylabel('Displacement (m)')
title('Sampled Paths After Burn-In')
subplot(212)
plot(squeeze(xk(:,2,nburn+1:end)),'Color',[0 0 0 0.005],'LineWidth',0.1);
hold on;
plot(mean(squeeze(xk(:,2,nburn+1:end)),2),'LineStyle','--','Color',cmap(3,:));
plot(y(:,2),'Color',cmap(5,:));
xlabel('Time')
ylabel('Velocity (m/s)')
ylim([-1 1])
if storefigs
    save_figs(['./outputs/',save_loc],'/sampled_paths_post_burn')    
end


%%
figure;
subplot(211)
plot(squeeze(xk(:,1,:)),'Color',[0 0 0 0.005],'LineWidth',0.1);
hold on;
plot(mean(squeeze(xk(:,1,:)),2),'LineStyle','--','Color',cmap(3,:));
% plot(squeeze(x(1,1,:))','r');
plot(y(:,1),'Color',cmap(5,:));
xlabel('Time')
ylabel('Displacement (m)')
title('All Sampled Paths')
subplot(212)
plot(squeeze(xk(:,2,:)),'Color',[0 0 0 0.005],'LineWidth',0.1);
hold on;
plot(mean(squeeze(xk(:,2,:)),2),'LineStyle','--','Color',cmap(3,:));
plot(y(:,2),'Color',cmap(5,:));
xlabel('Time')
ylabel('Velocity (m/s)')
ylim([-1 1])

if storefigs
    save_figs(['./outputs/',save_loc],'/all_sampled_paths')
end


%% Save results
save(['./outputs/',save_loc,'/degenerate_PGAS_PR_data'])