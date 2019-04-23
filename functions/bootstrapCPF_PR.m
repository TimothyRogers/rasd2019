function [pf] = bootstrapCPF_PR(dynamics,props)

% Conditional Bootstap Particle Filter with AS and particle rejuvenation

diagnostics = 0;

prop = dynamics.proposal; % Proposal distribution sampler from f(x,t)
proppdf = dynamics.proposalpdf; % Log Proposal distribution pdf log(f(x,t))
lik = dynamics.likelihood; % Log Likelihood model log(p(y | g(x,t)))
y = dynamics.y; % Measurements

xstar = props.xstar; % Trajectory to condition on
N = props.N_particles; % Number of particles

t = props.t; % Time vector
T = length(t);
sampler = props.sampler; % Resampling strategy
x0 = feval(props.prior,N-1); % Samples from prior

[~,D] = size(x0); % Dimensionality

% Preallocate
x = NaN(N,D,T);
w = NaN(N,T);
a = NaN(N,T);

% Initial Conditions
x(1:N-1,:,1) = x0; % Samples from prior
x(N,:,1:T) = xstar'; % Start of reference trajectory
xpaths = x; % Preallocate ancestral paths
w(:,1) = lik(y(1),squeeze(x(:,:,1)),t(1)); % Weighting
% ess = sum(w(:,1))^2./sum(w(:,1).^2);
% fprintf('ESS = %.3e\n',ess) 
w(:,1) = exp(w(:,1)-max(w(:,1)));
w(:,1) = w(:,1)./sum(w(:,1));
a(:,1) = 1:N;

% Sample Xi for t = 1
panc = w(:,1).*proppdf(x(N,:,2),x(:,:,1),t(1));
panc = exp(panc-max(panc));
panc = panc./sum(panc);
ind = min(N+1-sum(rand()<=cumsum(panc)),N);
a(N,1) = ind; % This is redundant since there is no ancestor to x|t=1
x(N,:,1) = x(ind,:,1);


for k = 2:T
    
    %% Ancestor sampling except reference
%     if ess < 0.4*N || k == T
        a(1:N-1,k) = resamp(w(:,k-1),sampler,N-1);
%     else
%         a(1:N-1,k) = 1:N-1;
%     end
    
    %% Propgation except reference
    x(1:N-1,:,k) = x(a(1:N-1,k),:,k-1);
    x(1:N-1,:,k) = prop(squeeze(x(1:N-1,:,k)),t(k-1));
    % Propogate reference
%     x(N,:,k) = prop(squeeze(x(N,:,k-1)),t(k-1));
    
    %% Ancestor sampling with Particle Rejuvenation using CIS kernel
    
    l = 1; % Number of steps ahead fixed at 1 for now...
    % Propagate again
    if k < T
        gxs = lik(y(k),squeeze(x(:,:,k)),t(k)); % Calculate likelihood of current particles
%         fxs = prop(squeeze(x(:,:,k)),t(k)); % Propagate current particles
%         not needed for l = 1;
        fprime = proppdf(x(N,:,k+1),x(:,:,k),t(k)); % Likelihood of reference at k+l
        panc = w(:,k-1)+fprime+gxs; % Eq (20) Lindsten 2015 PAS for near-degenerate
    else
        panc = w(:,k-1)+proppdf(x(N,:,k),x(:,:,k-1),t(k-1)); % At T usual ancestor sampling
    end
    panc = exp(panc-max(panc));
    panc = panc./sum(panc);
    ind = min(N+1-sum(rand()<=cumsum(panc)),N);
    a(N,k) = ind; % Update Ancestors
    x(N,:,k) = x(ind,:,k); % Update x prime
    
    if diagnostics
        if mod(k,20)==0
            close
            figure
            plot(squeeze(x(1:N-1,1,:))','Color',[0 0 0 0.1])
            hold on
            plot(xstar(:,1),'b')
            plot(squeeze(x(N,1,:))','r')
            plot(y(:,1),'Color',[1 0.4 0])
            pause
        end
    end
%     
%     xpaths(:,:,1:k-1) = xpaths(a(:,k),:,1:k-1); % Update ancestral paths
%     xpaths(:,:,k) = x(:,:,k);
    
    %% Weighting
    w(:,k) = lik(y(k),squeeze(x(:,:,k)),t(k));
%     ess = sum(w(:,k))^2./sum(w(:,k).^2);
%     fprintf('ESS = %.3e\n',ess) 
    w(:,k) = exp(w(:,k)-max(w(:,k)));
    w(:,k) = w(:,k)./sum(w(:,k));
    
end


xpaths(:,:,end) = x(:,:,end);
for k = T-1:-1:1    
    xpaths(:,:,k) = x(a(:,k+1),:,k);
end
    
%%


%%
% Outputs
pf.a = a;
pf.x = x;
pf.w = w;
pf.xpaths = xpaths;




end