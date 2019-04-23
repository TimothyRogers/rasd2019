function [x] = prior_sampler(N,dist,params)

switch lower(dist)
    case 'gaussian'
        mu = params.mu;
        sigma = params.sigma;
        x = mvnrnd(mu,sigma,N);
    otherwise
        error('Unsupported prior');
end

end