function [samps] = gamRnd(a,b,n)

% Sample from Gamma defined in terms of shape and rate
if nargin<3
    n=1;
end

% Gamma(alpha,b) generator using Marsaglia and Tsang method
% Algorithm 4.33
samps = NaN(n,1);
for k = 1:n
    if a>1
        d=a-1/3; c=1/sqrt(9*d); flag=1;
        while flag
            Z=randn;
            if Z>-1/c
                V=(1+c*Z)^3; U=rand;
                flag=log(U)>(0.5*Z^2+d-d*V+d*log(V));
            end
        end
        samps(k)=d*V/b;
    else
        samps(k)=gamRnd(a+1,b);
        samps(k)=samps(k)*rand^(1/a);
    end
end


end