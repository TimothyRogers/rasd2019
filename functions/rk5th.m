function [ Y ] = rk5th( odefun,tspan,y0,varargin )
%rk5th Solve a system of differential equations using a fixed step 5th
%order Runge-Kutta method (Butcher's Method 1964).
%   Y = rk5th( odefun, tspan, y0,  varargin )
%   
%   Inputs:
%       odefun = function handle for function defining derivative
%                relationship.
%       tspan = vector of sampling times
%       y0 = vector of initial conditions y0(1:dofs) = displacement,
%            y0(dofs+1:2*dofs) = velocity
%       varargin = addition arguments to pass to odefun
%
%   Outputs:
%       Y = vector of numerical solution at each timestep
%
% TIM ROGERS, UNIVERSITY OF SHEFFIELD, 2014


% Input checking
if ~isnumeric(tspan)
    error('numeric_time_steps');
end

if ~isnumeric(y0)
    error('numeric_initial_conditions');    
end

h = diff(tspan);
if any(sign(h(1))*h <= 0)
  error('tspan_order_error');
end  

% Calculate initial conditions output

try
  f0 = feval(odefun,tspan(1),y0,varargin{:});
catch
    y0 = y0';
    try
        f0 = feval(odefun,tspan(1),y0,varargin{:});
    catch err
        %error('t0_evaluation_error');
        rethrow(err);
    end
end  

y0 = y0(:);   % Make a column vector.
if ~isequal(size(y0),size(f0))
  error('y0_f0_size_error');
end

% Employ 5th order Runge-Kutta algorithm
neq = length(y0);
N = length(tspan);
Y = zeros(neq,N);
F = zeros(neq,6);

Y(:,1) = y0;
for i = 2:N
    ti = tspan(i-1);
    hi = h(i-1);
    yi = Y(:,i-1);
    F(:,1) = feval(odefun,ti,yi,varargin{:});
    F(:,2) = feval(odefun,ti+0.25*hi,yi+0.25*hi*F(:,1),varargin{:});
    F(:,3) = feval(odefun,ti+0.25*hi,yi+0.125*hi*F(:,1)+0.125*hi*F(:,2),varargin{:});
    F(:,4) = feval(odefun,ti+0.5*hi,yi-0.5*hi*F(:,2)+hi*F(:,3),varargin{:});
    F(:,5) = feval(odefun,ti+0.75*hi,yi+0.1875*hi*F(:,1)+0.5625*hi*F(:,4),varargin{:});
    F(:,6) = feval(odefun,ti+hi,yi-3/7*hi*F(:,1)+2/7*hi*F(:,2)+12/7*hi*F(:,3)-12/7*hi*F(:,4)+8/7*hi*F(:,5),varargin{:});
    Y(:,i) = yi + (hi/90)*(7*F(:,1)+32*F(:,3)+12*F(:,4)+32*F(:,5)+7*F(:,6));
end


% Y = Y.';