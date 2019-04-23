function [x] = duffing_step( x, t, dt, props, tAll,fAll,  method,scale )

if nargin < 7
    method = 'rk5';
end

if nargin < 8
    scale = 1;
end

odefun = @(tt,xx) dydt_duffing(tt,xx,tAll,fAll,props.m,props.k,props.c,props.k3,scale);

ti = t;
if isfield(props,'usFactor')
    hi = dt./props.usFactor;
else
    hi = dt;
end
yi = x;

switch method
    
    case 'euler'
        x = yi + hi*feval(odefun,ti,yi);

    case 'modeuler'
        % Modified Eulers Method
        x = yi + hi*feval(odefun,ti+0.5*hi,yi+0.5*feval(odefun,ti,yi));
    
    
    case 'rk3'
        % 3rd Order R-K Algorithm
        
        F(:,:,1) = feval(odefun,ti,yi);
        F(:,:,2) = feval(odefun,ti+1/3*hi,yi+1/3*hi.*F(:,:,1));
        F(:,:,3) = feval(odefun,ti+2/3*hi,yi+2/3*hi.*F(:,:,2));
        x = yi + 0.25*hi*(F(:,:,1)+3.*F(:,:,3));
        
        
        
    case 'rk5'
        % Employ 5th order Runge-Kutta algorithm
        
        F(:,:,1) = feval(odefun,ti,yi);
        F(:,:,2) = feval(odefun,ti+0.25*hi,yi+0.25*hi*F(:,:,1));
        F(:,:,3) = feval(odefun,ti+0.25*hi,yi+0.125*hi*F(:,:,1)+0.125*hi*F(:,:,2));
        F(:,:,4) = feval(odefun,ti+0.5*hi,yi-0.5*hi*F(:,:,2)+hi*F(:,:,3));
        F(:,:,5) = feval(odefun,ti+0.75*hi,yi+0.1875*hi*F(:,:,1)+0.5625*hi*F(:,:,4));
        F(:,:,6) = feval(odefun,ti+hi,yi-3/7*hi*F(:,:,1)+2/7*hi*F(:,:,2)+12/7*hi*F(:,:,3)-12/7*hi*F(:,:,4)+8/7*hi*F(:,:,5));
        x = yi + (hi/90)*(7*F(:,:,1)+32*F(:,:,3)+12*F(:,:,4)+32*F(:,:,5)+7*F(:,:,6));
        
        
        
        
end