function [ydot] = dydt_duffing(t,y,tsamp,f,m,k,c,k3,scale)

if size(y,2)==2 && size(y,1)~=2
    y = y';
    fl = true;
else
    fl =false;
end

% if t>tsamp(end)
%     error('Time beyond Limit!!!!')
% end

ft = f(tsamp==t);
if isempty(ft)
   ft = interp1(tsamp,f,t,'makina');
end

if nargin <9
    scale = 1;
end

ydot = NaN(size(y));
ydot(1,:) = y(2,:).*scale; % Scaling!
ydot(2,:) = (ft/m ...
    -k/m.*y(1,:) ...
    -c/m.*y(2,:).*scale ...
    -k3/m.*y(1,:).^3)./scale;

% ydot(1,:) = y(2,:);
% ydot(2,:) = ft/m ...
%          -k/m.*y(1,:) ...
%          -c/m.*y(2,:) ...
%          -k3/m.*y(1,:).^3;

if fl
    ydot = ydot';
end

end