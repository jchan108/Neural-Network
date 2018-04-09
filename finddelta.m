%Joshua Chang 861169756
%5/14/2017
%CS171 assignment 3



function d = finddelta(z,w,delta)
z(1,:) = []; %remove first row of z
w(:,1) = []; %remove first row of w


for n = 1:size(z,1)
    z(n) = z(n)*(1-z(n));
end

right = transpose(w)*delta;
d = z.*right;
    
