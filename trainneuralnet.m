%Joshua Chang 861169756
%5/14/2017
%CS171 assignment 3



function [W1, W2] =  trainneuralnet(X,Y,nhid,lambda)
stop = 0;
gridX = getgridpts(X)
gridX1 = [ones(size(gridX,1),1) gridX];

%randn([3 4]) returns a 3-by-4 matrix.
%need to add a column of 1 to our X
X = [ones(80,1) X];
W1 = randn(nhid,3); % 5 x 3 when nhid = 5
W2 = randn(1,nhid + 1); % 1 x 6 when nhid = 6

%runningweight = zeros(size(W1));
%runningweight2 = zeros(size(W2));
lowerror = false;
%for: while error is not low enough

while lowerror == false
runningweight = zeros(size(W1));
runningweight2 = zeros(size(W2));
for n = 1:80 %for each of our 80 examples
x = transpose(X(n,:)); %the example that we are dealing with for this iteration
a = W1*x; %returns vector of a's for first and only hidden layer 
z = sigmoid(a); %returns a vector of z's for the hidden layer
z = [1;z]; %adds 1 to the top of our vecctor (first row)
af = W2*z; %returns final a value for our neural network
zf = sigmoid(af); %returns final corresponding z value for neural network
%start backwards propogation
deltaf = zf - Y(n);
delta = finddelta(z,W2,deltaf);

gweight2 = deltaf*transpose(z); %computes our delta gradient
gweight1 = delta*transpose(x);

runningweight = runningweight + gweight1;
runningweight2 = runningweight2 + gweight2;
end %end for n = 1:80

runningweight = runningweight/80 + 2*lambda*W1;
runningweight2 = runningweight2/80 + 2*lambda*W2;

 
if (mod(stop,1000) == 0) %checking error
    runningweight
    runningweight2
end

max1 = max(abs(runningweight));
max2 = max(abs(runningweight2));


if (max1 < .0001)
    if (max2 < .0001)

        gridY = [];
     
     for n = 1:size(gridX1,1)
     x = transpose(gridX1(n,:)); %the example that we are dealing with for this iteration
a = W1*x; %returns vector of a's for first and only hidden layer 
z = sigmoid(a); %returns a vector of z's for the hidden layer
z = [1;z]; %adds 1 to the top of our vecctor (first row)
af = W2*z; %returns final a value for our neural network
zf = sigmoid(af); %returns final corresponding z value for neural network
  gridY = [gridY; zf];
     end
   
     plotdecision(X,Y,gridX,gridY)
    return
    end
end

W1 = W1 - .1*(runningweight);
W2 = W2 - .1*(runningweight2);


end %end while lowerror == false

