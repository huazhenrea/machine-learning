function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h_theta=sigmoid(X*theta);

theta0=theta(2:end);
X1=X(:,2:end);
grad1=grad(2:end);

sum1=0;
sum2=zeros(length(theta)-1,1);
sum3=0;

for i=1:m
    sum1 = sum1+(h_theta(i)-y(i)).*X(i,1);
    grad0=(1/m)*sum1;
    sum2 = sum2+(h_theta(i)-y(i)).*X1(i,:)';
    grad1=(1/m).*sum2+(lambda/m)*(theta0);
    grad=[grad0;grad1];
    
    
    sum3= sum3+((-y(i))*log(h_theta(i))-(1-y(i))*log(1-h_theta(i)));
    J=(1/m)*sum3+(lambda/(2*m))*sum(theta0.^2);
end 



% =============================================================

end
