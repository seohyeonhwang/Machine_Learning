%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data_train = load('Train_Data.txt');
z(:,1)= (data_train(:,2)+data_train(:,3)+data_train(:,4)+data_train(:,5))/4;
z(:,2)= (data_train(:,6)+data_train(:,7)+data_train(:,8)+data_train(:,9)+data_train(:,10))/5;

X = z; y = data_train(:, 11);
y(find(y == 2)) = 0;
y(find(y == 4)) = 1;


data_test = load('Test_Data.txt');
a(:,1)= (data_test(:,2)+data_test(:,3)+data_test(:,4)+data_test(:,5))/4;
a(:,2)= (data_test(:,6)+data_test(:,7)+data_test(:,8)+data_test(:,9)+data_test(:,10))/5;

X_test = a; y_test = data_test(:, 11); 
y_test(find(y_test == 2)) = 0;
y_test(find(y_test == 4)) = 1;



%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).


% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 0;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

expected = [0.0085, 0.0188, 0.0001, 0.0503, 0.0115];

standard_error=(expected - grad(1:5))/expected*100;
fprintf('standard_error is %f:', standard_error);

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0;

for i=1:5:100
    lambda=i;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Total test Accuracy = %f\n', 100 - (mean(double(p == y))*100 - 34.5)/34.5);

% Compute accuracy on our training set

X_test = mapFeature(X_test(:,1), X_test(:,2));
p = predict(theta, X_test);
fprintf('Using length(find(y_test==1)), test data class distribution is 22.11(benign) & 77.89(malignant)\n')
fprintf('Test Accuracy: %f\n', mean(double(p == y_test)) * 100);
fprintf('Total test Accuracy = %f\n', 100 - (mean(double(p == y_test))*100 - 22.11)/22.11);
end