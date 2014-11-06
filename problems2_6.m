%% in sample (training set) without regularization

train = load('train.txt');
X = train(:, 1:2); y = train(:, 3);
X = transform(X); m = size(X, 1);

% add intercept
X = [ones(m, 1) X];

% use normal equation to find the weights
w = pinv(X' * X) * X' * y;

% extimate y according to w
yEst = sign(X * w);

% determine Ein (prediction accuracy)
Ein = length(yEst(yEst~=y))/ length(y);
fprintf('Ein: %f\n', Ein);

%% out of sample (test set) without regularization
test = load('test.txt');
Xtest = test(:, 1:2); ytest = test(:, 3);
Xtest = transform(Xtest); n = size(Xtest, 1);

% add intercept
Xtest = [ones(n, 1) Xtest];

predictions = sign(Xtest * w);

% determine Eout
Eout = length(predictions(predictions~=ytest))/ length(ytest);
fprintf('Eout: %f\n', Eout);

%% with regularization
lambda = 10^-3; k = size(X, 2);
I = eye(k); I(1, 1) = 0;

% use normal equations to find the weights
w_reg = pinv(X' * X + lambda * I) * X' * y;

% extimate y according to w_reg
yEst_reg = sign(X * w_reg);

% determine Ein with regularization
Ein_reg = length(yEst_reg(yEst_reg~=y))/ length(y);
fprintf('Ein with regularization: %f\n', Ein_reg);

predictions_reg = sign(Xtest * w_reg);

% determine Eout with regularization
Eout_reg = length(predictions_reg(predictions_reg~=ytest))/ length(ytest);
fprintf('Eout with regularization: %f\n', Eout_reg);

%%