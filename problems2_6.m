%% in sample (training set)

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

%% out of sample (test set)
test = load('test.txt');
Xtest = test(:, 1:2); ytest = test(:, 3);
Xtest = transform(Xtest); n = size(Xtest, 1);

% add intercept
Xtest = [ones(n, 1) Xtest];

predictions = sign(Xtest * w);

% determine Eout
Eout = length(predictions(predictions~=ytest))/ length(ytest);
fprintf('Eout: %f\n', Eout);

