function X_transform = transform(X)
    x1 = X(:, 1); x2 = X(:, 2);
    x3 = x1.^2; x4 = x2.^2;
    x5 = x1 .* x2;
    x6 = abs(x1 - x2); x7 = abs(x1 + x2);
    X_transform = [x1 x2 x3 x4 x5 x6 x7];
end