function [log_prior, class_mean, class_var] = train_nb(train_data, train_label)
% Train a Naive Bayes binary classifier.  All conditional distributions are 
% Gaussian.
% 
% Usage:
%   [log_prior, class_mean, class_var] = train_nb(train_data, train_label)
%
% train_data: n_dimensions x n_examples matrix
% train_label: 1 x n_examples binary label vector
%
% log_prior: 1 x 2 vector, log_prior(i) = log p(C=i)
% class_mean: n_dimensions x 2 matrix, class_mean(:,i) is the mean vector for
%       class i.
% class_var: n_dimensions x 2 matrix, class_var(:,i) is the variance vector for
%       class i.
%

SMALL_CONSTANT = 1e-10;

[n_examples, n_dims] = size(train_data);
K = 2;

prior = zeros(K, 1);
class_mean = zeros(K, n_dims);
class_var = zeros(K, n_dims);

for k = 1 : K
    prior(k) = mean(train_label == (k-1));
    class_mean(k, :) = mean(train_data(train_label == (k-1), :), 1);
    class_var(k, :) = var(train_data(train_label == (k-1), :), 0, 1);
end

class_var = class_var + SMALL_CONSTANT;
log_prior = log(prior + SMALL_CONSTANT);

return
end

