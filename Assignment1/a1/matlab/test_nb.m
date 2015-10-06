function [prediction, accuracy] = test_nb(test_data, test_label, log_prior, class_mean, class_var)
% Test a learned Naive Bayes classifier.
%
% Usage:
%   [prediction, accuracy] = test_nb(test_data, test_label, log_prior, class_mean, class_var)
%
% test_data: n_dimensions x n_examples matrix
% test_label: 1 x n_examples binary label vector
% log_prior: 1 x 2 vector, log_prior(i) = log p(C=i)
% class_mean: n_dimensions x 2 matrix, class_mean(:,i) is the mean vector for class i.
% class_var: n_dimensions x 2 matrix, class_var(:,i) is the variance vector for class i.
%
% prediction: 1 x n_examples binary label vector
% accuracy: a real number
%

K = length(log_prior);
n_examples = size(test_data, 1);

log_prob = zeros(n_examples, K);

for k = 1 : K
    mean_mat = repmat(class_mean(k, :), [n_examples, 1]);
    var_mat = repmat(class_var(k, :), [n_examples, 1]);
    log_prob(:, k) = sum(-0.5 * (test_data - mean_mat).^2 ./ var_mat - 0.5 * log(var_mat), 2) + log_prior(k);
end

[~, prediction] = max(log_prob, [], 2);
prediction = prediction - 1;
accuracy = mean(prediction == test_label);

return
end

