function [output] = sigmoid(input)
% sigmoid: Computes the elementwise logistic sigmoid of the input.
%
% Inputs:
% 	input: Either a row vector or a column vector.
%

output = 1.0 ./ (ones(size(input)) + exp(-input));

