function plot_digits(digit_matrix)
% plot_digits: Visualizes each example contained in digit_matrix.
%
% Note: N is the number of examples 
%       and M is the number of features per example.
%
% Inputs:
%   digits: N x M matrix of pixel intensities.
%
% Note: After calling this function, re-size your figure window so that
%       each pixel is approximately square.

CLASS_EXAMPLES_PER_PANE = 5;

% assume two evenly split classes
examples_per_class = size(digit_matrix,1)/2;
num_panes = ceil(examples_per_class/CLASS_EXAMPLES_PER_PANE);

for pane = 1:num_panes
  fprintf('Displaying pane %d/%d\n', pane, num_panes);

  % set up plot
  current_figure = figure;
  colormap('gray');
  clf;
  
  for class_index = 1:2
    for example_index = 1:CLASS_EXAMPLES_PER_PANE
      if (pane-1)*CLASS_EXAMPLES_PER_PANE + example_index > examples_per_class
        break
      end
      
      % select appropriate subplot
      digit_index = (class_index-1)*examples_per_class + ...
                    (pane-1)*CLASS_EXAMPLES_PER_PANE + example_index;
      
      subplot(2, CLASS_EXAMPLES_PER_PANE, ...
              (class_index-1)*CLASS_EXAMPLES_PER_PANE + example_index);
      
      % plot it
      current_pixels = reshape(digit_matrix(digit_index,:), 28, 28)';
      imagesc(current_pixels);
      axis off;
    end
  end
  waitfor(current_figure);
end
