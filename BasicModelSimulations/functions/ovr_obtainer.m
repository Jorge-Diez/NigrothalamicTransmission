addpath(pwd)

root_dir = input("please introduce path of experiments: ", 's');
cd(root_dir);
nr_neurons = 20;

D = dir;

% Avoid . and .. folders
for k = 3:length(D)
   currdirectory = D(k).name; 
   freq_folder = fullfile(root_dir, currdirectory);
   for i = 1:nr_neurons
       neuron_folder = ['nr_neurons_' num2str(i)];
       full_path = fullfile(freq_folder, neuron_folder);
       vis_res_lumped_mats(full_path, 'MIP');
       
   end
   disp(["Directory " currdirectory " finished"])
end