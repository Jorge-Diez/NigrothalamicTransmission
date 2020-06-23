function [ output ] = myfunc( input_argument )
%myfunc - A simple function to demonstrate parameter sweeps using Sun Grid Engine (SGE).


%If you call this function from MATLAB, the input will be a double.
%If you call this function from a bash script, the input will be a string.
%handle the conversion from string here
if(ischar(input_argument))
  input_argument = str2double(input_argument);
end

output = input_argument /10;

end