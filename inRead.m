% Written by Mahmoud Afifi -- mafifi@eecs.yorku.ca | m.3afifi@gmail.com
% MIT License
% Requires Matlab 2019b or higher

function images = inRead(fileName)

[directory_stream1,name,ext] = fileparts (fileName);

I = imread(fileName);

directory_stream2 = strrep(directory_stream1,'stream1','stream2');

images = zeros(size(I,1),size(I,2), size(I,3) * 2, ...
    'like', I);

images(:,:,1:3) = I;

images(:,:,4:6) = imread(fullfile(directory_stream2,[name ext])); 

end