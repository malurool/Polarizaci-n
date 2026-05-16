function [wavelength, transmittance] = read_transmittance_file(filename)
    % READ_TRANSMITTANCE_FILE Reads a spectral transmittance file.
    % This function skips lines that start with '#' and extracts numerical data.
    %
    % Inputs:
    %   filename - Name of the file to read
    %
    % Outputs:
    %   wavelength - Column vector of wavelength values
    %   transmittance - Column vector of transmittance values

    % Open the file for reading
    fid = fopen(filename, 'r');
    
    % Check if file opened successfully
    if fid == -1
        error('Could not open file: %s', filename);
    end
    
    % Initialize an empty array to store the data
    data = [];

    % Read the file line by line
    while ~feof(fid)
        line = fgetl(fid); % Read a line from the file
        
        % Check if the line is not a comment
        if ~startsWith(line, '#')
            % Convert the line to numerical data and store it
            values = str2num(line); %#ok<ST2NM> % Convert string to numbers
            if ~isempty(values)
                data = [data; values]; %#ok<AGROW> % Append to data array
            end
        end
    end

    % Close the file
    fclose(fid);

    % Extract wavelength and transmittance columns if data is not empty
    if ~isempty(data)
        wavelength = data(:, 1);
        transmittance = data(:, 2);
    else
        error('No valid data found in file: %s', filename);
    end
end
