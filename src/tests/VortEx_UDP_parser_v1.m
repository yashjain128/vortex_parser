%% VORTEX UDP Parser
%
% Matrix format referenced is "41.127 Measurement List Matrix_RevA_8-11-22"
% Instrument data format is documented in "SEED_VortEx_BoardDataFormats".
% 
% Written for the Space and Atmospheric Instrumentation Laboratory
% by Josh Milford
% 9/20/22

%% Setup
clear; close all; fclose all; clc;

addpath('..\');

% 0 for file, 1 for UDP
readmode = 0;

% which figures to plot
plotfigure1 = true;     % ERAU instruments
plotfigure2 = true;     % Experiment
plotfigure3 = true;     % GPS plots
HKparsing = true;       % true to parse housekeeping, false to skip
HK_TV = false;           % true to express HK in physical units, false for counts

if (readmode == 0)
    % Open file for reading
    [name,path] = uigetfile({'*.udp;*.bin'},'Select a UDP recording'); 
    fid = fopen([path,name]);
elseif (readmode == 1)
    % Setup UDP Object
    LocalHost = '169.254.229.105';
    LocalPort = 5555;
    u = udpport('byte','IPV4','LocalHost',LocalHost,'LocalPort',LocalPort);
    configureMulticast(u, "226.0.0.1");
end

% Flight E-box Identifiers
mNLP_ID = bin2dec('0001 0010');
PIP_ID = bin2dec('0010 0010');
ACC_ID = bin2dec('0011 0011');

%% Fixed values
minorframelength = 2*40;
DIG_ACC_CONST = [0xF0,0xFF,0x0F];
HK_LENGTH = 11;

% The parsing strategy is to ignore UDP header information and search
% directly for frame sync words. Note that during transmission, the UDP
% header is followed by the frame sync, and then by the rest of the
% columns. For VortEx it would first be columns 39,40, then 1-38. Also note
% the swapped byte order.
% FS1 = 0b1111-1110-0110-1011 = 0xFE 0x6B = 254 107
% FS2 = 0b0010-1000-0100-0000 = 0x28 0x40 = 40 64
SYNC = [64,40,107,254];

% swap the endianness of each 32-bit group for TM matrix
kk = reshape(transpose((0:(minorframelength/4)-1)'*4 + (3:-1:0)),1,[]);

%% GPS definitions

rV_HEADER = double(['r','V','0','2','A']); % 2-byte header, 3-byte body length indicator (fixed)
rV_LENGTH = 48; % fixed message length in bytes, including header
wgs84 = wgs84Ellipsoid('meter');

% The map file is just a collection of three things:
% 1. 3D byte array consisting of (R,G,B) values for every point in a 2D
%    grid, essentially an image.
% 2. The latitude bounds of the rectangular image, stored as 1x2 double
% 3. The longitude bounds of the rectangular image, stored as 1x2 double
[newfile,path] = uigetfile('*.mat','Load Map File','map1.mat');
if newfile == 0
    warning("No map data loaded");
else
    filename=fullfile(path,newfile);
    load(filename,'-mat');
end

%% Workspace setup
% 5000 minor frames/s plus UDP overhead 44 bytes for every minor frame
% UDP bitrate = (40*2 + 44) bytes/minorframe * 8 bits/byte * 5000
% minorframes/s = 4.960 Mbps
plot_delay = .25; % seconds
udp_bitrate = 4960000; % bps

% 124 bytes per minor frame * 5000 minor frames/s = 620000 bytes/s
readlength = 124*5000*plot_delay; % when reading in real time, make this value independent of plot_delay
rawdata = zeros(readlength,1);
% An array pointer to move the last (incomplete) packet in rawdata
% to the beginning.
% readlength+1 means there is no "remainder" yet.
remainderPtr = readlength+1;

% We know that exactly minorframelength+44 bytes exist in a complete packet, so we can use
% this to check packet validity
packetlength = minorframelength+44;

% create variable for storing selected minor frame data in matrix format
TM_matrix = zeros(ceil(readlength/packetlength),minorframelength);

% Allocate memory for housekeeping parsing buffers
buffer_mNLP_HK = zeros(ceil(readlength/packetlength),1);
buffer_PIP_HK = zeros(ceil(readlength/packetlength)*2,1);
buffer_ACC_HK = zeros(ceil(readlength/packetlength),1);

% Allocate memory for parsed data
SFID = zeros(ceil(readlength/packetlength),1);
DATA_mNLP = zeros(ceil(readlength/packetlength),3);
DATA_mNLP_HK = zeros(floor(readlength/packetlength/11),11);
DATA_PIP = zeros(ceil(readlength/packetlength),2);
DATA_PIP_HK = zeros(floor(readlength/packetlength/11)*2,11);
DATA_ACC = zeros(ceil(readlength/packetlength),3);
DATA_ACC_HK = zeros(floor(readlength/packetlength/11),11);
DATA_D = zeros(ceil(readlength/packetlength),2);

% Digital accelerometer and Temp data comes in at 1/2 rate
DIG_ACC_numMinorFrames = 2;
DIG_ACC_minorframelength = 6;
DIG_ACC_bufferlength = ceil(udp_bitrate/8*plot_delay*DIG_ACC_minorframelength/packetlength)+(DIG_ACC_numMinorFrames-1)*DIG_ACC_minorframelength;
buffer_DIG_ACC = zeros(DIG_ACC_bufferlength,2);
DATA_DIG_ACC = zeros(ceil(udp_bitrate/8*plot_delay/packetlength/DIG_ACC_numMinorFrames),4);

%% Prepare figures
% Have a live plot showing the last 5 seconds of data
% 5000 minor frames/s plus UDP overhead 44 bytes for every minor frame
% UDP bitrate = (40*2 + 44) bytes/minorframe * 8 bits/byte * 5000
% minorframes/s = 4.960 Mbps
maxNumPoints = 25000;
maxNumPointsDigAcc = floor(maxNumPoints/2); %NPG -- pasted in many places [ctrl-F]
maxNumPointsD = floor(maxNumPoints/2);

% Create separate variables for storing plot YData
DATA_mNLP_Y = zeros(maxNumPoints,3);
DATA_mNLP_HK_Y = zeros(maxNumPoints,11);
DATA_PIP_Y = zeros(maxNumPoints,2);
DATA_PIP_HK_Y = zeros(maxNumPoints,11);
DATA_ACC_Y = zeros(maxNumPoints,3);
DATA_ACC_HK_Y = zeros(maxNumPoints,11);
DATA_DIG_ACC_Y = zeros(maxNumPointsDigAcc,4);
DATA_D_Y = zeros(maxNumPointsD,2);

% GPS
% Position and velocity is reported in 2's complement ECEF coordinate system
GPS_columns = zeros(ceil(readlength/packetlength),3);
GPS_onecolumn = zeros(ceil(readlength/packetlength)*3,1);
GPS_rV_packets = zeros(ceil(readlength/packetlength/rV_LENGTH),rV_LENGTH);
GPS_position_ecef = zeros(ceil(readlength/packetlength/rV_LENGTH),3);
GPS_position_geodetic = zeros(maxNumPoints,3);
GPS_position_valid = zeros(maxNumPoints,1);
GPS_velocity_ecef = zeros(ceil(readlength/packetlength/rV_LENGTH),3);
GPS_velocity_ENU = zeros(maxNumPoints,3);
GPS_velocity_valid = zeros(maxNumPoints,1);
GPS_numSats = zeros(maxNumPoints,1);

%NPG
lineStyle = '.';
ylim24 = 1.1*[-2^23 2^23]; % full range for signed 24-bit counts
ylim20 = 1.1*[-2^19 2^19]; % full range for signed 20-bit counts
ylim18 = 1.1*[-2^17 2^17]; % full range for signed 18-bit counts
ylim16 = 1.1*[-2^15 2^15] + 2^15;

fig1 = figure(1);
% fig1.Units = 'normalized';
% fig1.Position(1:2) = [0.05,0.60];
subplot(2,3,3); mNLP_1_line = plot(1:maxNumPoints,DATA_mNLP_Y(:,1),lineStyle,'MarkerSize',3); xlabel('Sample Number'); ylabel('Counts'); title('mNLP'); hold on; xlim([1,maxNumPoints]); ylim(ylim24);
                mNLP_2_line = plot(1:maxNumPoints,DATA_mNLP_Y(:,2),lineStyle,'MarkerSize',3);
                mNLP_3_line = plot(1:maxNumPoints,DATA_mNLP_Y(:,3),lineStyle,'MarkerSize',3);
subplot(2,3,1); PIP_1_line = plot(1:maxNumPoints,DATA_PIP_Y(:,1),lineStyle,'MarkerSize',3); xlabel('Sample Number'); ylabel('Counts'); title('PIP'); hold on; xlim([1,maxNumPoints]); ylim(ylim24);
                PIP_2_line = plot(1:maxNumPoints,DATA_PIP_Y(:,2),lineStyle,'MarkerSize',3);
subplot(2,3,2); ACC_1_line = plot(1:maxNumPoints,DATA_ACC_Y(:,1),lineStyle,'MarkerSize',3); xlabel('Sample Number'); ylabel('Counts'); title('ACC'); hold on; xlim([1,maxNumPoints]); ylim(ylim18);
                ACC_2_line = plot(1:maxNumPoints,DATA_ACC_Y(:,2),lineStyle,'MarkerSize',3);
                ACC_3_line = plot(1:maxNumPoints,DATA_ACC_Y(:,3),lineStyle,'MarkerSize',3);
subplot(2,3,5); DIG_ACC_1_line = plot(1:maxNumPointsDigAcc,DATA_DIG_ACC_Y(:,1),'.','MarkerSize',3); xlabel('Sample Number'); ylabel('Counts'); title('DIG ACC'); hold on; xlim([1,maxNumPointsDigAcc]); ylim(ylim20);
                DIG_ACC_2_line = plot(1:maxNumPointsDigAcc,DATA_DIG_ACC_Y(:,2),lineStyle,'MarkerSize',3);
                DIG_ACC_3_line = plot(1:maxNumPointsDigAcc,DATA_DIG_ACC_Y(:,3),lineStyle,'MarkerSize',3);                
subplot(2,3,6); D1_line = plot(1:maxNumPointsD,DATA_D_Y(:,1),lineStyle,'MarkerSize',3); xlabel('Sample Number'); ylabel('Counts'); title('Experiment D1/D2'); hold on; xlim([1,maxNumPointsD]); ylim(ylim16);
                D2_line = plot(1:maxNumPointsD,DATA_D_Y(:,2),lineStyle,'MarkerSize',3);
                
if(exist("latlim","var") && exist("lonlim","var") && exist("ZA","var"))
    fig2 = figure(2);
%     fig2.Units = 'normalized';
%     fig2.Position(1:2) = [0.05,0.10];
    subplot(1,2,1);
    imagesc(lonlim,latlim,flipud(ZA)); hold on;
    map2D_ax = gca; map2D_ax.YDir = 'normal';
    xlim(lonlim); ylim(latlim);
    GPS_map2D_line = plot(GPS_position_geodetic(1:maxNumPoints,2),GPS_position_geodetic(1:maxNumPoints,1),'r.');
    xlabel('Longitude'); ylabel('Latitude'); title('2D Position');
    
    subplot(1,2,2);
    imagesc(lonlim,latlim,flipud(ZA)); hold on;
    map3D_ax = gca; map3D_ax.YDir = 'normal'; map3D_ax.View = [-45,45];
    xlim(lonlim); ylim(latlim);
    GPS_map3D_line = plot3(GPS_position_geodetic(1:maxNumPoints,2),GPS_position_geodetic(1:maxNumPoints,1),GPS_position_geodetic(1:maxNumPoints,3),'r.');
    xlabel('Longitude'); ylabel('Latitude'); zlabel('Altitude (km)'); title('3D Position'); zlim([-0.020,160]);
end

%% Prepare file for writing
% writefilename = 'UDP_stream_parsed.bin';
% fid_write = fopen(writefilename,'w');

%% Main loop

if (readmode == 1)
    while (u.NumBytesAvailable < readlength)
        % wait until data starts coming in
    end
end

read_flag = true;
disp_message_exp = 0; % display lack of data available
tic;
plot_tic = tic;

while (read_flag)

    % indexes 1:readlength-remainderPtr+1 are populated by the previous remainder
    % remainderPtr-1 bytes are to be read from the input buffer
    if (readmode == 0)
        [A,count] = fread(fid,remainderPtr-1);
        rawdata(readlength-remainderPtr+2:readlength-remainderPtr+1+count) = A;
        % identify the indices of all elements that match the beginning of the sync words
        packet_ind = strfind(rawdata(1:readlength-remainderPtr+1+count)',SYNC);
    elseif (readmode == 1)
        rawdata(readlength-remainderPtr+2:readlength) = read(u,remainderPtr-1,"uint8");
        % identify the indices of all elements that match the beginning of the sync words
        packet_ind = strfind(rawdata',SYNC);
    end

    % Remember the index of the last frame sync, as this one can't be checked
    % using a forward difference. It will roll over to the next batch.
    remainderPtr = packet_ind(end);

    % Check that each packet is the expected length.
    % If the distance from one frame sync to the next is not equal to the
    % expected packet length, the packet associated with the first frame
    % sync is dropped.
    deltaIndex = diff(packet_ind);
    packet_ind = packet_ind(deltaIndex == packetlength);

    % use rawdata to do main experiment SFID checking
    ii = (1:length(packet_ind))';
    SFID = rawdata(packet_ind(ii) + 6);
    SFID_diff = diff(SFID);
    SFID_diff = SFID_diff + 20*(SFID_diff < 0);
    % For main experiment parsing, don't consider minor frames with
    % duplicate main experiment SFIDs
    aa = find(SFID_diff == 1);
    exp_numpackets = length(aa);

    %NPG
    if(isempty(aa))
        if(disp_message_exp > 100)
            disp('No valid packets detected in data stream')
            disp_message_exp = 0;
        end
        disp_message_exp = disp_message_exp + 1;
    else
        
        % extract the minor frames of correct length up to remainderPtr and put
        % them in a matrix
        ii = packet_ind(packet_ind < remainderPtr);
        numKAMframes = length(ii);
        TM_matrix(1:numKAMframes,1:minorframelength) = rawdata(ii' + kk);
        
        % Move incomplete packet to front of rawdata
        rawdata(1:readlength-remainderPtr+1) = rawdata(remainderPtr:readlength);
        
        % Reset the remainder pointer if the buffer has been filled with junk data
        if (remainderPtr == 1)
            remainderPtr = readlength + 1;
        end
        
        % Fill data buffers
        % N = word column # in matrix spreadsheet      
        % byte column # = 2*(2+N)-1  if first byte in word
        %               = 2*(2+N)    if second byte in word
        ii = (1:exp_numpackets)';
        DATA_mNLP(ii,1) = 2^16*TM_matrix(aa,2*(2+5)-1) + 2^8*TM_matrix(aa,2*(2+5)) + TM_matrix(aa,2*(2+6)-1);
        DATA_mNLP(ii,2) = 2^16*TM_matrix(aa,2*(2+6)) + 2^8*TM_matrix(aa,2*(2+7)-1) + TM_matrix(aa,2*(2+7));
        DATA_mNLP(ii,3) = 2^16*TM_matrix(aa,2*(2+8)-1) + 2^8*TM_matrix(aa,2*(2+8)) + TM_matrix(aa,2*(2+9)-1);
        buffer_mNLP_HK(ii) = TM_matrix(aa,2*(2+9));
        DATA_PIP(ii,1) = 2^16*TM_matrix(aa,2*(2+15)-1) + 2^8*TM_matrix(aa,2*(2+15)) + TM_matrix(aa,2*(2+16)-1);
        DATA_PIP(ii,2) = 2^16*TM_matrix(aa,2*(2+16)) + 2^8*TM_matrix(aa,2*(2+17)-1) + TM_matrix(aa,2*(2+17));
        buffer_PIP_HK(2*ii-1) = TM_matrix(aa,2*(2+18)-1);
        buffer_PIP_HK(2*ii) = TM_matrix(aa,2*(2+18));
        DATA_ACC(ii,1) = 2^10*TM_matrix(aa,2*(2+25)-1) + 2^2*TM_matrix(aa,2*(2+25)) + bitand(bitshift(TM_matrix(aa,2*(2+28)-1),-6),3);
        DATA_ACC(ii,2) = 2^10*TM_matrix(aa,2*(2+26)-1) + 2^2*TM_matrix(aa,2*(2+26)) + bitand(bitshift(TM_matrix(aa,2*(2+28)-1),-4),3);
        DATA_ACC(ii,3) = 2^10*TM_matrix(aa,2*(2+27)-1) + 2^2*TM_matrix(aa,2*(2+27)) + bitand(bitshift(TM_matrix(aa,2*(2+28)-1),-2),3);
        dig_acc_row1 = find(bitand(TM_matrix(aa,2*(2+28)-1),3)==1);
        dig_acc_row2 = find(bitand(TM_matrix(aa,2*(2+28)-1),3)==2);
        DATA_DIG_ACC(1:length(dig_acc_row1),1) = 2^12*TM_matrix(dig_acc_row1,2*(2+28)) + 2^4*TM_matrix(dig_acc_row1,2*(2+29)-1) + bitand(bitshift(TM_matrix(dig_acc_row1,2*(2+30)),4),15);
        DATA_DIG_ACC(1:length(dig_acc_row1),2) = 2^12*TM_matrix(dig_acc_row1,2*(2+29)) + 2^4*TM_matrix(dig_acc_row1,2*(2+30)-1) + bitand(TM_matrix(dig_acc_row1,2*(2+30)),15);
        DATA_DIG_ACC(1:length(dig_acc_row2),3) = 2^12*TM_matrix(dig_acc_row2,2*(2+28)) + 2^4*TM_matrix(dig_acc_row2,2*(2+29)-1) + bitand(bitshift(TM_matrix(dig_acc_row2,2*(2+29)),4),15);
        DATA_DIG_ACC(1:length(dig_acc_row2),4) = 2^8*bitand(TM_matrix(dig_acc_row2,2*(2+29)),15) + TM_matrix(dig_acc_row2,2*(2+30)-1);
        buffer_ACC_HK(1:length(dig_acc_row2)) = TM_matrix(dig_acc_row2,2*(2+30));
        % For digital accelerometer: note that duplicate or missing frames 
        % will make the lengths of dig_acc_row1 and dig_acc_row2 not equal
        % in general. This implies that DIG_ACC_1 and DIG_ACC_2 could have
        % a different number of samples than DIG_ACC_3 and DIG_ACC_T in a
        % given period. To handle this, simply circle shift by the maximum
        % of the two lengths for both sets. This may create apparent holes
        % in the data but will approximately preserve the coincidence of
        % the two sets. (approximately because both sets in a given sample 
        % period will have the same number of points but the "hole" appears
        % at the left end of the set rather than wherever it appeared).
        % A search could be done to find the location of skipped frames and
        % thus assign a hole in these spots, preserving coincidence
        % everywhere. But this is more work than it's worth. Note that
        % duplicate frames are disregarded so the only source of holes
        % would be skipped frames, which occur less frequently (~.05% of
        % frames based on Speed Demon tests).
        dig_acc_numpoints = max([length(dig_acc_row1),length(dig_acc_row2)]);
        
        sfid = TM_matrix(aa,2*(2+1));
        sfid_even = find(mod(sfid,2)==0);
        sfid_odd = find(mod(sfid,2)==1);
        % sfid runs 0->19: D1 is even, D2 is odd
        DATA_D(1:length(sfid_even),1) = 2^8*TM_matrix(sfid_even,2*(2+38)-1) + TM_matrix(sfid_even,2*(2+38));
        DATA_D(1:length(sfid_odd ),2) = 2^8*TM_matrix(sfid_odd ,2*(2+38)-1) + TM_matrix(sfid_odd ,2*(2+38));
        % The same dilemma we face with DIG_ACC is also faced with
        % experiment.
        D_numpoints = max([length(sfid_even),length(sfid_odd)]);
        
        % end filling data buffers for main exp
        
        DATA_mNLP = DATA_mNLP - (DATA_mNLP >= 2^23)*2^24;
        DATA_PIP = DATA_PIP - (DATA_PIP >= 2^23)*2^24;
        DATA_ACC = DATA_ACC - (DATA_ACC >= 2^17)*2^18;
        DATA_DIG_ACC(:,1:3) = DATA_DIG_ACC(:,1:3) - (DATA_DIG_ACC(:,1:3) >= 2^19)*2^20;
        
        % Update main instrument data plots
        DATA_mNLP_Y = circshift(DATA_mNLP_Y,-exp_numpackets);
        DATA_PIP_Y = circshift(DATA_PIP_Y,-exp_numpackets);
        DATA_ACC_Y = circshift(DATA_ACC_Y,-exp_numpackets);
        DATA_DIG_ACC_Y = circshift(DATA_DIG_ACC_Y,-dig_acc_numpoints);
        DATA_D_Y = circshift(DATA_D_Y,-D_numpoints);
        
        DATA_mNLP_Y((maxNumPoints-exp_numpackets+1):maxNumPoints,:) = DATA_mNLP(1:exp_numpackets,:);
        DATA_PIP_Y((maxNumPoints-exp_numpackets+1):maxNumPoints,:) = DATA_PIP(1:exp_numpackets,:);
        DATA_ACC_Y((maxNumPoints-exp_numpackets+1):maxNumPoints,:) = DATA_ACC(1:exp_numpackets,:);
        DATA_DIG_ACC_Y((maxNumPointsDigAcc-dig_acc_numpoints+1):(maxNumPointsDigAcc-length(dig_acc_row1)),1:2) = NaN;
        DATA_DIG_ACC_Y((maxNumPointsDigAcc-dig_acc_numpoints+1):(maxNumPointsDigAcc-length(dig_acc_row2)),3:4) = NaN;
        DATA_DIG_ACC_Y((maxNumPointsDigAcc-length(dig_acc_row1)+1):maxNumPointsDigAcc,1:2) = DATA_DIG_ACC(1:length(dig_acc_row1),1:2);
        DATA_DIG_ACC_Y((maxNumPointsDigAcc-length(dig_acc_row2)+1):maxNumPointsDigAcc,3) = DATA_DIG_ACC(1:length(dig_acc_row2),3);
        if (HK_TV)
            DATA_DIG_ACC_Y((maxNumPointsDigAcc-length(dig_acc_row2)+1):maxNumPointsDigAcc,4) = (-DATA_DIG_ACC(1:length(dig_acc_row2),4)+2078.25) / 9.05; % source: ADXL datasheet
        else
            DATA_DIG_ACC_Y((maxNumPointsDigAcc-length(dig_acc_row2)+1):maxNumPointsDigAcc,4) = DATA_DIG_ACC(1:length(dig_acc_row2),4);
        end
        DATA_D_Y((maxNumPointsD-D_numpoints+1):(maxNumPointsD-length(sfid_even)),1) = NaN;
        DATA_D_Y((maxNumPointsD-D_numpoints+1):(maxNumPointsD-length(sfid_odd )),2) = NaN;
        DATA_D_Y((maxNumPointsD-length(sfid_even)+1):maxNumPointsD,1) = DATA_D(1:length(sfid_even),1);
        DATA_D_Y((maxNumPointsD-length(sfid_odd )+1):maxNumPointsD,2) = DATA_D(1:length(sfid_odd ),2);
        
        % GPS
        GPS_columns(ii,1) = 2^8*TM_matrix(aa,2*(2+2)-1) + TM_matrix(aa,2*(2+2));
        GPS_columns(ii,2) = 2^8*TM_matrix(aa,2*(2+12)-1) + TM_matrix(aa,2*(2+12));
        GPS_columns(ii,3) = 2^8*TM_matrix(aa,2*(2+22)-1) + TM_matrix(aa,2*(2+22));
        GPS_columns(ii,4) = 2^8*TM_matrix(aa,2*(2+32)-1) + TM_matrix(aa,2*(2+32));
        
        GPS_onecolumn(1:exp_numpackets*4,1) = reshape(transpose(GPS_columns(ii,1:4)),exp_numpackets*4,1);
        GPS_stream = bitshift(GPS_onecolumn(logical(bitand(GPS_onecolumn,128))),-8);
        
        GPS_rV = strfind(GPS_stream',rV_HEADER);
        if ~isempty(GPS_rV)
            if (GPS_rV(end) + rV_LENGTH > length(GPS_stream))
                GPS_rV = GPS_rV(1:end-1);
            end
            % Calculate the CRC and compare with the reported CRC checksum
            numRVpackets = length(GPS_rV);
            GPS_rV_packets(1:numRVpackets,:) = reshape(GPS_stream(GPS_rV' + (0:rV_LENGTH-1)),numRVpackets,rV_LENGTH);
            GPS_rV_crc = zeros(length(GPS_rV),2);
            for ii = 1:length(GPS_rV)
                GPS_rV_crc(ii,1) = crc16_Dec(GPS_rV_packets(ii,1:rV_LENGTH-3));
                GPS_rV_crc(ii,2) = 256*GPS_rV_packets(ii,rV_LENGTH-1) + GPS_rV_packets(ii,rV_LENGTH-2);
            end
            GPS_rV_valid = find(GPS_rV_crc(:,1) == GPS_rV_crc(:,2));
            GPS_rV = GPS_rV(GPS_rV_valid);
            numRVpackets = length(GPS_rV);
            
            if ~isempty(GPS_rV) % if there are any valid packets
                GPS_position_geodetic = circshift(GPS_position_geodetic,-numRVpackets);
                GPS_position_valid = circshift(GPS_position_valid, -numRVpackets);
                GPS_velocity_ENU = circshift(GPS_velocity_ENU, -numRVpackets);
                GPS_velocity_valid = circshift(GPS_velocity_valid, -numRVpackets);
                GPS_numSats = circshift(GPS_numSats, -numRVpackets);
                
                % extract GPS information
                GPS_rV_packets(1:numRVpackets,:) = reshape(GPS_stream(GPS_rV' + (0:rV_LENGTH-1)),numRVpackets,rV_LENGTH);
                GPS_position_valid(maxNumPoints-numRVpackets+1:maxNumPoints) = logical(bitand(GPS_rV_packets(1:numRVpackets,16),128));
                GPS_velocity_valid(maxNumPoints-numRVpackets+1:maxNumPoints) = logical(bitand(GPS_rV_packets(1:numRVpackets,15),128));
                
                GPS_position_ecef(1:numRVpackets,1:3) = (2^32*GPS_rV_packets(1:numRVpackets,[13,21,29]) + 2^24*GPS_rV_packets(1:numRVpackets,[12,20,28]) + 2^16*GPS_rV_packets(1:numRVpackets,[11,19,27]) + 2^8*GPS_rV_packets(1:numRVpackets,[10,18,26]) + GPS_rV_packets(1:numRVpackets,[17,25,33]) - (GPS_rV_packets(1:numRVpackets,[13,21,29]) >= 128)*2^40) / 10000; % meters signed (ECEF)
                [lat,lon,alt] = ecef2geodetic(wgs84,GPS_position_ecef(1:numRVpackets,1),GPS_position_ecef(1:numRVpackets,2),GPS_position_ecef(1:numRVpackets,3));
                GPS_position_geodetic(maxNumPoints-numRVpackets+1:maxNumPoints,1:3) = [lat,lon,alt/1000]; % [lat] = [lon] = degrees, [alt/1000] = km
                
                GPS_velocity_ecef(1:numRVpackets,1:3) = (2^20*GPS_rV_packets(1:numRVpackets,[37,41,45]) + 2^12*GPS_rV_packets(1:numRVpackets,[36,40,44]) + 2^4*GPS_rV_packets(1:numRVpackets,[35,39,43]) + bitand(GPS_rV_packets(1:numRVpackets,[34,38,42]),double([0xF0,0xF0,0xF0])) - (GPS_rV_packets(1:numRVpackets,[37,41,45]) >= 128)*2^28) / 10000; % meters/s signed (ECEF)
                [vEast,vNorth,vUp] = ecef2enuv(GPS_velocity_ecef(1:numRVpackets,1),GPS_velocity_ecef(1:numRVpackets,2),GPS_velocity_ecef(1:numRVpackets,3),lat,lon);
                GPS_velocity_ENU(maxNumPoints-numRVpackets+1:maxNumPoints,1:3) = [vEast,vNorth,vUp];
                
                GPS_numSats(maxNumPoints-numRVpackets+1:maxNumPoints) = bitand(GPS_rV_packets(1:numRVpackets,16),31); % 0b 0001 1111 = 31 (decimal)
            end
        end
        % end GPS
        
        
        if (toc(plot_tic) >= plot_delay)
            plot_tic = tic;
            
            if (plotfigure1)
                % Figure 1
                set(mNLP_1_line,'YData',DATA_mNLP_Y(:,1));
                set(mNLP_2_line,'YData',DATA_mNLP_Y(:,2));
                set(mNLP_3_line,'YData',DATA_mNLP_Y(:,3));
                set(PIP_1_line,'YData',DATA_PIP_Y(:,1));
                set(PIP_2_line,'YData',DATA_PIP_Y(:,2));
                set(ACC_1_line,'YData',DATA_ACC_Y(:,1));
                set(ACC_2_line,'YData',DATA_ACC_Y(:,2));
                set(ACC_3_line,'YData',DATA_ACC_Y(:,3));
                set(DIG_ACC_1_line,'YData',DATA_DIG_ACC_Y(:,1));
                set(DIG_ACC_2_line,'YData',DATA_DIG_ACC_Y(:,2));
                set(DIG_ACC_3_line,'YData',DATA_DIG_ACC_Y(:,3));
                set(D1_line,'YData',DATA_D_Y(:,1));
                set(D2_line,'YData',DATA_D_Y(:,2));
            end
            if (plotfigure2)
                if(exist("latlim","var") && exist("lonlim","var") && exist("ZA","var"))
                    set(GPS_map2D_line,'XData',GPS_position_geodetic(1:maxNumPoints,2),'YData',GPS_position_geodetic(1:maxNumPoints,1));
                    set(GPS_map3D_line,'XData',GPS_position_geodetic(1:maxNumPoints,2),'YData',GPS_position_geodetic(1:maxNumPoints,1),'ZData',GPS_position_geodetic(1:maxNumPoints,3));
                end
            end
            
            % Housekeeping parsing - must find the BOARD ID to locate beginning of
            % HK loop. Must do this process for each box. Note that this approach
            % forgoes remainder buffering at the expense of dropping some
            % HK measurements.
            if (HKparsing)
                % mNLP
                mNLP_HK_index = strfind(buffer_mNLP_HK',mNLP_ID);
                mNLP_HK_index = mNLP_HK_index(diff(mNLP_HK_index) == HK_LENGTH);
                if ~isempty(mNLP_HK_index)
                    % Extract all valid HK data
                    ii = 1:length(mNLP_HK_index);
                    DATA_mNLP_HK(ii,:) = buffer_mNLP_HK(mNLP_HK_index' + (0:HK_LENGTH-1));
                    % Add HK data to plotting buffer
                    DATA_mNLP_HK_Y = circshift(DATA_mNLP_HK_Y,-length(mNLP_HK_index));
                    DATA_mNLP_HK_Y((maxNumPoints-length(mNLP_HK_index)+1):maxNumPoints,:) = DATA_mNLP_HK(ii,:);
                end
                
                % PIP
                PIP_HK_index = strfind(buffer_PIP_HK',PIP_ID);
                PIP_HK_index = PIP_HK_index(diff(PIP_HK_index) == HK_LENGTH);
                if ~isempty(PIP_HK_index)
                    % Extract all valid HK data
                    ii = 1:length(PIP_HK_index);
                    DATA_PIP_HK(ii,:) = buffer_PIP_HK(PIP_HK_index' + (0:HK_LENGTH-1));
                    % Add HK data to plotting buffer
                    DATA_PIP_HK_Y = circshift(DATA_PIP_HK_Y,-length(PIP_HK_index));
                    DATA_PIP_HK_Y((maxNumPoints-length(PIP_HK_index)+1):maxNumPoints,:) = DATA_PIP_HK(ii,:);
                end
                
                % ACC
                ACC_HK_index = strfind(buffer_ACC_HK',ACC_ID);
                ACC_HK_index = ACC_HK_index(diff(ACC_HK_index) == HK_LENGTH);
                if ~isempty(ACC_HK_index)
                    % Extract all valid HK data
                    ii = 1:length(ACC_HK_index);
                    DATA_ACC_HK(ii,:) = buffer_ACC_HK(ACC_HK_index' + (0:HK_LENGTH-1));
                    % Add HK data to plotting buffer
                    DATA_ACC_HK_Y = circshift(DATA_ACC_HK_Y,-length(ACC_HK_index));
                    DATA_ACC_HK_Y((maxNumPoints-length(ACC_HK_index)+1):maxNumPoints,:) = DATA_ACC_HK(ii,:);
                end
                
                % Convert HK counts to temperatures and voltages
                if HK_TV
                    DATA_mNLP_HK_Y((maxNumPoints-length(mNLP_HK_index)+1):maxNumPoints,2:5) = -76.9231* (DATA_mNLP_HK(1:length(mNLP_HK_index),2:5) *2.5/256 - 0.5*2.5/256) + 202.54;
                    DATA_mNLP_HK_Y((maxNumPoints-length(mNLP_HK_index)+1):maxNumPoints,6:11) = [16,6.15,7.329,3,2,2].*(DATA_mNLP_HK(1:length(mNLP_HK_index),6:11) *2.5/256 - 0.5*2.5/256) + [0,-16.88,0,0,0,0];
                    DATA_PIP_HK_Y((maxNumPoints-length(PIP_HK_index)+1):maxNumPoints,2:5) = -76.9231* (DATA_PIP_HK(1:length(PIP_HK_index),2:5) *2.5/256 - 0.5*2.5/256) + 202.54;
                    DATA_PIP_HK_Y((maxNumPoints-length(PIP_HK_index)+1):maxNumPoints,6:11) = [16,6.15,7.329,3,2,2].*(DATA_PIP_HK(1:length(PIP_HK_index),6:11) *2.5/256 - 0.5*2.5/256) + [0,-16.88,0,0,0,0];
                    DATA_ACC_HK_Y((maxNumPoints-length(ACC_HK_index)+1):maxNumPoints,2:5) = -76.9231* (DATA_ACC_HK(1:length(ACC_HK_index),2:5) *2.5/256 - 0.5*2.5/256) + 202.54;
                    DATA_ACC_HK_Y((maxNumPoints-length(ACC_HK_index)+1):maxNumPoints,6:11) = [16,6.15,7.329,3,2,2].*(DATA_ACC_HK(1:length(ACC_HK_index),6:11) *2.5/256 - 0.5*2.5/256) + [0,-16.88,0,0,0,0];
                end
                
                % end HK parsing
            end
            
            drawnow
        end

        % Don't deliberately pause if reading live data. Want code to run
        % as fast as possible. This pause is only to simulate 'real-time'
        % reading
        if (readmode == 0)
            pause(plot_delay);
        end
    end
    
    if (readmode == 1)
        % Wait here for at most 5 seconds. If no data comes through, exit
        exit_tic = tic;
        while ((u.NumBytesAvailable < remainderPtr-1) && (toc(exit_tic) < 5))
            % do nothing
        end
        if ~(toc(exit_tic) < 5)
            read_flag = false;
        end
    elseif (readmode == 0)
        if feof(fid)
            read_flag = false;
            fclose(fid);
        end
    end
    
    % end main loop
end

toc
